"""Demo test sets: ten flights seen by every view, every combination of contact heights per end.

Flights: physical test flights seen by the broadcast and all five static cameras, nearest the
10th, 20th, ..., 100th percentile of ArcPrior Base's broadcast error with true heights and clean
tracks (the Table 1 set). Each view-flight is repeated for the nine (start, end) height pairs, each
end true (the GT height), off (GT + 0.5 m, or - 0.5 m above 2.7 m) or unknown, on the clean
track; anchors are cut as in build_test_sets.py. Row group = flight * 10 + pair, so every pair is
its own flight for predict.py and run_grad.py; `flight` and `pair` keep the originals.
Output: <data-root>/demo_<setting>/fold_<k>/test.npz and demo_flights.json.
Run in the snv3d env: scripts/py --env snv3d scripts/build_demo_sets.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as Config  # noqa: E402
from arcprior.contacts import CONTACT_MAX_M, contact_anchors, fill_short_gaps  # noqa: E402
from arcprior.metrics import e_full  # noqa: E402
from arcprior.rays import rays_from_pixels  # noqa: E402
from build_test_sets import dev_anchors, ground_anchors  # noqa: E402

LEVELS = ("true", "off", "none")
OFF_M = 0.5
PERCENTILES = tuple(range(10, 101, 10))
VIEW_COUNT = len(Config.VIEWS)


def load(root: Path, name: str, fold: str) -> dict:
    with np.load(root / name / f"fold_{fold}" / "test.npz") as z:
        return {k: z[k] for k in z.files}


def choose_flights(args: argparse.Namespace) -> list[dict]:
    """Six-view test flights at fixed percentiles of ArcPrior Base's broadcast error."""
    freeze = json.loads(args.freeze.read_text())
    choice = freeze["choices"][args.rank_choice]
    parts = []
    for fold in Config.FOLDS:
        for seed in range(3):
            folder = args.pred_root / args.rank_set / f"base_s{seed}"
            table = pd.read_csv(folder / f"fold_{fold}_test_flights.csv.gz")
            parts.append(table[table["method"] == choice[fold]].assign(fold=fold))
    keys = ["view", "half", "group", "share", "fold"]
    table = pd.concat(parts).groupby(keys)["rmse_cap"].mean().reset_index().assign(method="base")
    error = e_full(table, "base", "broadcast")
    views = table.groupby("group")["view"].nunique()
    folds = table.groupby("group")["fold"].first()
    halves = table.groupby("group")["half"].first()
    pool = error[views.reindex(error.index) == VIEW_COUNT].sort_values()
    rank = pool.rank(pct=True) * 100
    chosen = []
    for q in PERCENTILES:
        for group in (rank - q).abs().sort_values().index:
            if group not in [c["flight"] for c in chosen]:
                chosen.append({"flight": int(group), "percentile": q, "fold": str(folds[group]),
                               "half": int(halves[group]), "error_m": round(float(pool[group]), 3)})
                break
    return chosen


def heights(z: np.ndarray, level: str) -> tuple[np.ndarray, np.ndarray]:
    if level == "none":
        return np.zeros_like(z), np.zeros(len(z), bool)
    if level == "true":
        return z.astype(np.float32), np.isfinite(z)
    off = np.where(z + OFF_M <= CONTACT_MAX_M, z + OFF_M, z - OFF_M)
    return np.clip(off, 0.0, CONTACT_MAX_M).astype(np.float32), np.isfinite(z)


def demo_set(arrays: dict, rows: np.ndarray, anchors_kind: str, points: dict | None) -> dict:
    base = {k: v[rows] for k, v in arrays.items()}
    pairs = [(s, e) for s in LEVELS for e in LEVELS]
    out = {k: np.concatenate([v] * len(pairs)) for k, v in base.items()}
    count = len(rows)
    pair = np.repeat(np.arange(len(pairs)), count)
    out["flight"] = out["group"].copy()
    out["pair"] = pair.astype(np.int8)
    out["group"] = (out["flight"] * 10 + pair).astype(out["flight"].dtype)
    n = torch.as_tensor(out["n"]).long()
    uv, filled = fill_short_gaps(torch.as_tensor(out["uv_det"]).float(), n)
    contact = {}
    for end, index in (("start", 0), ("end", 1)):
        level = np.array([pairs[p][index] for p in pair])
        value = np.zeros(len(pair), np.float32)
        given = np.zeros(len(pair), bool)
        for name in LEVELS:
            mask = level == name
            v, g = heights(out[f"z_{end}"][mask], name)
            value[mask], given[mask] = v, g
        contact[f"h_{end}"], contact[f"{end}_given"] = value, given
        out[f"{end}_level"] = np.array([("true", "off", "none").index(x) for x in level], np.int8)
    tensors = {k: torch.as_tensor(v) for k, v in contact.items()}
    frames = np.arange(out["xyz"].shape[1])[None]
    present = torch.isfinite(uv[..., 0]) & (torch.as_tensor(frames) < n[:, None])
    rot, trans, intr = (torch.as_tensor(out[k]).float() for k in ("rot", "trans", "intr"))
    anchors = contact_anchors(uv, rot, trans, intr, present, n, tensors)
    pitch = np.full(out["pitch_xy"].shape, np.nan, dtype=np.float32)
    r = np.arange(len(pair))
    pitch[r, 0] = anchors["start_xy"].numpy()
    pitch[r, out["n"] - 1] = anchors["end_xy"].numpy()
    first = anchors["start_frame"].clamp(min=0)
    _, ray = rays_from_pixels(uv[r, first].nan_to_num(), rot[r, first], trans[r, first],
                              intr[r, first])
    out.update(uv_det=uv.numpy().astype(np.float32), filled=filled.numpy(), pitch_xy=pitch,
               start_slope=(ray[:, :2].norm(dim=1) / ray[:, 2].abs().clamp(min=1e-3)).numpy(),
               start_given=contact["start_given"] & anchors["has_start"].numpy(),
               end_given=contact["end_given"] & anchors["has_end"].numpy(),
               h_start=contact["h_start"], h_end=contact["h_end"],
               track_masked=np.zeros(len(pair), bool))
    out["pitch_xy_ball3d"] = (dev_anchors(out, points) if anchors_kind == "dev"
                              else ground_anchors(out))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Config.EVAL_DATA)
    parser.add_argument("--pred-root", type=Path, default=Config.EVAL_ROOT / "pred")
    parser.add_argument("--freeze", type=Path, default=Config.EVAL_ROOT / "freeze.json")
    parser.add_argument("--sources", nargs="+", default=["gt=gt_src", "real=real_src",
                                                         "heads=heads_src"])
    parser.add_argument("--rank-set", default="gt_fixcorrect")
    parser.add_argument("--rank-choice", default="base_gt_full")
    args = parser.parse_args()
    chosen = choose_flights(args)
    for c in chosen:
        print(c)
    from dataset_contacts import rule_flags
    points = rule_flags()[1]
    for pair_arg in args.sources:
        setting, name = pair_arg.split("=")
        for fold in sorted({c["fold"] for c in chosen}):
            arrays = load(args.data_root, name, fold)
            groups = [c["flight"] for c in chosen if c["fold"] == fold]
            rows = np.where(np.isin(arrays["group"], groups))[0]
            built = demo_set(arrays, rows, "dev" if setting == "gt" else "ground", points)
            target = args.data_root / f"demo_{setting}" / f"fold_{fold}" / "test.npz"
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target, **built)
            print(setting, fold, len(built["n"]), "view-flight pairs")
    (args.data_root / "demo_flights.json").write_text(json.dumps(chosen, indent=1))


if __name__ == "__main__":
    main()
