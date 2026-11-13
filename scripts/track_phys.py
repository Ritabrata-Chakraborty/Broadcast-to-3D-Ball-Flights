"""Track-Phys: the tracker with physics in the loop, on a Track-CV track.

From the Track-CV track, fit LM-fitg, project the fit, take in each frame the candidate box whose
ball bottom is nearest the projection within R px (none if none is that close), and refit; two
rounds. R is tuned per fold on its training flights (frames within 40 px of the projected GT minus
frames further away), or taken from another build with --radius-from. With --no-contacts (input
from drop_contacts.py) the fits use no anchors, so no contact information enters the track.
Output: outputs/folds/<out>/ with the input's arrays and the new track.
Run in the snv3d env: scripts/py --env snv3d scripts/track_phys.py real_cv --out real_phys \
    --no-contacts
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as Config
from build_folds import contact_arrays
from predict import best_of, path_of, physics_starts
from arcprior.network import build_batch3, to_world3
from arcprior.rays import FlightSet
from arcprior.solver import LM_FITG_CFG
from video_inputs import TUNE_HIT_PX, candidates

RADII_PX = (10.0, 20.0, 40.0, 80.0)
ROUNDS = 2
FIT = LM_FITG_CFG


def project(world: np.ndarray, rot: np.ndarray, trans: np.ndarray, intr: np.ndarray) -> np.ndarray:
    x_cam = np.einsum("tij,tj->ti", rot, world * [1.0, 1.0, -1.0]) + trans
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = np.c_[intr[:, 0] * x_cam[:, 0] / x_cam[:, 2] + intr[:, 2],
                   intr[:, 1] * x_cam[:, 1] / x_cam[:, 2] + intr[:, 3]]
    uv[x_cam[:, 2] <= 0] = np.nan
    return uv


@torch.no_grad()
def fitted_pixels(arrays: dict, root_tmp: Path, fold: str, split: str,
                  has_contacts: bool = True) -> np.ndarray:
    """lm_fitg fit of every view-flight on the current track, projected to its frames (1920)."""
    path_npz = root_tmp / f"fold_{fold}" / f"{split}.npz"
    path_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path_npz, **arrays)
    flights = FlightSet(fold, split, "cpu", root=root_tmp)
    size = arrays["uv_det"].shape[1]
    out = np.full((flights.size, size, 2), np.nan)
    for begin in range(0, flights.size, 256):
        index = torch.arange(begin, min(begin + 256, flights.size))
        batch = build_batch3(flights, index)
        theta, _ = best_of(physics_starts(batch), batch, FIT)
        world = to_world3(path_of(theta, batch.n_frames), batch).numpy().astype(float)
        for r, i in enumerate(index.tolist()):
            n = int(arrays["n"][i])
            ok = bool(batch.has_obs[r] & (batch.has_pitch_start[r] | (not has_contacts)))
            if ok:
                out[i, :n] = project(world[r, :n], arrays["rot"][i, :n].astype(float),
                                     arrays["trans"][i, :n].astype(float),
                                     arrays["intr"][i, :n].astype(float))
    return out


def reassign(arrays: dict, guide: np.ndarray, boxes: dict, radius: float) -> dict:
    """New track: per frame the candidate nearest the projected fit within radius (px, 1920)."""
    uv = np.full(arrays["uv_det"].shape, np.nan, np.float32)
    size = np.full(arrays["size_det"].shape, np.nan, np.float32)
    score = np.full(arrays["score_det"].shape, np.nan, np.float32)
    for i in range(len(arrays["n"])):
        n = int(arrays["n"][i])
        unit = boxes[(int(arrays["view"][i]), int(arrays["half"][i]))]
        for t in range(n):
            target = guide[i, t]
            options = unit.get(int(arrays["a"][i]) + t)
            if options is None or not np.isfinite(target).all():
                if not np.isfinite(target).all():
                    uv[i, t] = arrays["uv_det"][i, t]
                    size[i, t] = arrays["size_det"][i, t]
                    score[i, t] = arrays["score_det"][i, t]
                continue
            gap = np.hypot(options[:, 0] - target[0], options[:, 1] - target[1])
            k = int(np.argmin(gap))
            if gap[k] <= radius:
                uv[i, t] = options[k, :2]
                size[i, t] = options[k, 3]
                score[i, t] = options[k, 4]
    return {"uv_det": uv, "size_det": size, "score_det": score}


def track_score(arrays: dict, uv: np.ndarray) -> float:
    """(hits - wrong) / GT frames, the Track-CV tuning score (projected GT, centre ~ bottom)."""
    hits = wrong = total = 0
    for i in range(len(arrays["n"])):
        n = int(arrays["n"][i])
        gt = np.where(arrays["visible"][i, :n, None], arrays["uv_gt"][i, :n], np.nan)
        seen = np.isfinite(gt[:, 0])
        gap = np.linalg.norm(uv[i, :n] - gt, axis=1)
        hits += int((gap[seen] < TUNE_HIT_PX).sum())
        wrong += int((gap[seen] >= TUNE_HIT_PX).sum())
        total += int(seen.sum())
    return (hits - wrong) / max(total, 1)


def run_t5(arrays: dict, boxes: dict, radius: float, tmp: Path, fold: str, split: str,
           has_contacts: bool = True) -> dict:
    current = dict(arrays)
    for _ in range(ROUNDS):
        guide = fitted_pixels(current, tmp, fold, split, has_contacts)
        current.update(reassign(current, guide, boxes, radius))
        if has_contacts:
            current.update(contact_arrays(current, current))
    return current


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", help="input built by build_folds.py (Track-CV tracks)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--radius", type=float, default=None, help="fixed radius (px), no tuning")
    parser.add_argument("--radius-from", default=None,
                        help="use the per-fold radii of this Track-Phys build (no tuning)")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    parser.add_argument("--folds", nargs="+", default=sorted(Config.FOLDS))
    parser.add_argument("--no-contacts", action="store_true",
                        help="input built by drop_contacts.py: fit without anchors")
    args = parser.parse_args()
    has_contacts = not args.no_contacts
    torch.set_num_threads(args.threads)
    source = Config.FOLDS_DIR / args.scenario
    out = Config.FOLDS_DIR / args.out
    if (out / "build.json").exists():
        raise FileExistsError(f"{out} exists")
    tmp = out / "tmp"
    built = json.loads((source / "build.json").read_text())
    report = {"source": args.scenario, "radii": RADII_PX, "rounds": ROUNDS,
              "no_contacts": args.no_contacts, "folds": {}}
    if args.no_contacts:
        built = built["source_build"]
    for fold in args.folds:
        extract = Path(built["folds"][fold]["extract"])
        data = {}
        for split in ("train", "val", "test"):
            with np.load(source / f"fold_{fold}" / f"{split}.npz") as z:
                data[split] = {k: z[k] for k in z.files}
        units = sorted({(v, h) for split in data.values()
                        for v, h in zip(split["view"].tolist(), split["half"].tolist())})
        boxes = {u: candidates(*u, extract)[0] for u in units}
        grid = {}
        if args.radius_from is not None:
            other = json.loads((Config.FOLDS_DIR / args.radius_from / "build.json").read_text())
            radius = float(other["folds"][fold]["radius"])
        elif args.radius is None:
            for radius in RADII_PX:
                new = run_t5(data["train"], boxes, radius, tmp / f"r{radius}", fold, "train",
                             has_contacts)
                grid[radius] = track_score(data["train"], new["uv_det"])
            grid["t4"] = track_score(data["train"], data["train"]["uv_det"])
            radius = max(RADII_PX, key=grid.get)
        else:
            radius = args.radius
        report["folds"][fold] = {"grid_train": grid, "radius": radius}
        print(fold, grid, radius, flush=True)
        for split in args.splits:
            new = run_t5(data[split], boxes, radius, tmp / "final", fold, split, has_contacts)
            target = out / f"fold_{fold}" / f"{split}.npz"
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target, **new)
    shutil.rmtree(tmp)
    shutil.copy(source / "flights.csv", out / "flights.csv")
    (out / "build.json").write_text(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
