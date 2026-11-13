"""Per-fold view-flight datasets: four blocked time folds, broadcast and static cameras 01-05.

gt: the dataset's track and cameras. Video inputs (real: published heads; heads: fine-tuned heads,
one extraction folder per fold with --head-map): the candidate YOLO boxes of extract_video.py, the
Track-CV tracker (a global constant-velocity dynamic programme) tuned on each fold's training
flights, PnLCalib MV cameras, and the static 180-degree turn decided on the other half.
Every input gets the annotated contact frames; drop_contacts.py removes every contact-derived input
from the video datasets before evaluation.
Output: outputs/folds/<name>/fold_<k>/{train,val,test}.npz, flights.csv, build.json.
Run in the ball3d env: scripts/py scripts/build_folds.py gt [--workers 14]
"""
import argparse
import itertools
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as Config
from video_inputs import (DEFAULT_TRACKER, TRACKER_GRID, TUNE_HIT_PX, build_flight, candidates,
                           choose_flip, fill_track, load_cameras, static_camera, track)

VIEW_NAMES = ("broadcast", *[f"camera{c:02d}" for c in range(1, 7)])
HALF_TEST = {"half_1": ("b", "test"), "half_2": ("a", "test")}   # each half is a test split once
# Arrays every view-flight carries, from the frozen dataset.
SHARED = ("half", "view", "group", "a", "n", "image_h", "xyz", "uv_gt", "visible",
          "oracle_params", "oracle_rmse", "drag_rmse", "start_class")


# ---------------------------------------------------------------- Flights and folds


def assign_blocks(flights: pd.DataFrame) -> pd.DataFrame:
    """Per half, cut the time-ordered flights at the >= 50-frame gap closest to the middle."""
    flights = flights.copy()
    flights["block"] = ""
    for half, letter in zip(Config.HALVES, "AB"):
        part = flights[flights["half"] == half].sort_values("broadcast_a")
        a, b = part["broadcast_a"].to_numpy(), part["broadcast_b"].to_numpy()
        cuts = [i for i in range(len(part) - 1) if a[i + 1] - b[i] >= Config.BLOCK_GAP_FRAMES]
        cut = min(cuts, key=lambda i: abs(i + 0.5 - len(part) / 2))
        early = part.index[:cut + 1]
        flights.loc[part.index, "block"] = f"{letter}2"
        flights.loc[early, "block"] = f"{letter}1"
    return flights


def split_of(block: str, fold: str) -> str:
    test, val, train = Config.FOLDS[fold]
    if block == test:
        return "test"
    if block == val:
        return "val"
    return "train" if block in train else "drop"


def load_view_flights() -> dict:
    """Every view-flight of broadcast and cameras 01-05 once, from the half splits' test sets."""
    parts = []
    for half, (fold, split) in HALF_TEST.items():
        with np.load(Config.FLIGHTS_DIR / f"fold_{fold}" / f"{split}.npz") as archive:
            keep = np.isin(archive["view"], Config.VIEWS)
            parts.append({k: archive[k][keep] for k in archive.files})
    data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    order = np.lexsort((data["group"], data["view"]))
    return {k: v[order] for k, v in data.items()}


# ---------------------------------------------------------------- Oracle contacts


def ray_at_height(uv: np.ndarray, rot: np.ndarray, trans: np.ndarray, intr: np.ndarray,
                  height: float) -> np.ndarray:
    """Pitch xy (m) where the ray through one pixel (1920-wide units) reaches a height (m)."""
    direction = rot.T @ np.array([(uv[0] - intr[2]) / intr[0], (uv[1] - intr[3]) / intr[1], 1.0])
    direction = direction * [1.0, 1.0, -1.0]
    centre = -(rot.T @ trans) * [1.0, 1.0, -1.0]
    if direction[2] >= 0:
        return np.full(2, np.nan)
    along = (height - centre[2]) / direction[2]
    if not 0 < along <= Config.MAX_RAY_M:
        return np.full(2, np.nan)
    return (centre + along * direction)[:2]


def contact_height(xyz: np.ndarray, frame: int, step: int) -> float:
    """GT ball-centre height at a contact frame, or the nearest GT inside the flight."""
    for k in range(Config.CONTACT_SEARCH_FRAMES + 1):
        z = xyz[frame + step * k, 2]
        if np.isfinite(z):
            return float(z)
    return float("nan")


def oracle_contacts(n: int, xyz: np.ndarray, uv: np.ndarray, rot: np.ndarray, trans: np.ndarray,
                    intr: np.ndarray, size: int) -> dict:
    """Anchors, types and GT heights at the start and end contact of one view-flight."""
    pitch = np.full((size, 2), np.nan)
    out = {}
    for name, frame, step in (("start", 0, 1), ("end", n - 1, -1)):
        z = contact_height(xyz, frame, step)
        out[f"z_{name}"] = z
        out[f"{name}_high"] = bool(np.isfinite(z) and z >= Config.HIGH_CONTACT_M)
        out[f"{name}_anchor_shift"] = -1
        if not np.isfinite(z):
            continue
        for k in range(Config.ANCHOR_SEARCH_FRAMES + 1):
            f = frame + step * k
            if np.isfinite(uv[f]).all():
                hit = ray_at_height(uv[f], rot[f], trans[f], intr[f],
                                    max(z - Config.BALL_RADIUS_M, 0.0))
                if np.isfinite(hit).all():
                    pitch[frame] = hit
                    out[f"{name}_anchor_shift"] = k
                break
    out["pitch_xy"] = pitch
    return out


# ---------------------------------------------------------------- Scenario inputs


def gt_inputs(data: dict) -> dict:
    """ball3d's shipped track as the ball's own bottom pixel, and its box diameter, at 1920 wide.

    The shipped `yk` is replaced by the nearest player's foot point at ball3d's high pivots; with
    oracle contacts that correction is not needed, so `yk_original + b` (the ball bottom) is used.
    """
    uv = np.full(data["uv_det"].shape, np.nan, np.float32)
    size = np.full(data["uv_det"].shape[:2], np.nan, np.float32)
    corrected = np.zeros(data["uv_det"].shape[:2], bool)
    for view in np.unique(data["view"]):
        for half in np.unique(data["half"]):
            chosen = np.where((data["view"] == view) & (data["half"] == half))[0]
            if not len(chosen):
                continue
            base = Config.DATA_ROOT / Config.LP / VIEW_NAMES[view] / Config.HALVES[half]
            meta = json.loads((base / "sequence_metadata.json").read_text())
            table = pd.read_csv(base / "track/ball_3d.basic_fitg.csv", low_memory=False,
                                usecols=["file_name", "xk", "yk", "yk_original", "b", "size"])
            table = table.drop_duplicates("file_name").set_index("file_name")
            scale = Config.IMAGE_WIDTH_PX / float(meta["width"])
            for i in chosen:
                n = int(data["n"][i])
                rows = table.reindex(int(data["a"][i]) + np.arange(n))
                bottom = rows["yk_original"] + rows["b"]
                uv[i, :n, 0] = rows["xk"].to_numpy(float) * scale
                uv[i, :n, 1] = bottom.to_numpy(float) * scale
                size[i, :n] = rows["size"].to_numpy(float) * scale / np.sqrt(2.0)
                corrected[i, :n] = (rows["yk"] - bottom).abs().to_numpy() >= 0.5
    size[~np.isfinite(uv[..., 0])] = np.nan
    return {"uv_det": uv, "rot": data["rot"], "trans": data["trans"], "intr": data["intr"],
            "size_det": size, "score_det": np.where(np.isfinite(uv[..., 0]), 1.0,
                                                    np.nan).astype(np.float32),
            "foot_corrected": corrected}


def unit_cameras(view: int, half: int, extract: Path) -> dict:
    """PnLCalib solves of one unit, in the pitch orientation chosen on the other half."""
    table = load_cameras(view, half, extract)
    other = load_cameras(view, 1 - half, extract)
    is_flipped, vote = choose_flip(other, view, 1 - half)
    table = table[table["flipped"] == is_flipped]
    unit = {"frames": table, "fallback": static_camera(table), "flipped": bool(is_flipped),
            "flip_vote_other_half": float(vote)}
    if view != 0:
        unit["static"] = static_camera(table)
    return unit


def unit_scores(params: dict, items: list[dict], boxes: dict) -> np.ndarray:
    """Hits, wrong frames and GT frames of one tracker setting on one unit's view-flights."""
    total = np.zeros(3)
    for item in items:
        frames = np.arange(item["a"], item["a"] + item["n"])
        path = fill_track(track(frames, boxes, **params))[0]
        gt = item["uv_gt"]
        seen = np.isfinite(gt[:, 0])
        gap = np.linalg.norm(path[:, [0, 2]] - gt, axis=1)
        total += [(gap[seen] < TUNE_HIT_PX).sum(), (gap[seen] >= TUNE_HIT_PX).sum(), seen.sum()]
    return total


def tune(items: list[dict], boxes: dict, workers: int) -> tuple[dict, pd.DataFrame]:
    """The tracker grid on one fold's train view-flights (same score as video_inputs.py)."""
    from joblib import Parallel, delayed
    grid = [dict(zip(TRACKER_GRID, v)) for v in itertools.product(*TRACKER_GRID.values())]
    units = sorted({(it["view"], it["half"]) for it in items})
    by_unit = {u: [it for it in items if (it["view"], it["half"]) == u] for u in units}
    jobs = [(g, u) for g in range(len(grid)) for u in units]
    done = Parallel(n_jobs=workers)(
        delayed(unit_scores)(grid[g], by_unit[u], boxes[u]) for g, u in jobs)
    totals = np.zeros((len(grid), 3))
    for (g, _), counts in zip(jobs, done):
        totals[g] += counts
    rows = [{**grid[g], "hit": t[0] / t[2], "wrong": t[1] / t[2], "score": (t[0] - t[1]) / t[2]}
            for g, t in enumerate(totals)]
    table = pd.DataFrame(rows).sort_values("score", ascending=False)
    return table.iloc[0][list(TRACKER_GRID)].to_dict(), table


def video_inputs(item: dict, boxes: dict, cams: dict, tracker: dict, size: int) -> dict:
    """Tracked ball, box diameter and score, and cameras of one view-flight (video_inputs)."""
    row = SimpleNamespace(a=item["a"], n=item["n"], view=item["view"], half=item["half"])
    key = (item["view"], item["half"])
    built = build_flight(row, {key: boxes}, {key: cams}, tracker, size, {}, ("track", "camera"))
    return {k: built[k] for k in ("uv_det", "rot", "trans", "intr", "size_det", "score_det",
                                  "rival_score_det", "filled", "cam_solved", "cam_rep_err")} | \
        {"has_camera": built["has_camera"]}


def unit_inputs(items: list[tuple[int, dict]], boxes: dict, cams: dict, tracker: dict, size: int
                ) -> list[tuple[int, dict]]:
    return [(i, video_inputs(it, boxes, cams, tracker, size)) for i, it in items]


# ---------------------------------------------------------------- Writing


def write_fold(out: Path, fold: str, data: dict, inputs: dict, contacts: dict, flights: pd.DataFrame
               ) -> dict:
    """train / val / test archives of one fold; groups never cross splits."""
    block = flights.set_index("group")["block"]
    split = np.array([split_of(block[g], fold) for g in data["group"]])
    counts = {}
    for name in ("train", "val", "test"):
        keep = split == name
        arrays = {k: data[k][keep] for k in SHARED}
        arrays.update({k: v[keep] for k, v in inputs.items()})
        arrays.update({k: v[keep] for k, v in contacts.items()})
        (out / f"fold_{fold}").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / f"fold_{fold}" / f"{name}.npz", **arrays)
        counts[name] = {"view_flights": int(keep.sum()),
                        "flights": int(len(np.unique(data["group"][keep])))}
    groups = {name: set(data["group"][split == name]) for name in ("train", "val", "test")}
    if groups["train"] & groups["test"] or groups["val"] & groups["test"] or \
            groups["train"] & groups["val"]:
        raise RuntimeError(f"fold {fold}: a physical flight is in two splits")
    return counts


def contact_arrays(data: dict, inputs: dict) -> dict:
    size = data["xyz"].shape[1]
    rows = [oracle_contacts(int(data["n"][i]), data["xyz"][i], inputs["uv_det"][i],
                            inputs["rot"][i], inputs["trans"][i], inputs["intr"][i], size)
            for i in range(len(data["n"]))]
    out = {"pitch_xy": np.stack([r["pitch_xy"] for r in rows]).astype(np.float32)}
    for key in ("z_start", "z_end"):
        out[key] = np.array([r[key] for r in rows], np.float32)
    for key in ("start_high", "end_high"):
        out[key] = np.array([r[key] for r in rows])
    for key in ("start_anchor_shift", "end_anchor_shift"):
        out[key] = np.array([r[key] for r in rows], np.int8)
    return out


def anchor_report(data: dict, contacts: dict) -> dict:
    """Median horizontal anchor error against the GT ball (m), by contact type and view."""
    rows = []
    for i in range(len(data["n"])):
        n = int(data["n"][i])
        for name, frame in (("start", 0), ("end", n - 1)):
            gt = data["xyz"][i, frame, :2]
            hit = contacts["pitch_xy"][i, frame]
            rows.append({"view": int(data["view"][i]), "contact": name,
                         "high": bool(contacts[f"{name}_high"][i]),
                         "has_anchor": bool(np.isfinite(hit).all()),
                         "error_m": float(np.linalg.norm(hit - gt))})
    table = pd.DataFrame(rows)
    summary = table.groupby(["view", "high"]).agg(
        contacts=("error_m", "size"), anchored=("has_anchor", "mean"),
        median_error_m=("error_m", "median")).reset_index()
    return {"overall_anchored": float(table["has_anchor"].mean()),
            "median_error_m": float(table["error_m"].median()),
            "by_view_type": summary.round(3).to_dict(orient="records")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", help="gt, or a name for a video build (real, heads)")
    parser.add_argument("--extract", type=Path, default=Config.EXTRACT_DIR / "published",
                        help="extraction folder for video inputs")
    parser.add_argument("--head-map", type=Path, default=None,
                        help="JSON {fold: extraction folder}: per-fold fine-tuned heads")
    parser.add_argument("--views", nargs="+", type=int, default=list(Config.VIEWS))
    parser.add_argument("--folds", nargs="+", default=sorted(Config.FOLDS))
    parser.add_argument("--tracker-json", type=Path, default=None,
                        help="build.json whose per-fold tracker to reuse (no tuning)")
    parser.add_argument("--workers", type=int, default=14)
    args = parser.parse_args()
    out = Config.FOLDS_DIR / args.scenario
    if (out / "build.json").exists():
        raise FileExistsError(f"{out} exists; move it away first")
    out.mkdir(parents=True, exist_ok=True)

    flights = assign_blocks(pd.read_csv(Config.FLIGHTS_DIR / "flights.csv"))
    flights.to_csv(out / "flights.csv", index=False)
    data = load_view_flights()
    keep = np.isin(data["view"], args.views)
    data = {k: v[keep] for k, v in data.items()}
    report = {"scenario": args.scenario, "view_flights": int(len(data["n"])),
              "blocks": flights["block"].value_counts().sort_index().to_dict(),
              "views": args.views, "folds": {}}
    print(report, flush=True)

    if args.scenario == "gt":
        inputs = gt_inputs(data)
        contacts = contact_arrays(data, inputs)
        report["anchors"] = anchor_report(data, contacts)
        for fold in Config.FOLDS:
            report["folds"][fold] = write_fold(out, fold, data, inputs, contacts, flights)
    else:
        from joblib import Parallel, delayed
        head_map = json.loads(args.head_map.read_text()) if args.head_map else {}
        size = data["xyz"].shape[1]
        report["extract"] = str(args.extract)
        report["head_map"] = head_map
        report["tracker_json"] = str(args.tracker_json) if args.tracker_json else None
        for fold in args.folds:
            extract = Path(head_map.get(fold, args.extract))
            units = sorted(set(zip(data["view"].tolist(), data["half"].tolist())))
            boxes = {u: candidates(*u, extract)[0] for u in units}
            cams = {u: unit_cameras(*u, extract) for u in units}
            block = flights.set_index("group")["block"]
            split = np.array([split_of(block[g], fold) for g in data["group"]])
            items = [{"a": int(data["a"][i]), "n": int(data["n"][i]), "view": int(data["view"][i]),
                      "half": int(data["half"][i]),
                      "uv_gt": np.where(data["visible"][i, :data["n"][i], None],
                                        data["uv_gt"][i, :data["n"][i]], np.nan)}
                     for i in range(len(data["n"]))]
            train = [items[i] for i in np.where(split == "train")[0]]
            if args.tracker_json is not None:
                fixed = json.loads(args.tracker_json.read_text())["folds"]
                tracker = fixed[fold]["tracker"]
            else:
                tracker, grid = tune(train, boxes, args.workers)
                grid.to_csv(out / f"tracker_grid_fold_{fold}.csv", index=False)
            by_unit = {u: [(i, it) for i, it in enumerate(items)
                           if (it["view"], it["half"]) == u] for u in units}
            chunks = [(u, part[k:k + 40]) for u, part in by_unit.items()
                      for k in range(0, len(part), 40)]
            done = Parallel(n_jobs=args.workers)(
                delayed(unit_inputs)(chunk, boxes[u], cams[u], tracker, size)
                for u, chunk in chunks)
            built = [None] * len(items)
            for chunk in done:
                for i, b in chunk:
                    built[i] = b
            inputs = {k: np.stack([b[k] for b in built]) for k in built[0] if k != "has_camera"}
            inputs = {k: v.astype(np.float32) if v.dtype.kind == "f" else v
                      for k, v in inputs.items()}
            inputs["has_camera"] = np.array([b["has_camera"] for b in built])
            contacts = contact_arrays(data, inputs)
            report["folds"][fold] = {
                "extract": str(extract), "tracker": tracker,
                "flips": {f"{VIEW_NAMES[v]}_{Config.HALVES[h]}":
                          [c["flipped"], round(c["flip_vote_other_half"], 3)]
                          for (v, h), c in cams.items()},
                "anchors": anchor_report(data, contacts),
                "splits": write_fold(out, fold, data, inputs, contacts, flights)}
            print(fold, json.dumps(report["folds"][fold]["tracker"]),
                  report["folds"][fold]["anchors"]["median_error_m"], flush=True)
    report["default_tracker"] = DEFAULT_TRACKER
    (out / "build.json").write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps({k: v for k, v in report.items() if k != "folds"}, indent=1, default=float))


if __name__ == "__main__":
    main()
