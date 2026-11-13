"""Test sets per input setting: full and partial flights, mixed conditions, or one fixed level.

Every physical flight gets one balanced, seeded cell (track clean / masked; contact height correct /
wrong / missing per end; partial flights: start only), the same in every input setting
(arcprior.contacts.build_test_set). The train split is clean with correct heights. Masking draws
its run lengths from the fold's training video-track gaps. --fixed builds clean tracks with one
level at both ends (Table 1). `pitch_xy_ball3d` holds Grad et al.'s own contact points: their
released pitch points with GT inputs (--ball3d-anchors dev), or from video the ray of the first
and last detection cut at the ground (--ball3d-anchors ground).
Output: outputs/eval/data/<setting>_{full,part}/ or <setting>_fix<level>/, fold_<k>/{split}.npz.
Run in the snv3d env: scripts/py --env snv3d scripts/build_test_sets.py gt_src --name gt \
    --source-root outputs/eval/data --ball3d-anchors dev
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as Config
from arcprior.contacts import LEVELS, build_test_set, gap_table

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_folds import ray_at_height  # noqa: E402


def fold_gaps(fold: str) -> dict:
    with np.load(Config.FOLDS_DIR / f"real_{Config.TRACK_SOURCE[fold]}" / f"fold_{fold}" /
                 "train.npz") as video:
        return gap_table(video["uv_det"], video["n"], video["view"])


def ground_anchors(arrays: dict) -> np.ndarray:
    """Grad et al. from video: the rays of the flight's first and last detections cut at the
    ground (there are no player boxes)."""
    pitch = np.full(arrays["pitch_xy"].shape, np.nan, dtype=np.float32)
    for i in range(len(arrays["n"])):
        n = int(arrays["n"][i])
        seen = np.where(np.isfinite(arrays["uv_det"][i, :n, 0]))[0]
        if not len(seen):
            continue
        # The first / last detection is the nearest one to the contact frame.
        for row, frame in ((0, seen[0]), (n - 1, seen[-1])):
            hit = ray_at_height(arrays["uv_det"][i, frame].astype(float),
                                arrays["rot"][i, frame].astype(float),
                                arrays["trans"][i, frame].astype(float),
                                arrays["intr"][i, frame].astype(float), 0.0)
            pitch[i, row] = hit
    return pitch


def dev_anchors(arrays: dict, points: dict) -> np.ndarray:
    """Grad et al. with GT inputs: their released pitch points at the contact frames."""
    from dataset_contacts import ball3d_anchors
    return ball3d_anchors(arrays, points).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario", help="input dataset under --source-root")
    parser.add_argument("--kinds", nargs="+", default=["full", "part"])
    parser.add_argument("--source-root", type=Path, default=Config.FOLDS_DIR)
    parser.add_argument("--name", default=None, help="output stem (default: the scenario)")
    parser.add_argument("--ball3d-anchors", choices=("dev", "ground"), required=True)
    parser.add_argument("--fixed", choices=LEVELS, default=None,
                        help="clean tracks, this contact level at both ends (kind full only)")
    args = parser.parse_args()
    source = args.source_root / args.scenario
    points = None
    if args.ball3d_anchors == "dev":
        from dataset_contacts import rule_flags
        points = rule_flags()[1]
    for kind in args.kinds:
        stem = args.name or args.scenario
        out = Config.EVAL_DATA / (f"{stem}_fix{args.fixed}" if args.fixed else f"{stem}_{kind}")
        if (out / "build.json").exists():
            raise FileExistsError(out)
        report = {}
        for path in sorted(source.glob("fold_*/*.npz")):
            fold = path.parent.name.split("_")[1]
            with np.load(path) as archive:
                arrays = {k: archive[k] for k in archive.files}
            built = build_test_set(arrays, kind, path.stem, fold_gaps(fold), args.fixed)
            built["pitch_xy_ball3d"] = (dev_anchors(built, points) if points is not None
                                        else ground_anchors(built))
            target = out / path.parent.name / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target, **built)
            report[f"{path.parent.name}/{path.stem}"] = {
                "view_flights": int(len(built["n"])),
                "masked": float(built["track_masked"].mean()),
                "start_level": {lv: float((built["start_level"] == k).mean())
                                for k, lv in enumerate(LEVELS)},
                "end_level": {lv: float((built["end_level"] == k).mean())
                              for k, lv in enumerate(LEVELS)},
                "start_given": float(built["start_given"].mean()),
                "end_given": float(built["end_given"].mean()),
                "missing_share": float(1 - np.isfinite(built["uv_det"][..., 0]).sum() /
                                       built["n"].sum()),
                "filled_share": float(built["filled"].sum() / built["n"].sum())}
        shutil.copy(source / "flights.csv", out / "flights.csv")
        (out / "build.json").write_text(json.dumps(
            {"source": str(source), "kind": kind, "splits": report}, indent=1))
        print(kind, json.dumps(report["fold_1/test"]))


if __name__ == "__main__":
    main()
