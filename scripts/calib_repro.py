"""Reproduction gate: PnLCalib at its published settings against its published numbers.

Also runs the official evaluator on one dataset, to check that this project's scorer agrees.
Output: outputs/calibration/reproduction.csv. Exits non-zero if a check misses by over 2 points.
"""
import argparse
import gzip
import json
import sys
import zipfile
from multiprocessing import get_context
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calib_common as Common  # noqa: E402
import calib_eval as Eval  # noqa: E402
import calib_homography as Hom  # noqa: E402
import calib_solve as Solve  # noqa: E402
from calib_common import Config  # noqa: E402
from evalai_camera import evaluate  # noqa: E402

MAX_GAP = 2.0
CAMERA_CHECKS = (
    ("SN22-test-center", "pnlcalib", 0.0966, 0.3441, 100.0,
     {"jac5": 80.6, "jac10": 91.6, "jac20": 93.7, "cr": 98.7, "fs": 79.5}),
    ("SN23-test", "pnlcalib_mv", 0.0712, 0.2571, 38.0,
     {"jac5": 78.7, "jac10": 89.6, "jac20": 91.9, "cr": 78.4, "fs": 61.8}),
    ("WC14-test", "pnlcalib", 0.00561, 0.14599, 44.0,
     {"jac5": 85.9, "jac10": 93.7, "jac20": 95.8, "cr": 100.0, "fs": 85.9}),
)
HOMOGRAPHY_CHECKS = (
    ("WC14-test", "pnlcalib_ft_wc14", 0.1274, 0.1439,
     {"iou_part": 97.0, "iou_whole": 93.4}),
    ("TSWC-test", "pnlcalib_ft_tswc", 0.2432, 0.8482,
     {"iou_part": 98.6, "iou_whole": 96.3}),
)


def official_check(model: str, tag: str, dataset: str, limit: float) -> dict:
    """Official evaluate() on the stored cameras; its script fixes the threshold at 20 px."""
    work = Config.CALIB_ROOT / "tmp"
    work.mkdir(exist_ok=True)
    root = Common.annotation_root(dataset)
    gt_zip, pred_zip = work / f"{dataset}_gt.zip", work / f"{dataset}_pred.zip"
    source = Config.CALIB_ROOT / "cams" / model / tag / f"{dataset}.jsonl.gz"
    with zipfile.ZipFile(gt_zip, "w") as gt, zipfile.ZipFile(pred_zip, "w") as pred, \
            gzip.open(source, "rt") as handle:
        for line in handle:
            row = json.loads(line)
            gt.writestr(f"{row['id']}.json", (root / f"{row['id']}.json").read_text())
            if row["ok"] and row["rep_err"] * Eval.EVAL_W / row["wh"][0] <= limit:
                pred.writestr(f"camera_{row['id']}.json",
                              json.dumps(Eval.scale_camera(row["cam"], row["wh"])))
    return evaluate(str(gt_zip), str(pred_zip))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    rows = []
    with get_context("fork").Pool(args.workers, Solve.init_worker) as pool:
        for dataset, model, kp_th, line_th, limit, published in CAMERA_CHECKS:
            tag = Solve.pair_tag(kp_th, line_th)
            Solve.solve_dataset(pool, model, dataset, kp_th, line_th, tag)
            frames = pd.read_csv(Eval.eval_dataset(pool, model, tag, dataset), dtype={"id": str})
            is_kept = frames["ok"] & (frames["rep_err_960"] <= limit)
            kept = frames.loc[is_kept]
            cr = is_kept.mean() * 100.0
            ours = {"cr": cr, "fs": cr * kept["jac5"].mean()}
            ours.update({k: kept[k].mean() * 100.0 for k in ("jac5", "jac10", "jac20")})
            rows += [{"dataset": dataset, "weights": model, "metric": k, "published": v,
                      "ours": ours[k]} for k, v in published.items()]
        for dataset, model, kp_th, line_th, published in HOMOGRAPHY_CHECKS:
            frames = pd.read_csv(Hom.score_dataset(pool, model, dataset, kp_th, line_th))
            rows += [{"dataset": dataset, "weights": model, "metric": k, "published": v,
                      "ours": frames.loc[frames["ok"], k].mean() * 100.0}
                     for k, v in published.items()]
    table = pd.DataFrame(rows)
    table["gap"] = table["ours"] - table["published"]
    table["is_pass"] = table["gap"].abs() <= MAX_GAP
    table.to_csv(Config.CALIB_ROOT / "reproduction.csv", index=False)
    print(table.round(2).to_string())

    dataset, model, kp_th, line_th, limit, _ = CAMERA_CHECKS[0]
    official = official_check(model, Solve.pair_tag(kp_th, line_th), dataset, limit)
    print("official evaluator at 20 px:", {k: round(v * 100.0, 2) for k, v in official.items()})
    if not table["is_pass"].all():
        sys.exit(1)


if __name__ == "__main__":
    main()
