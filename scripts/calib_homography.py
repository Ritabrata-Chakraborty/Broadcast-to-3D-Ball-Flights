"""Published homography metrics on WC14 and TSWC, used only for PnLCalib's reproduction check.

Solves the ground-plane homography from stored peaks, as in PnLCalib's eval_wc14.py and
eval_tswc.py, and compares it with the annotated one using PnLCalib's model/metrics.py.
Output: outputs/calibration/homography/<model>/<tag>/<dataset>.csv.gz
"""
import argparse
import random
import sys
import zlib
from multiprocessing import get_context
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calib_common as Common  # noqa: E402
import calib_solve as Solve  # noqa: E402
from calib_common import Config  # noqa: E402

from model.metrics import (  # noqa: E402
    calc_iou_part,
    calc_iou_whole_with_poly,
    calc_proj_error,
    calc_reproj_error,
)
from utils.utils_calib import FramebyFrameCalib  # noqa: E402
from utils.utils_heatmap import complete_keypoints  # noqa: E402

METER_TO_YARD = 1.09361
YARD_TO_METER = 0.9144
SOLVER: list[FramebyFrameCalib] = []


def pitch_to_template(homography: np.ndarray) -> np.ndarray:
    """Image-to-pitch in centred metres, to image-to-template in yards (PnLCalib's conversion)."""
    shift = np.eye(3)
    shift[:2, 2] = 105.0 / 2.0, 68.0 / 2.0
    return np.diag([METER_TO_YARD, METER_TO_YARD, 1.0]) @ shift @ homography


def metrics(pred: np.ndarray, gt: np.ndarray, seed: int) -> dict:
    """The four published metrics; a degenerate prediction scores 0 IoU and undefined errors."""
    random.seed(seed)
    out = {"iou_part": 0.0, "iou_whole": 0.0, "reproj": np.nan, "proj_m": np.nan}
    try:
        out["iou_part"] = float(calc_iou_part(pred, gt))
        out["iou_whole"] = float(calc_iou_whole_with_poly(pred, gt)[0])
        out["reproj"] = float(calc_reproj_error(pred, gt))
        out["proj_m"] = float(calc_proj_error(pred, gt)) * YARD_TO_METER
    except (ValueError, IndexError, AssertionError, np.linalg.LinAlgError, cv2.error) as err:
        out["error"] = f"{type(err).__name__}: {err}"[:120]
    return out


def solve_ground(job: tuple) -> dict:
    """Ground-plane solve and metrics of one image."""
    frame_id, dataset, kp, ln, net_wh, kp_th, line_th = job
    if not SOLVER:
        SOLVER.append(FramebyFrameCalib(*Common.HOMOGRAPHY_WH, denormalize=True))
    seed = zlib.crc32(f"{dataset}/{frame_id}".encode())
    cv2.setRNGSeed(seed % 2**31)
    np.random.seed(seed)
    keypoints, lines = Common.threshold_points(kp, ln, kp_th, line_th, is_ground_only=True)
    row = {"id": frame_id, "ok": False, "iou_part": 0.0, "iou_whole": 0.0, "reproj": np.nan,
           "proj_m": np.nan}
    try:
        kp_all, ln_all = complete_keypoints(keypoints, lines, w=net_wh[0], h=net_wh[1],
                                            normalize=True)
        SOLVER[0].update(kp_all, ln_all)
        result = SOLVER[0].heuristic_voting_ground(refine_lines=True)
    except Solve.HANDLED as err:
        row["error"] = f"{type(err).__name__}: {err}"[:120]
        return row
    if result is None:
        return row
    pred = pitch_to_template(np.asarray(result["homography"], float))
    row.update(ok=True, **metrics(pred, Common.annotated_homography(dataset, frame_id), seed))
    return row


def score_dataset(pool, model: str, dataset: str, kp_th: float, line_th: float) -> Path:
    """Solve and score one dataset; an existing result is kept, never overwritten."""
    tag = Solve.pair_tag(kp_th, line_th)
    out = Config.CALIB_ROOT / "homography" / model / tag / f"{dataset}.csv.gz"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)

    def jobs():
        for path in sorted(Common.raw_dir(model, dataset).glob("part_*.npz")):
            with np.load(path) as data:
                ids, kps, lns = data["ids"], data["kp"], data["ln"]
            for i, frame_id in enumerate(ids.tolist()):
                yield frame_id, dataset, kps[i], lns[i], Config.CALIB_NET_WH[model], kp_th, line_th

    frames = pd.DataFrame(list(pool.imap(solve_ground, jobs(), chunksize=8)))
    tmp = out.with_suffix(".tmp.gz")
    frames.to_csv(tmp, index=False)
    tmp.replace(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=Config.CALIB_VARIANTS)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--kp-th", type=float, required=True)
    parser.add_argument("--line-th", type=float, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    with get_context("fork").Pool(args.workers, Solve.init_worker) as pool:
        for dataset in args.datasets:
            print(f"[ok] {score_dataset(pool, args.model, dataset, args.kp_th, args.line_th)}")


if __name__ == "__main__":
    main()
