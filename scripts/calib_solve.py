"""Turn stored network peaks into cameras with PnLCalib's solver, for one threshold pair.

Every image is solved at 960x540, as in PnLCalib's benchmark script: the voting uses a fixed 5 px
threshold, so the solve resolution changes which camera it picks.
Output: outputs/calibration/cams/<model>/<tag>/<dataset>.jsonl.gz, one row per frame. Seeds are set
per frame, so results do not depend on scheduling.
"""
import argparse
import gzip
import json
import sys
import zlib
from multiprocessing import get_context
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calib_common as Common  # noqa: E402
from calib_common import Config  # noqa: E402

sys.path.insert(0, str(Config.THIRD_PARTY_ROOT / "PnLCalib"))
from utils.utils_calib import FramebyFrameCalib  # noqa: E402
from utils.utils_heatmap import complete_keypoints  # noqa: E402

SOLVERS: dict[tuple[int, int], FramebyFrameCalib] = {}
HANDLED = (ValueError, np.linalg.LinAlgError, cv2.error, ZeroDivisionError, IndexError)


def init_worker() -> None:
    """OpenCV must not spawn threads inside the pool."""
    cv2.setNumThreads(1)


def solve_frame(job: tuple) -> dict:
    """Solve one frame; a failure of the solver counts as no camera, as in the benchmark scripts."""
    frame_id, unit, kp, ln, net_wh, kp_th, line_th = job
    size = Config.CALIB_EVAL_WH
    if size not in SOLVERS:
        SOLVERS[size] = FramebyFrameCalib(iwidth=size[0], iheight=size[1], denormalize=True)
    solver = SOLVERS[size]
    seed = zlib.crc32(f"{unit}/{frame_id}".encode())
    cv2.setRNGSeed(seed % 2**31)
    np.random.seed(seed)
    row = {"id": frame_id, "unit": unit, "wh": size, "ok": False}
    keypoints, lines = Common.threshold_points(kp, ln, kp_th, line_th)
    row.update(n_kp=len(keypoints), n_ln=len(lines))
    try:
        kp_all, ln_all = complete_keypoints(keypoints, lines, w=net_wh[0], h=net_wh[1],
                                            normalize=True)
        solver.update(kp_all, ln_all)
        result = solver.heuristic_voting(refine=False, refine_lines=True)
    except HANDLED as err:
        row["error"] = f"{type(err).__name__}: {err}"[:120]
        return row
    if result:
        row.update(ok=True, rep_err=float(result["rep_err"]), mode=result["mode"],
                   cam=json.loads(json.dumps(result["cam_params"], default=float)))
    return row


def iter_jobs(model: str, dataset: str, kp_th: float, line_th: float):
    """One job per stored frame of a dataset."""
    net_wh = Config.CALIB_NET_WH.get(model, Config.CALIB_NET_WH["pnlcalib_mv"])
    for path in sorted(Common.raw_dir(model, dataset).glob("*.npz")):
        unit = dataset if path.stem.startswith("part_") else path.stem.replace("__", "/")
        with np.load(path) as data:
            ids, kps, lns = data["ids"], data["kp"], data["ln"]
        for i, frame_id in enumerate(ids.tolist()):
            yield frame_id, unit, kps[i], lns[i], net_wh, kp_th, line_th


def solve_dataset(pool, model: str, dataset: str, kp_th: float, line_th: float, tag: str) -> Path:
    """Solve all stored frames of a dataset; an existing result is kept, never overwritten."""
    out = Config.CALIB_ROOT / "cams" / model / tag / f"{dataset}.jsonl.gz"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    with gzip.open(tmp, "wt", compresslevel=3) as handle:
        jobs = iter_jobs(model, dataset, kp_th, line_th)
        for row in pool.imap(solve_frame, jobs, chunksize=8):
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    tmp.replace(out)
    return out


def pair_tag(kp_th: float, line_th: float) -> str:
    return f"kp{kp_th:g}_ln{line_th:g}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                        help="a CALIB_VARIANTS name, or pnlcalib_mv_<tag> for a fine-tuned head")
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--kp-th", type=float, required=True)
    parser.add_argument("--line-th", type=float, required=True)
    parser.add_argument("--workers", type=int, default=14)
    args = parser.parse_args()
    if args.model not in Config.CALIB_VARIANTS and not args.model.startswith("pnlcalib_mv_"):
        parser.error(f"unknown model {args.model}")
    with get_context("fork").Pool(args.workers, init_worker) as pool:
        for dataset in args.datasets:
            path = solve_dataset(pool, args.model, dataset, args.kp_th, args.line_th,
                                 pair_tag(args.kp_th, args.line_th))
            print(f"[ok] {path}", flush=True)


if __name__ == "__main__":
    main()
