"""Camera solved from hand-annotated lines, as a second focal length reference.

Annotated lines give keypoints and line ends through PnLCalib's own label code, and these go through
the same solver as the networks, so no network is involved. Images whose oracle camera does not fit
its own annotation (JaC at 5 px below CALIB_ORACLE_MIN_JAC) are not used as a reference.
Output: outputs/calibration/oracle/<dataset>.csv
"""
import argparse
import json
import sys
import zlib
from multiprocessing import get_context
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calib_common as Common  # noqa: E402
import calib_eval as Eval  # noqa: E402
import calib_solve as Solve  # noqa: E402
from calib_common import Config  # noqa: E402

from utils.utils_calib import FramebyFrameCalib  # noqa: E402
from utils.utils_heatmap import (  # noqa: E402
    coords_to_dict,
    get_keypoints_from_heatmap_batch_maxpool,
    get_keypoints_from_heatmap_batch_maxpool_l,
)
from utils.utils_keypoints import KeypointsDB  # noqa: E402
from utils.utils_lines import LineKeypointsDB  # noqa: E402

LABEL_THRESHOLD = 0.01     # PnLCalib's --use_gt path
LABEL_ERRORS = (*Solve.HANDLED, KeyError, TypeError, AttributeError)


def solve_annotation(job: tuple) -> dict:
    """Oracle camera of one annotated image at 960x540."""
    path, dataset = job
    frame_id = Common.image_id(dataset, Path(path))
    seed = zlib.crc32(f"{dataset}/{frame_id}".encode())
    cv2.setRNGSeed(seed % 2**31)
    np.random.seed(seed)
    row = {"id": frame_id, "ok": False, "f_oracle": np.nan, "jac5": np.nan}
    annotation = json.loads(Path(path).with_suffix(".json").read_text())
    image = Image.open(path).convert("RGB").resize(Config.CALIB_EVAL_WH)
    tensor = F.to_tensor(image)
    try:
        kp_map = torch.tensor(KeypointsDB(annotation, tensor).get_tensor_w_mask()[0]).unsqueeze(0)
        ln_map = torch.tensor(LineKeypointsDB(annotation, tensor).get_tensor()).unsqueeze(0)
        keypoints = coords_to_dict(get_keypoints_from_heatmap_batch_maxpool(kp_map[:, :-1]),
                                   threshold=LABEL_THRESHOLD)[0]
        lines = coords_to_dict(get_keypoints_from_heatmap_batch_maxpool_l(ln_map[:, :-1]),
                               threshold=LABEL_THRESHOLD)[0]
        solver = FramebyFrameCalib(*Config.CALIB_EVAL_WH)
        solver.update(keypoints, lines)
        result = solver.heuristic_voting(refine=False, refine_lines=True)
    except LABEL_ERRORS as err:
        row["error"] = f"{type(err).__name__}: {err}"[:120]
        return row
    if result:
        cam = result["cam_params"]
        lines_960 = Eval.scale_points(annotation, Eval.EVAL_W, Eval.EVAL_H)
        row.update(ok=True, f_oracle=float(cam["x_focal_length"]),
                   jac5=Eval.jaccard_scores(cam, lines_960)["jac5"])
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["SN22-test-center", "SN23-test",
                                                          "WC14-test"])
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    out_dir = Config.CALIB_ROOT / "oracle"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    with get_context("fork").Pool(args.workers, Solve.init_worker) as pool:
        for dataset in args.datasets:
            out = out_dir / f"{dataset}.csv"
            if out.exists():
                print(f"[keep] {out}", flush=True)
                continue
            jobs = [(str(p), dataset) for p in Common.dataset_images(dataset)]
            table = pd.DataFrame(list(pool.imap(solve_annotation, jobs, chunksize=8)))
            table.to_csv(out, index=False)
            usable = (table["jac5"] >= Config.CALIB_ORACLE_MIN_JAC).mean()
            print(f"[ok] {out}: usable share {usable:.3f}", flush=True)


if __name__ == "__main__":
    main()
