"""Score stored cameras per image: focal length against its reference, and line fit.

Line fit uses the official SoccerNet evaluator at 960x540: hand-annotated lines on SoccerNet and
WC14, and lines projected by the dataset camera on video (agreement with that camera). TSWC has no
line annotations.
Focal length references: annotated homographies on WC14 and TSWC, the dataset camera on video.
Output: outputs/calibration/frames/<model>/<tag>/<dataset>.csv.gz
"""
import argparse
import gzip
import json
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calib_common as Common  # noqa: E402
from calib_common import Config  # noqa: E402
from snv3d.localize import CAMERA_COLUMNS, REF_WIDTH_PX, project_ball  # noqa: E402

sys.path.insert(0, str(Config.REPO_ROOT / "external" / "PnLCalib" / "sn_calibration" / "src"))
from evaluate_camera import (  # noqa: E402
    evaluate_camera_prediction,
    get_polylines,
    scale_points,
)
from evaluate_extremities import mirror_labels  # noqa: E402
from soccerpitch import SoccerPitch  # noqa: E402

EVAL_W, EVAL_H = Config.CALIB_EVAL_WH
PRED_SAMPLING = 0.9     # sampling_factor of the official evaluator
GT_STEP_M = 2.0
MIN_VISIBLE_POINTS = 2
UNITS: dict[str, dict] = {}


def jaccard_scores(prediction: dict, annotation: dict) -> dict:
    """Official per-image JaC at each threshold; the better pitch orientation wins at each."""
    projected = get_polylines(prediction, EVAL_W, EVAL_H, sampling_factor=PRED_SAMPLING)
    mirrored = mirror_labels(annotation)
    out = {}
    for threshold in Config.CALIB_JAC_PX:
        direct = evaluate_camera_prediction(projected, annotation, threshold)[0]
        flipped = evaluate_camera_prediction(projected, mirrored, threshold)[0]
        jac_direct = direct[0, 0] / direct.sum() if direct.sum() > 0 else 0.0
        jac_flipped = flipped[0, 0] / flipped.sum() if flipped.sum() > 0 else 0.0
        out[f"jac{threshold}"] = float(max(jac_direct, jac_flipped))
    return out


def scale_camera(cam: dict, size: tuple[int, int]) -> dict:
    """Camera solved at full resolution, expressed at the evaluation resolution."""
    sx, sy = EVAL_W / size[0], EVAL_H / size[1]
    scaled = dict(cam)
    scaled["x_focal_length"] = cam["x_focal_length"] * sx
    scaled["y_focal_length"] = cam["y_focal_length"] * sy
    scaled["principal_point"] = [cam["principal_point"][0] * sx, cam["principal_point"][1] * sy]
    return scaled


def pseudo_annotation(vec: np.ndarray, centre: tuple) -> dict:
    """Official pitch classes projected by the dataset camera, at the evaluation resolution."""
    lines = {}
    samples = SoccerPitch().sample_field_points(GT_STEP_M, GT_STEP_M)
    for name, points in samples.items():
        metres = np.asarray(points, float)
        world = np.c_[metres[:, 0] * Config.DM_PER_M + centre[0],
                      metres[:, 1] * Config.DM_PER_M + centre[1],
                      -metres[:, 2] * Config.DM_PER_M]
        cams = np.repeat(vec[None], len(world), axis=0)
        uvd = project_ball(world, cams, (float(EVAL_W), float(EVAL_H)), 1.0)
        uv = uvd[uvd[:, 2] > 0, :2]
        uv = uv[(uv[:, 0] >= 0) & (uv[:, 0] < EVAL_W) & (uv[:, 1] >= 0) & (uv[:, 1] < EVAL_H)]
        if len(uv) >= MIN_VISIBLE_POINTS:
            lines[name] = [{"x": float(u), "y": float(v)} for u, v in uv]
    return lines


def load_unit(unit: str) -> dict:
    """Dataset cameras of one video unit, cached per worker."""
    if unit not in UNITS:
        cams = pd.read_csv(Config.DATA_ROOT / unit / "camera_smooth.csv")
        cams = cams.drop_duplicates("frame_index").set_index("frame_index")[CAMERA_COLUMNS]
        UNITS[unit] = {"cams": cams, "centre": Config.PITCH_CENTRE_DM[unit.split("/")[0]],
                       "cached": None, "lines": None}
    return UNITS[unit]


def reference_focal(dataset: str, row: dict) -> float:
    """Reference focal length at 960 wide, or NaN when this image has none."""
    if dataset in Common.HOMOGRAPHY_DATASETS:
        width, height = Common.HOMOGRAPHY_WH
        forward = np.linalg.inv(Common.annotated_homography(dataset, str(row["id"])))
        return Common.focal_from_plane(forward, (width / 2.0, height / 2.0)) * EVAL_W / width
    if dataset in Common.VIDEO_DATASETS:
        vec = load_unit(row["unit"])["cams"].loc[int(row["id"])].to_numpy(float)
        return float(vec[CAMERA_COLUMNS.index("fx")] * EVAL_W / REF_WIDTH_PX)
    return np.nan


def score_row(job: tuple) -> dict:
    """Metrics of one solved image; a rejected or missing camera has no focal length or JaC."""
    row, dataset, annotation_path = job
    limit = Config.CALIB_MAX_REP_ERR_PX * row["wh"][0] / EVAL_W
    is_ok = bool(row["ok"]) and row["rep_err"] <= limit
    out = {"id": row["id"], "unit": row["unit"], "ok": is_ok, "n_kp": row.get("n_kp", 0),
           "n_ln": row.get("n_ln", 0), "mode": row.get("mode", ""),
           "rep_err_960": row.get("rep_err", np.nan) * EVAL_W / row["wh"][0],
           "f_ref": reference_focal(dataset, row), "f_pred": np.nan}
    if not is_ok:
        return out
    cam = scale_camera(row["cam"], row["wh"])
    out["f_pred"] = cam["x_focal_length"]
    if annotation_path is not None:
        annotation = json.loads(Path(annotation_path).read_text())
        out.update(jaccard_scores(cam, scale_points(annotation, EVAL_W, EVAL_H)))
    elif dataset in Common.VIDEO_DATASETS:
        info = load_unit(row["unit"])
        vec = info["cams"].loc[int(row["id"])].to_numpy(float)
        if info["cached"] is None or not np.array_equal(vec, info["cached"]):
            info["cached"], info["lines"] = vec, pseudo_annotation(vec, info["centre"])
        out.update(jaccard_scores(cam, info["lines"]))
    return out


def eval_dataset(pool, model: str, tag: str, dataset: str) -> Path:
    """Score one solved dataset; an existing result is kept, never overwritten."""
    source = Config.CALIB_ROOT / "cams" / model / tag / f"{dataset}.jsonl.gz"
    out = Config.CALIB_ROOT / "frames" / model / tag / f"{dataset}.csv.gz"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    root = Common.annotation_root(dataset)

    def jobs():
        with gzip.open(source, "rt") as handle:
            for line in handle:
                row = json.loads(line)
                yield row, dataset, (str(root / f"{row['id']}.json") if root else None)

    frames = pd.DataFrame(list(pool.imap(score_row, jobs(), chunksize=16)))
    if dataset in Common.VIDEO_DATASETS:
        plan = pd.read_csv(Config.CALIB_ROOT / "video_plan.csv")
        plan = plan.loc[plan["dataset"] == dataset, ["unit", "clip", "frame"]]
        frames = frames.merge(plan, left_on=["unit", "id"], right_on=["unit", "frame"],
                              how="left", validate="one_to_one").drop(columns="frame")
    tmp = out.with_suffix(".tmp.gz")
    frames.to_csv(tmp, index=False)
    tmp.replace(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                        help="a CALIB_VARIANTS name, or pnlcalib_mv_<tag> for a fine-tuned head")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if args.model not in Config.CALIB_VARIANTS and not args.model.startswith("pnlcalib_mv_"):
        parser.error(f"unknown model {args.model}")
    with get_context("fork").Pool(args.workers) as pool:
        for dataset in args.datasets:
            print(f"[ok] {eval_dataset(pool, args.model, args.tag, dataset)}", flush=True)


if __name__ == "__main__":
    main()
