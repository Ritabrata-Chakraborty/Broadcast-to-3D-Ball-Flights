"""Grad et al.'s own fitting code on our test sets, unchanged.

Runs their segment fit (fit_one_trajectory_segment: straight model first, then the arc, with their
rejection rules) from the fitg configuration on each test set's track (ball bottom), box sizes and
cameras (converted to their camera_smooth format), and their own contact points: with GT inputs the
pitch points of their released preprocessing (camera01/05 turned back about the pitch centre with
--unrotate-dev-views), with video inputs the first and last detections' rays cut at the ground.
Output: outputs/eval/pred/<test set>/<tag>/fold_<k>_<split>_{flights,frames}.csv.gz (the same
columns as predict.py). Run in the ball3d env:
scripts/py scripts/run_grad.py gt_full --fold 1 --split test --tag grad --unrotate-dev-views
"""
import argparse
import json
import logging
import os
import pickle
import platform
import sys
import time
import warnings
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from joblib import Parallel, delayed
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as Config
from arcprior.metrics import flight_metrics

logging.disable(logging.WARNING)
warnings.filterwarnings("ignore")
VIEW_NAMES = ("broadcast", *[f"camera{c:02d}" for c in range(1, 7)])
CONF = Config.BALL3D_ROOT / "conf"
TRAJECTORY = "basic_kinetic_fitg"
METHOD = "b1"


def estimator_for(root: Path, camera_smooth: pd.DataFrame, metadata: dict):
    """Their KineticTrajectoryEstimator built from their own Hydra config (fitg)."""
    with initialize_config_dir(config_dir=str(CONF), version_base=None):
        cfg = compose(config_name="base", overrides=[f"root={root}", f"trajectory={TRAJECTORY}"])
    params = cfg.trajectory.estimator_params
    with open(root.parent / "pitch_geom" / "calibrate_camera_dict.pickle", "rb") as handle:
        camera = pickle.load(handle)
    return instantiate({"_target_": f"ball_estimation.trajectory.estimator."
                                    f"{cfg.trajectory.estimator_name}",
                        **params, "camera": camera, "camera_smooth_df": camera_smooth,
                        "video_metadata": metadata})


def camera_frame(rot: np.ndarray, trans: np.ndarray, intr: np.ndarray, frames: np.ndarray
                 ) -> pd.DataFrame:
    """Scenario cameras (m, 1920-wide pixels) in ball3d's camera_smooth format (dm, 640x360)."""
    rotvec = Rotation.from_matrix(rot.astype(float)).as_rotvec()
    scale = 640.0 / Config.IMAGE_WIDTH_PX
    return pd.DataFrame({"rot_x": rotvec[:, 0], "rot_y": rotvec[:, 1], "rot_z": rotvec[:, 2],
                         "tx": trans[:, 0] * Config.DM_PER_M, "ty": trans[:, 1] * Config.DM_PER_M,
                         "tz": trans[:, 2] * Config.DM_PER_M, "fx": intr[:, 0] * scale,
                         "fy": intr[:, 1] * scale, "princ_x": intr[:, 2] * scale,
                         "princ_y": intr[:, 3] * scale, "error": 0.0, "frame_index": frames})


def flight_frame(arrays: dict, i: int, width: float) -> pd.DataFrame:
    """One flight's per-frame inputs in ball3d's units: native pixels and decimetres."""
    n = int(arrays["n"][i])
    scale = width / Config.IMAGE_WIDTH_PX
    uv = arrays["uv_det"][i, :n].astype(float) * scale
    size = arrays["size_det"][i, :n].astype(float) * scale * np.sqrt(2.0)
    pitch = arrays["pitch_xy"][i, :n].astype(float) * Config.DM_PER_M
    return pd.DataFrame({"file_name": int(arrays["a"][i]) + np.arange(n), "xk": uv[:, 0],
                         "yk": uv[:, 1], "size": size, "x_pitch2D": pitch[:, 0],
                         "y_pitch2D": pitch[:, 1]})


def fit_unit(arrays: dict, chosen: np.ndarray) -> list[dict]:
    """Every chosen view-flight of one unit; paths in decimetres."""
    view, half = int(arrays["view"][chosen[0]]), int(arrays["half"][chosen[0]])
    root = Config.DATA_ROOT / Config.LP / VIEW_NAMES[view] / Config.HALVES[half]
    metadata = json.loads((root / "sequence_metadata.json").read_text())
    width = float(metadata["width"])
    parts = []
    for i in chosen:
        n = int(arrays["n"][i])
        frames = int(arrays["a"][i]) + np.arange(n)
        parts.append(camera_frame(arrays["rot"][i, :n], arrays["trans"][i, :n],
                                  arrays["intr"][i, :n], frames))
    cameras = pd.concat(parts).drop_duplicates("frame_index").sort_values("frame_index")
    estimator = estimator_for(root, cameras.reset_index(drop=True), metadata)
    out = []
    for i in chosen:
        n = int(arrays["n"][i])
        a = int(arrays["a"][i])
        b = a + n - 1
        df = flight_frame(arrays, i, width)
        path, err, kind = None, None, "not_fitted"
        if "has_camera" in arrays and not arrays["has_camera"][i]:
            kind = "no_camera"
        elif not (np.isfinite(df[["x_pitch2D", "y_pitch2D"]].iloc[[0, -1]].values).all()):
            kind = "no_anchor"
        else:
            try:
                with open(os.devnull, "w") as sink, redirect_stdout(sink):
                    xt, yt, zt, err, _, kind, *_ = estimator.fit_one_trajectory_segment(
                        df, a, b, True)
                path = None if xt is None else np.c_[xt, yt, zt]
            except Exception as error:   # lmfit raises on degenerate inputs
                kind = f"error: {type(error).__name__}"
                path = None
        out.append({"row": int(i), "method": METHOD, "kind": kind, "err": err, "path": path})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario")
    parser.add_argument("--fold", required=True, choices=sorted(Config.FOLDS))
    parser.add_argument("--split", required=True, choices=("train", "val", "test"))
    parser.add_argument("--tag", required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--data-root", type=Path, default=Config.EVAL_DATA)
    parser.add_argument("--out-root", type=Path, default=Config.EVAL_ROOT / "pred")
    parser.add_argument("--unrotate-dev-views", action="store_true",
                        help="turn the anchor points of Config.LP_ROTATED_DEV_VIEWS back by 180 "
                             "degrees about the pitch centre (released points in a rotated frame)")
    args = parser.parse_args()
    out = args.out_root / args.scenario / args.tag
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"fold_{args.fold}_{args.split}"
    if Path(f"{stem}_flights.csv.gz").exists():
        raise FileExistsError(f"{stem}_flights.csv.gz exists")
    with np.load(args.data_root / args.scenario / f"fold_{args.fold}" /
                 f"{args.split}.npz") as archive:
        arrays = {k: archive[k] for k in archive.files}
    # Grad et al.'s own contact points, stored beside ours by build_test_sets.py.
    arrays["pitch_xy"] = arrays["pitch_xy_ball3d"]
    if args.unrotate_dev_views:
        centre = np.array(Config.LP_PITCH_CENTRE_DM) / Config.DM_PER_M
        rows = np.isin(arrays["view"], Config.LP_ROTATED_DEV_VIEWS)
        arrays["pitch_xy"] = arrays["pitch_xy"].copy()
        arrays["pitch_xy"][rows] = 2.0 * centre - arrays["pitch_xy"][rows]
    units = sorted(set(zip(arrays["view"].tolist(), arrays["half"].tolist())))
    chunks = []
    for view, half in units:
        chosen = np.where((arrays["view"] == view) & (arrays["half"] == half))[0]
        chunks += [chosen[k:k + 25] for k in range(0, len(chosen), 25)]
    began = time.time()
    done = Parallel(n_jobs=args.workers, verbose=0)(
        delayed(fit_unit)(arrays, chunk) for chunk in chunks)
    rows, frames = [], []
    for result in (r for chunk in done for r in chunk):
        i = result["row"]
        n = int(arrays["n"][i])
        metrics = flight_metrics(result["path"], arrays["xyz"][i, :n].astype(float))
        rows.append({"method": result["method"], "share": 1.0, "view": int(arrays["view"][i]),
                     "half": int(arrays["half"][i]), "group": int(arrays["group"][i]), "n": n,
                     "cut": n, "start_high": bool(arrays["start_high"][i]),
                     "end_high": bool(arrays["end_high"][i]),
                     "z_start": float(arrays["z_start"][i]), "z_end": float(arrays["z_end"][i]),
                     "apex_true": float(np.nanmax(arrays["xyz"][i, :n, 2])),
                     "fit_score": result["err"] if metrics["answered"] else np.nan,
                     "omega": np.nan, "kind": result["kind"], **metrics})
        values = np.full((n, 3), np.nan) if result["path"] is None else \
            np.asarray(result["path"], float)[:n]
        frames.append(pd.DataFrame({
            "method": result["method"], "view": int(arrays["view"][i]),
            "half": int(arrays["half"][i]), "group": int(arrays["group"][i]),
            "file_name": int(arrays["a"][i]) + np.arange(n), "x_predicted": values[:, 0],
            "y_predicted": values[:, 1], "z_predicted": values[:, 2],
            "fit_score": rows[-1]["fit_score"], "omega": np.nan}))
    pd.DataFrame(rows).to_csv(f"{stem}_flights.csv.gz", index=False)
    pd.concat(frames).to_csv(f"{stem}_frames.csv.gz", index=False)
    meta = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    meta.update(view_flights=int(len(arrays["n"])), seconds=round(time.time() - began),
                python=platform.python_version(), numpy=np.__version__,
                finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    Path(f"{stem}_meta.json").write_text(json.dumps(meta, indent=1))
    table = pd.DataFrame(rows)
    print(table.groupby(["method", table["view"] == 0])[["rmse_cap", "coverage"]].median())
    print(table.groupby("method")["kind"].value_counts().to_string())


if __name__ == "__main__":
    main()
