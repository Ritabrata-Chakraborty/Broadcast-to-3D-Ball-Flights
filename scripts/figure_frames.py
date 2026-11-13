"""LP broadcast frames and the dataset camera, for drawing 3D paths on video frames.

The frames are non-commercial research data: figures that hold them must never be committed.
"""
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as Config  # noqa: E402

CAM_COLS = ["rot_x", "rot_y", "rot_z", "tx", "ty", "tz", "fx", "fy", "princ_x", "princ_y"]


def unit_dir(half: int) -> Path:
    return Config.DATA_ROOT / Config.LP / "broadcast" / Config.HALVES[half]


def camera(half: int, frame: int) -> tuple:
    cams = pd.read_csv(unit_dir(half) / "camera_smooth.csv").drop_duplicates("frame_index")
    row = cams.set_index("frame_index").loc[frame, CAM_COLS].to_numpy(float)
    return Rotation.from_rotvec(row[:3].copy()).as_matrix(), row[3:6], row[6:10]


def project(xyz_m: np.ndarray, cam: tuple, width: float, height: float) -> np.ndarray:
    """Pixels of pitch points (metres, z up) in the benchmark's camera (decimetres, z down)."""
    rot, trans, intr = cam
    point = xyz_m * Config.DM_PER_M * np.array([1.0, 1.0, -1.0])
    x_cam = point @ rot.T + trans
    fx, fy = intr[0] * width / 640.0, intr[1] * height / 360.0
    cx, cy = intr[2] * width / 640.0, intr[3] * height / 360.0
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.c_[fx * x_cam[:, 0] / x_cam[:, 2] + cx, fy * x_cam[:, 1] / x_cam[:, 2] + cy]


def read_frame(half: int, frame: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(unit_dir(half) / "input.mkv"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
    ok, image = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read frame {frame}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
