"""SoccerNet-v3D single-image 3D ball localization (Gutierrez-Perez & Agudo, CVPRW 2025, Eq. 5-7).

Depth along the ray through the ball centre is set so that the detected pixel diameter d matches
the true ball diameter phi. Cameras follow the ball3d convention (camera_smooth.csv):
x_cam = R X + t with world z pointing down, intrinsics defined on a 640x360 reference image.
"""
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

CAMERA_COLUMNS = ["rot_x", "rot_y", "rot_z", "tx", "ty", "tz", "fx", "fy", "princ_x", "princ_y"]
REF_WIDTH_PX = 640.0
REF_HEIGHT_PX = 360.0


def localize_ball(
    u: np.ndarray,
    v: np.ndarray,
    d: np.ndarray,
    cameras: np.ndarray,
    image_size: tuple[float, float],
    diameter: float,
) -> np.ndarray:
    """Return (N, 3) world positions (x, y, height; height up) for N detections.

    u, v, d are full-resolution pixel centre and diameter; cameras is (N, 10) in CAMERA_COLUMNS
    order; image_size is (width, height) of the source video; diameter is in world units.
    """
    width, height = image_size
    sx = width / REF_WIDTH_PX
    sy = height / REF_HEIGHT_PX

    # Full-resolution intrinsics: K_full = diag(sx, sy, 1) @ K_ref.
    fx = cameras[:, 6] * sx
    fy = cameras[:, 7] * sy
    cx = cameras[:, 8] * sx
    cy = cameras[:, 9] * sy

    # Eq. 5-6: p_c = K^-1 [u, v, 1]; |p_c^+ - p_c^-| = d / fx for horizontally opposed edges.
    p_c = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], axis=1)
    edge_dist = d / fx

    # Eq. 7: scale the ray so the ball subtends d pixels, then map camera -> world.
    x_cam = diameter * p_c / edge_dist[:, None]
    rot = Rotation.from_rotvec(np.array(cameras[:, 0:3])).as_matrix()
    world = np.einsum("nji,nj->ni", rot, x_cam - cameras[:, 3:6])

    world[:, 2] *= -1.0
    return world


def camera_centres(cameras: np.ndarray) -> np.ndarray:
    """Camera centres (N, 3) in world units with height up, from (N, 10) camera rows."""
    rot = Rotation.from_rotvec(np.array(cameras[:, 0:3])).as_matrix()
    centres = -np.einsum("nji,nj->ni", rot, cameras[:, 3:6])
    centres[:, 2] *= -1.0
    return centres


def project_ball(
    world: np.ndarray, cameras: np.ndarray, image_size: tuple[float, float], diameter: float
) -> np.ndarray:
    """Project (N, 3) world points (height up) to full-res pixels; return (N, 3) [u, v, d]."""
    width, height = image_size
    sx = width / REF_WIDTH_PX
    sy = height / REF_HEIGHT_PX
    rot = Rotation.from_rotvec(np.array(cameras[:, 0:3])).as_matrix()
    flipped = world * np.array([1.0, 1.0, -1.0])
    x_cam = np.einsum("nij,nj->ni", rot, flipped) + cameras[:, 3:6]

    u = cameras[:, 6] * sx * x_cam[:, 0] / x_cam[:, 2] + cameras[:, 8] * sx
    v = cameras[:, 7] * sy * x_cam[:, 1] / x_cam[:, 2] + cameras[:, 9] * sy
    d = cameras[:, 6] * sx * diameter / x_cam[:, 2]
    return np.stack([u, v, d], axis=1)


def select_detections(detections: pd.DataFrame, min_score: float = 0.0) -> pd.DataFrame:
    """Keep the highest-scoring ball box per frame and add centre/diameter columns."""
    det = detections.loc[detections["score"] >= min_score].copy()
    det = det.sort_values("score", ascending=False).drop_duplicates("file_name")
    det["u"] = 0.5 * (det["x0"] + det["x1"])
    det["v"] = 0.5 * (det["y0"] + det["y1"])

    # Optimized SNv3D boxes are square (width = height = d); average to reduce blur elongation.
    det["d"] = 0.5 * ((det["x1"] - det["x0"]) + (det["y1"] - det["y0"]))
    return det.sort_values("file_name").reset_index(drop=True)
