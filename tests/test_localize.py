"""Round-trip tests for SNv3D-size localisation against Grad et al.'s own projection code."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "external" / "ball3d" / "src"))

pytest.importorskip("numba")   # Grad et al.'s camera code needs numba (ball3d env)
from ball_estimation.camera import Camera  # noqa: E402
from snv3d.localize import localize_ball, select_detections  # noqa: E402

# Real calibrations from camera_smooth.csv (LP camera03 4K and SW pano), 640x360 reference space.
CAMERAS = {
    "lp_static_4k": (np.array([-1.241, -0.107, -0.127, -401.564, -127.752, 1088.331,
                               522.42, 522.42, 319.917, 179.917]), (3840, 2160)),
    "sw_pano": (np.array([-0.273, -1.203, -2.547, -60.412, -155.17, 360.399,
                          100.036, 351.688, 287.464, 305.871]), (7500, 1200)),
}
DIAMETER_DM = 2.2


def project_ball3d(cam_row: np.ndarray, points: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Project world points (height up) to full-res pixels exactly as ball3d's loss does."""
    proj = Camera.from_numpy(cam_row).to_projection_matrix()
    flipped = points * np.array([1.0, 1.0, -1.0])
    hom = (proj @ np.c_[flipped, np.ones(len(points))].T).T
    uv = hom[:, :2] / hom[:, 2:]
    return uv * np.array([size[0] / 640.0, size[1] / 360.0])


@pytest.mark.parametrize("name", list(CAMERAS))
def test_round_trip_recovers_world_point(name: str) -> None:
    cam_row, size = CAMERAS[name]
    rng = np.random.default_rng(0)
    points = np.c_[rng.uniform(100, 900, 20), rng.uniform(100, 600, 20), rng.uniform(0, 80, 20)]
    if name == "sw_pano":
        points = np.c_[rng.uniform(-30, 80, 20), rng.uniform(-20, 30, 20), rng.uniform(0, 30, 20)]

    uv = project_ball3d(cam_row, points, size)

    # Exact pixel diameter of the ball for the pinhole model: d = fx_full * phi / Z_cam.
    rot = Camera.from_numpy(cam_row)
    import cv2 as cv
    R = cv.Rodrigues(rot.rotation)[0]
    z_cam = (R @ (points * np.array([1.0, 1.0, -1.0])).T + rot.translation).T[:, 2]
    d = cam_row[6] * size[0] / 640.0 * DIAMETER_DM / z_cam

    cams = np.repeat(cam_row[None], len(points), axis=0)
    est = localize_ball(uv[:, 0], uv[:, 1], d, cams, size, DIAMETER_DM)
    np.testing.assert_allclose(est, points, atol=1e-6)


def test_size_error_scales_depth() -> None:
    cam_row, size = CAMERAS["lp_static_4k"]
    point = np.array([[500.0, 300.0, 0.0]])
    uv = project_ball3d(cam_row, point, size)
    cams = cam_row[None]
    pos = Camera.from_numpy(cam_row).get_pos().ravel() * np.array([1, 1, -1])

    d_true = localize_ball(uv[:, 0], uv[:, 1], np.array([1.0]), cams, size, DIAMETER_DM)
    d_half = localize_ball(uv[:, 0], uv[:, 1], np.array([2.0]), cams, size, DIAMETER_DM)

    # Doubling the pixel diameter halves the camera distance along the same ray.
    np.testing.assert_allclose(np.linalg.norm(d_half - pos), 0.5 * np.linalg.norm(d_true - pos))


def test_select_detections_keeps_best_per_frame() -> None:
    det = pd.DataFrame({
        "file_name": [0, 0, 1], "x0": [0, 10, 5], "y0": [0, 10, 5],
        "x1": [4, 18, 9], "y1": [4, 16, 11], "score": [0.2, 0.9, 0.5],
    })
    out = select_detections(det)
    assert out["file_name"].tolist() == [0, 1]
    assert out.loc[0, "u"] == 14 and out.loc[0, "v"] == 13 and out.loc[0, "d"] == 7
    assert out.loc[1, "d"] == 5
