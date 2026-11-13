"""Calibration comparison: focal reference, official-metric wrapper and decision helpers."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import calib_common as Common  # noqa: E402
import calib_eval as Eval  # noqa: E402
import calib_report as Report  # noqa: E402


def make_camera(focal: float = 800.0) -> dict:
    """A camera on the near side looking at the pitch, in SoccerNet's z-down frame."""
    return {"pan_degrees": 0.0, "tilt_degrees": 75.0, "roll_degrees": 0.0,
            "position_meters": [0.0, 55.0, -15.0], "x_focal_length": focal, "y_focal_length": focal,
            "principal_point": [480.0, 270.0], "radial_distortion": [0.0] * 6,
            "tangential_distortion": [0.0] * 2, "thin_prism_distortion": [0.0] * 4}


def plane_to_image(focal: float, rotvec: tuple, translation: tuple, principal: tuple,
                   plane_scale: float = 1.0) -> np.ndarray:
    """Homography of the z = 0 plane under a pinhole camera."""
    rot = Rotation.from_rotvec(rotvec).as_matrix()
    intrinsics = np.array([[focal, 0.0, principal[0]], [0.0, focal, principal[1]], [0.0, 0.0, 1.0]])
    return intrinsics @ np.c_[rot[:, 0] * plane_scale, rot[:, 1] * plane_scale, translation]


def annotation_of(camera: dict) -> dict:
    return Eval.get_polylines(camera, Eval.EVAL_W, Eval.EVAL_H, sampling_factor=2.0)


@pytest.mark.parametrize("focal", [600.0, 1500.0, 4000.0])
def test_focal_recovered_from_homography(focal: float) -> None:
    forward = plane_to_image(focal, (1.9, 0.5, -0.3), (5.0, 3.0, 70.0), (640.0, 360.0), 0.91)
    assert Common.focal_from_plane(forward * 3.7, (640.0, 360.0)) == pytest.approx(focal, rel=1e-6)


def test_focal_rejects_image_parallel_axis() -> None:
    forward = plane_to_image(1000.0, (1.9, 0.0, 0.0), (0.0, 0.0, 70.0), (640.0, 360.0))
    assert np.isnan(Common.focal_from_plane(forward, (640.0, 360.0)))


def test_focal_rejects_disagreeing_constraints() -> None:
    forward = plane_to_image(1000.0, (1.9, 0.5, -0.3), (5.0, 3.0, 70.0), (640.0, 360.0))
    assert np.isnan(Common.focal_from_plane(forward @ np.diag([1.5, 1.0, 1.0]), (640.0, 360.0)))


def test_exact_camera_scores_one() -> None:
    camera = make_camera()
    annotation = annotation_of(camera)
    assert annotation
    assert Eval.jaccard_scores(camera, annotation)["jac5"] == pytest.approx(1.0)


def test_wrong_camera_scores_lower_at_strict_threshold() -> None:
    scores = Eval.jaccard_scores(make_camera(860.0), annotation_of(make_camera()))
    assert scores["jac5"] < 1.0
    assert scores["jac5"] <= scores["jac10"] <= scores["jac20"]


def test_rotated_pitch_scores_one_via_mirror() -> None:
    camera = make_camera()
    mirrored = Eval.mirror_labels(annotation_of(camera))
    assert Eval.jaccard_scores(camera, mirrored)["jac5"] == pytest.approx(1.0)


def test_scale_camera_keeps_ratio() -> None:
    cam = {"x_focal_length": 2000.0, "y_focal_length": 2000.0, "principal_point": [960.0, 544.0]}
    scaled = Eval.scale_camera(cam, (1920, 1088))
    assert scaled["principal_point"] == pytest.approx([480.0, 270.0])
    assert scaled["x_focal_length"] == pytest.approx(1000.0)
    assert scaled["y_focal_length"] == pytest.approx(2000.0 * 540 / 1088)


def test_threshold_points_needs_both_line_ends_and_drops_goal_points() -> None:
    kp = np.zeros((Common.N_KEYPOINTS, 3))
    kp[3] = (10.0, 20.0, 0.9)
    kp[11] = (30.0, 40.0, 0.9)
    ln = np.zeros((Common.N_LINES, 2, 3))
    ln[0] = ((1.0, 2.0, 0.9), (3.0, 4.0, 0.9))
    ln[1] = ((1.0, 2.0, 0.9), (3.0, 4.0, 0.1))
    ln[6] = ((1.0, 2.0, 0.9), (3.0, 4.0, 0.9))
    keypoints, lines = Common.threshold_points(kp, ln, 0.5, 0.5)
    assert sorted(keypoints) == [4, 12]
    assert sorted(lines) == [1, 7]
    keypoints, lines = Common.threshold_points(kp, ln, 0.5, 0.5, is_ground_only=True)
    assert list(keypoints) == [4]
    assert list(lines) == [1]


def test_focal_errors_count_no_camera_as_failure_and_drop_missing_reference() -> None:
    frames = pd.DataFrame({"ok": [True, False, True], "f_pred": [110.0, np.nan, 90.0],
                           "f_ref": [100.0, 100.0, np.nan]})
    errors = Report.focal_errors(frames).to_numpy()
    assert errors[0] == pytest.approx(0.1)
    assert np.isinf(errors[1])
    assert len(errors) == 2


def test_final_score_ignores_jac_of_missing_cameras() -> None:
    frames = pd.DataFrame({"ok": [True, True, False, False], "jac5": [1.0, 0.5, np.nan, np.nan]})
    assert Report.final_score(frames) == pytest.approx(0.5 * 0.75)


def test_paired_median_diff_handles_failures() -> None:
    a = np.array([0.1, 0.1, 0.1, 0.1])
    b = np.full(4, np.inf)
    clusters, strata = np.arange(4), np.zeros(4, int)
    point, low, high = Report.paired_median_diff(a, b, clusters, strata)
    assert point == -np.inf and high == -np.inf and low == -np.inf
    assert Report.winner_of(low, high, lower_is_better=True) == "pnlcalib"


def test_winner_needs_interval_to_exclude_zero() -> None:
    assert Report.winner_of(-0.1, 0.1, lower_is_better=True) == "none"
    assert Report.winner_of(0.1, 0.2, lower_is_better=True) == "soccermaster"
    assert Report.winner_of(0.1, 0.2, lower_is_better=False) == "pnlcalib"


def test_decision_tiers() -> None:
    table = pd.DataFrame([
        {"dataset": "WC14-test", "metric": "median focal error", "winner": "none"},
        {"dataset": "TSWC-test", "metric": "median focal error", "winner": "none"},
        {"dataset": "ISSIA", "metric": "median focal error", "winner": "soccermaster"},
        {"dataset": "LP-static", "metric": "median focal error", "winner": "none"}])
    decision = Report.decide(table)
    assert decision["winner"] == "soccermaster"
    assert decision["decided_by"] == "dataset camera focal references"






