"""The flight model, the oracle fit, rays and projection, the flights and splits, the tracker."""

import numpy as np
import pandas as pd
import pytest

import config as Config
from arcprior.oracle import fit_oracle, simulate

FPS, G, SUB = Config.FPS, Config.G_MPS2, Config.SUBSTEPS
PARAMS = np.array([1.0, 2.0, 0.2, 18.0, -6.0, 9.0, 0.018, -0.011, 0.007])


def test_simulate_matches_ball3d_ode() -> None:
    model = pytest.importorskip("ball_estimation.trajectory.model")
    x, y, z = model.ArcTrajectoryModel.get_arc_trajectory(
        PARAMS[0] * 10, PARAMS[1] * 10, PARAMS[2] * 10, PARAMS[3] * 10, PARAMS[4] * 10,
        PARAMS[5] * 10, PARAMS[6], 40, FPS, G, kl=PARAMS[7], ks=PARAMS[8])
    theirs = np.c_[x, y, z] / 10.0
    assert np.abs(simulate(PARAMS, 40, FPS, G, SUB) - theirs).max() < 1e-3


def test_oracle_fit_recovers_planted_spin_and_handles_gaps() -> None:
    path = simulate(PARAMS, 40, FPS, G, SUB)
    path[[5, 6, 20]] = np.nan
    fit = fit_oracle(path, FPS, G, SUB, Config.K3_MAX, Config.KL_KS_MAX)
    assert fit["rmse_spin_m"] < 1e-3
    assert fit["rmse_spin_m"] <= fit["rmse_drag_m"]
    assert np.abs(fit["params"][6:9] - PARAMS[6:9]).max() < 1e-3


@pytest.mark.parametrize("fold,half", list(Config.HALF_FOLDS.items()))
def test_folds_share_no_flight_and_test_is_the_other_half(fold: str, half: str) -> None:
    root = Config.FLIGHTS_DIR
    if not (root / "flights.csv").exists():
        pytest.skip("dataset not built")
    flights = pd.read_csv(root / "flights.csv")
    split = flights[f"split_{fold}"]
    assert (split[flights["half"] != half] == "test").all()
    assert flights.loc[split == "train", "broadcast_b"].max() < \
        flights.loc[split == "val", "broadcast_a"].min()
    groups = [set(np.load(root / f"fold_{fold}" / f"{s}.npz")["group"])
              for s in ("train", "val", "test")]
    assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]


def test_every_flight_is_tested_once_across_folds() -> None:
    root = Config.FLIGHTS_DIR
    if not (root / "flights.csv").exists():
        pytest.skip("dataset not built")
    flights = pd.read_csv(root / "flights.csv")
    tested = (flights["split_a"] == "test").astype(int) + (flights["split_b"] == "test").astype(int)
    assert (tested == 1).all()


def test_selection_rule_is_respected() -> None:
    root = Config.FLIGHTS_DIR
    if not (root / "flights.csv").exists():
        pytest.skip("dataset not built")
    flights = pd.read_csv(root / "flights.csv")
    assert (flights["apex_m"] >= Config.MIN_APEX_M).all()
    assert (flights["n"] >= Config.MIN_FRAMES).all()
    assert (flights["speed_mps"] <= Config.MAX_SPEED_MPS).all()


# ---------------------------------------------------------------- Torch (snv3d env)


def test_torch_integrator_matches_numpy() -> None:
    torch = pytest.importorskip("torch")
    from arcprior.flight import integrate
    p = torch.tensor(PARAMS, dtype=torch.float64)[None]
    path = integrate(p[:, :3], p[:, 3:6], p[:, 6], p[:, 7], p[:, 8], 40)[0].numpy()
    assert np.abs(path - simulate(PARAMS, 40, FPS, G, SUB)).max() < 1e-6


def test_mirroring_a_flight_negates_sidespin_only() -> None:
    torch = pytest.importorskip("torch")
    from arcprior.flight import integrate
    p = torch.tensor(PARAMS, dtype=torch.float64)[None]
    path = integrate(p[:, :3], p[:, 3:6], p[:, 6], p[:, 7], p[:, 8], 30)
    flip = torch.tensor([1.0, -1.0, 1.0], dtype=torch.float64)
    mirrored = integrate(p[:, :3] * flip, p[:, 3:6] * flip, p[:, 6], p[:, 7], -p[:, 8], 30)
    assert (path * flip - mirrored).abs().max() < 1e-9


def test_pixels_and_rays_round_trip() -> None:
    torch = pytest.importorskip("torch")
    from arcprior.rays import ground_hits, project, rays_from_pixels
    rot = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])[None]
    trans = torch.tensor([[-50.0, 10.0, 60.0]])
    intr = torch.tensor([[1500.0, 1500.0, 960.0, 540.0]])
    point = torch.tensor([[40.0, 35.0, 3.0]])
    uv, depth = project(point, rot, trans, intr)
    assert depth.item() > 0
    centre, ray = rays_from_pixels(uv, rot, trans, intr)
    gap = torch.cross(point - centre, ray, dim=-1).norm()
    assert gap < 1e-3
    assert ground_hits(centre, ray)[0, 2].abs() < 1e-3


def test_reprojection_is_zero_on_the_ray_and_one_pixel_per_focal() -> None:
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace

    from arcprior.flight import reprojection_px
    centre = torch.zeros(1, 1, 3)
    ray = torch.tensor([[[1.0, 0.0, 0.0]]])
    batch = SimpleNamespace(centre=centre, ray=ray, focal_px=torch.tensor([[1000.0]]))
    assert reprojection_px(torch.tensor([[[50.0, 0.0, 0.0]]]), batch).item() < 1e-3
    offset = torch.tensor([[[50.0, 0.05, 0.0]]])
    assert reprojection_px(offset, batch).item() == pytest.approx(1.0, rel=1e-3)




def test_to_world_undoes_the_flight_frame() -> None:
    torch = pytest.importorskip("torch")
    if not (Config.FLIGHTS_DIR / "fold_a" / "train.npz").exists():
        pytest.skip("dataset not built")
    from arcprior.rays import FlightSet, to_world
    flights = FlightSet("a", "train", "cpu", views=(0,))
    index = torch.arange(8)
    batch = flights.real_batch(index, "R0", torch.Generator().manual_seed(0))
    world = to_world(batch.xyz, batch)
    size = batch.n_frames
    expected = flights.xyz[index, :size]
    known = batch.has_gt
    assert (world - expected)[known].abs().max() < 1e-2




def _synthetic_batch(torch, path, start_high, end_high):
    """Rays from two metres above a touchline camera through a known path, with no noise."""
    from types import SimpleNamespace
    centre = torch.tensor([20.0, -60.0, 25.0], dtype=path.dtype).expand(path.shape[0], 1, 3)
    centre = centre.expand(path.shape[0], path.shape[1], 3)
    ray = path - centre
    ray = ray / ray.norm(dim=-1, keepdim=True)
    present = torch.ones(path.shape[:2], dtype=torch.bool)
    is_known = torch.ones(path.shape[0], dtype=torch.bool)
    return SimpleNamespace(centre=centre, ray=ray, present=present, in_flight=present,
                           hit_end=path[:, -1], pitch_start=path[:, 0, :2],
                           has_pitch_start=is_known, has_pitch_end=is_known,
                           v_init=torch.zeros(path.shape[0], 3, dtype=path.dtype),
                           end_high=torch.tensor(end_high),
                           n_frames=path.shape[1]), torch.tensor(start_high)


def test_refine_recovers_a_planted_flight_from_most_starts() -> None:
    """Noise-free rays have a second, nearly equal solution, so only most starts find the truth."""
    torch = pytest.importorskip("torch")
    from arcprior.flight import integrate
    from arcprior.solver import RefineConfig, refine
    p = torch.tensor([[0.0, 0.0, 0.1, 14.0, 1.0, 7.0, 0.012, -0.008, 0.004, Config.G_MPS2]],
                     dtype=torch.float64)
    path = integrate(p[:, :3], p[:, 3:6], p[:, 6], p[:, 7], p[:, 8], 40)
    count = 16
    gen = torch.Generator().manual_seed(0)
    guess = p.repeat(count, 1)
    guess[:, 3:6] *= 1 + 0.2 * torch.randn(count, 3, generator=gen, dtype=torch.float64)
    guess[:, 7:9] = (torch.rand(count, 2, generator=gen, dtype=torch.float64) - 0.5) * 0.04
    batch, _ = _synthetic_batch(torch, path.repeat(count, 1, 1), [False] * count, [False] * count)
    theta, score = refine(guess, batch, RefineConfig(w_height=0.0, n_iters=60))
    fitted = integrate(theta[:, :3], theta[:, 3:6], theta[:, 6], theta[:, 7], theta[:, 8], 40)
    gaps = (fitted - path).norm(dim=-1).max(dim=1).values
    assert (gaps < 0.05).float().mean().item() >= 0.5
    assert score[gaps < 0.05].max().item() < 1e-3
    assert torch.abs(theta[gaps.argmin(), 6:9] - p[0, 6:9]).max().item() < 1e-3


def test_refine_without_spin_keeps_spin_at_zero_and_handles_a_false_detection() -> None:
    torch = pytest.importorskip("torch")
    from arcprior.flight import integrate
    from arcprior.solver import RefineConfig, refine
    p = torch.tensor([[0.0, 0.0, 0.1, 12.0, 0.0, 6.0, 0.012, 0.0, 0.0, Config.G_MPS2]],
                     dtype=torch.float64)
    path = integrate(p[:, :3], p[:, 3:6], p[:, 6], p[:, 7], p[:, 8], 30)
    batch, _ = _synthetic_batch(torch, path, [False], [False])
    batch.ray = batch.ray.clone()
    batch.ray[0, 12] = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    guess = p.clone()
    guess[:, 3:6] *= 0.9
    theta, _ = refine(guess, batch, RefineConfig(has_spin=False, w_height=0.0))
    assert theta[0, 7:9].abs().max().item() == 0.0
    fitted = integrate(theta[:, :3], theta[:, 3:6], theta[:, 6], theta[:, 7], theta[:, 8], 30)
    robust = (fitted - path).norm(dim=-1).median().item()
    plain, _ = refine(guess, batch, RefineConfig(has_spin=False, w_height=0.0, huber_m=1e6))
    fitted = integrate(plain[:, :3], plain[:, 3:6], plain[:, 6], plain[:, 7], plain[:, 8], 30)
    assert robust < 0.4
    assert robust < (fitted - path).norm(dim=-1).median().item()




def test_mirrored_batch_flips_the_end_anchor_only_sideways() -> None:
    torch = pytest.importorskip("torch")
    if not (Config.FLIGHTS_DIR / "fold_a" / "train.npz").exists():
        pytest.skip("dataset not built")
    from arcprior.rays import FlightSet
    flights = FlightSet("a", "train", "cpu", views=(0,))
    index = torch.arange(16)
    plain = flights.real_batch(index, "R0", torch.Generator().manual_seed(0))
    plain_again = build_mirrored(flights, index)
    assert (plain.hit_end[:, 0] - plain_again.hit_end[:, 0]).abs().max() < 1e-3
    assert (plain.hit_end[:, 1] + plain_again.hit_end[:, 1]).abs().max() < 1e-3


def build_mirrored(flights, index):
    """Every flight of the batch mirrored."""
    import torch

    from arcprior.rays import build_batch
    return build_batch(flights.uv_gt[index], flights.visible[index], flights.n[index],
                       flights.rot[index], flights.trans[index], flights.intr[index],
                       flights.xyz[index], flights.start_high[index], flights.end_high[index],
                       flights.oracle_params[index, 6:9], torch.ones(len(index)), index,
                       torch.ones(len(index), dtype=torch.bool), flights.pitch_xy[index])








def load_script(name: str):
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def pinhole(xyz: np.ndarray, rot: np.ndarray, trans: np.ndarray, intr: np.ndarray) -> np.ndarray:
    """Dataset-frame points (metres, height up) to pixels with per-point cameras."""
    camera = np.einsum("nij,nj->ni", rot, xyz * [1.0, 1.0, -1.0]) + trans
    return np.c_[intr[:, 0] * camera[:, 0] / camera[:, 2] + intr[:, 2],
                 intr[:, 1] * camera[:, 1] / camera[:, 2] + intr[:, 3]]


def test_video_camera_matches_pnlcalib_projection_in_both_orientations() -> None:
    build = load_script("video_inputs")
    # A camera in the stand, 25 m up, looking at the pitch centre (SoccerNet frame, z down).
    position = np.array([3.0, 70.0, -25.0])
    forward = -position / np.linalg.norm(position)
    right = np.cross(forward, [0.0, 0.0, -1.0])
    right /= np.linalg.norm(right)
    rot_sn = np.stack([right, np.cross(forward, right), forward])
    cam = {"rotation_matrix": rot_sn.tolist(), "position_meters": position.tolist(),
           "x_focal_length": 900.0, "y_focal_length": 900.0, "principal_point": [479.5, 269.5]}
    ball = np.array([60.0, 35.0, 4.0])            # dataset frame: metres, height up
    for is_flipped in (False, True):
        rot, trans, intr = build.to_dataset_camera(cam, (960, 540), is_flipped)
        uv = pinhole(ball[None], rot[None], trans[None], intr[None])[0]
        # PnLCalib's own projection of the same ball in its frame (centre origin, z down).
        flip = build.FLIP if is_flipped else np.eye(3)
        point_sn = flip @ (ball * [1.0, 1.0, -1.0] - build.CENTRE_M)
        x_cam = rot_sn @ (point_sn - np.array(cam["position_meters"]))
        expected = (np.array([900.0 * x_cam[0] / x_cam[2] + 479.5,
                              900.0 * x_cam[1] / x_cam[2] + 269.5]) * 2.0)
        assert np.allclose(uv, expected, atol=1e-6)
        # The pitch hit of the ray through the ball's foot point is the ball's xy.
        foot = pinhole((ball * [1.0, 1.0, 0.0])[None], rot[None], trans[None], intr[None])
        hit = build.ground_hit(foot, rot[None], trans[None], intr[None])[0]
        assert np.allclose(hit, ball[:2], atol=1e-6)


def test_video_tracker_follows_the_ball_through_a_miss_and_skips_a_far_false_box() -> None:
    build = load_script("video_inputs")
    frames = np.arange(6)
    boxes = {}
    for t in frames:
        ball = [100.0 + 10 * t, 200.0, 195.0, 8.0, 0.4]
        # The ball is missed at t = 3, where a brighter false box appears far away.
        boxes[int(t)] = np.array([ball] if t != 3 else [[900.0, 50.0, 45.0, 8.0, 0.9]])
    path = build.track(frames, boxes, vmax_px=60.0, accel_px=5.0, min_score=0.1)
    assert np.isnan(path[3, 0])
    kept = [t for t in frames if t != 3]
    assert np.allclose(path[kept, 0], 100.0 + 10 * np.array(kept))


def test_video_tracker_does_not_charge_a_fast_steady_ball_for_its_speed() -> None:
    build = load_script("video_inputs")
    frames = np.arange(12)
    boxes = {}
    for t in frames:
        # A displacement penalty would pick the slow box; a constant-velocity one must not.
        ball = [800.0 - 14 * t, 735.0 - 3 * t, 730.0 - 3 * t, 8.0, 0.5]
        player = [372.0 + 2 * t, 806.0, 800.0, 8.0, 0.45]
        boxes[int(t)] = np.array([player, ball])
    path = build.track(frames, boxes, vmax_px=80.0, accel_px=5.0, min_score=0.1)
    assert np.allclose(path[:, 0], 800.0 - 14 * frames)


def test_video_clutter_filter_drops_a_fixed_box_and_keeps_a_moving_ball() -> None:
    build = load_script("video_inputs")
    rows = []
    for t in range(200):
        rows.append((t, 500.0, 500.0, 508.0, 508.0, 0.9))                 # a floodlight
        rows.append((t, 10.0 + 9 * t, 300.0, 18.0 + 9 * t, 308.0, 0.4))   # the ball
    boxes = pd.DataFrame(rows, columns=["file_name", "x0", "y0", "x1", "y1", "score"])
    kept = build.drop_clutter(boxes, n_frames=200, scale=1.0)
    assert (kept["x0"] != 500.0).all()
    assert len(kept) == 200


def test_video_fill_track_interpolates_gaps_and_extends_ends_only_a_little() -> None:
    build = load_script("video_inputs")
    path = np.full((30, 5), np.nan)
    for t in range(8, 20):
        if t not in (12, 13):
            path[t] = [3.0 * t, 2.0 * t, 2.0 * t - 4, 8.0, 0.6]
    out, filled = build.fill_track(path)
    assert np.allclose(out[12:14, 0], [36.0, 39.0])
    assert np.allclose(out[3:8, 0], 3.0 * np.arange(3, 8))      # five frames before the first box
    assert np.isnan(out[:3, 0]).all() and np.isnan(out[25:, 0]).all()
    assert filled[12] and filled[3] and not filled[10]
    assert np.isnan(out[12, 4])


# ---------------------------------------------------------------- Trust (joint stage)








