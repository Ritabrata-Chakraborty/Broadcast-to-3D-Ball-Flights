"""Contact anchors, cut batches, the ArcPrior network, the solver priors, Omega and the metrics."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")   # snv3d env

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import config as Config  # noqa: E402
from build_folds import contact_height, oracle_contacts, ray_at_height  # noqa: E402
from arcprior.network import (ArcPrior, build_batch3, compute_losses3, cut_at,  # noqa: E402
                              decode3, sample_cuts)
from arcprior.flight import integrate  # noqa: E402
from arcprior.observability import observability, weak_projection  # noqa: E402
from arcprior.solver import (N_PRIOR, RefineConfig, pack, pack_prior,  # noqa: E402
                             prior_residual, refine)
from arcprior.metrics import bootstrap, e_full, e_part, holm  # noqa: E402

ROT = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])   # looks along +y, level


def camera(position: np.ndarray, pitch_deg: float = 20.0) -> tuple:
    """A camera at position (m, height up) looking along +y, tilted down by pitch_deg."""
    a = np.deg2rad(pitch_deg)
    tilt = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    rot = tilt @ ROT
    trans = -rot @ (position * [1.0, 1.0, -1.0])
    intr = np.array([1500.0, 1500.0, 960.0, 540.0])
    return rot, trans, intr


def pixel(point: np.ndarray, rot, trans, intr) -> np.ndarray:
    x_cam = rot @ (point * [1.0, 1.0, -1.0]) + trans
    return np.array([intr[0] * x_cam[0] / x_cam[2] + intr[2],
                     intr[1] * x_cam[1] / x_cam[2] + intr[3]])


def test_ray_at_height_recovers_the_point():
    rot, trans, intr = camera(np.array([0.0, -40.0, 15.0]))
    for point in (np.array([3.0, 5.0, 0.0]), np.array([-4.0, 10.0, 1.7])):
        hit = ray_at_height(pixel(point, rot, trans, intr), rot, trans, intr, point[2])
        np.testing.assert_allclose(hit, point[:2], atol=1e-6)


def test_ray_above_the_horizon_has_no_anchor():
    rot, trans, intr = camera(np.array([0.0, -40.0, 15.0]), pitch_deg=0.0)
    above = pixel(np.array([0.0, 400.0, 40.0]), rot, trans, intr)   # higher than the camera
    assert np.isnan(ray_at_height(above, rot, trans, intr, 0.0)).all()


def test_oracle_contacts_type_and_anchor():
    n, size = 10, 12
    rot, trans, intr = camera(np.array([0.0, -40.0, 15.0]))
    xyz = np.full((size, 3), np.nan)
    xyz[:n] = np.c_[np.linspace(0, 9, n), np.full(n, 5.0), np.linspace(0.11, 2.0, n)]
    uv = np.full((size, 2), np.nan)
    for t in range(n):
        bottom = xyz[t] - [0.0, 0.0, Config.BALL_RADIUS_M]
        uv[t] = pixel(bottom, rot, trans, intr)
    stack = [np.repeat(x[None], size, 0) for x in (rot, trans, intr)]
    out = oracle_contacts(n, xyz, uv, *stack, size)
    assert not out["start_high"] and out["end_high"]
    np.testing.assert_allclose(out["pitch_xy"][0], xyz[0, :2], atol=1e-6)
    np.testing.assert_allclose(out["pitch_xy"][n - 1], xyz[n - 1, :2], atol=1e-6)
    assert np.isnan(out["pitch_xy"][1:n - 1]).all()


def test_contact_height_skips_missing_frames():
    xyz = np.full((5, 3), np.nan)
    xyz[2, 2] = 0.4
    assert contact_height(xyz, 0, 1) == pytest.approx(0.4)


# ---------------------------------------------------------------- Flights in a FlightSet layout


def toy_flights(count: int = 3, n: int = 20, size: int = 24) -> SimpleNamespace:
    rot, trans, intr = camera(np.array([0.0, -45.0, 18.0]))
    start = torch.tensor([[0.0, 0.0, 0.11]]).repeat(count, 1)
    velocity = torch.tensor([[12.0, 3.0, 8.0]]).repeat(count, 1)
    zeros = torch.zeros(count)
    path = integrate(start, velocity, zeros + 0.01, zeros, zeros, size).numpy()
    xyz = path.copy()
    xyz[:, n:] = np.nan
    uv = np.stack([[pixel(p - [0, 0, 0.11], rot, trans, intr) for p in flight] for flight in path])
    uv[:, n:] = np.nan
    pitch = np.full((count, size, 2), np.nan)
    pitch[:, 0] = path[:, 0, :2]
    pitch[:, n - 1] = path[:, n - 1, :2]
    oracle = np.c_[path[:, 0], np.repeat([[12.0, 3.0, 8.0]], count, 0), np.full((count, 3), 0.0)]
    oracle[:, 6] = 0.01

    def t(x, dtype=torch.float32):
        return torch.as_tensor(np.asarray(x), dtype=dtype)

    def tile(x: np.ndarray) -> torch.Tensor:
        return t(np.repeat(x[None, None], count, 0).repeat(size, 1))

    return SimpleNamespace(
        device="cpu", n=torch.full((count,), n), uv_det=t(uv), rot=tile(rot), trans=tile(trans),
        intr=tile(intr), xyz=t(xyz),
        pitch_xy=t(pitch), oracle_params=t(oracle), size_det=t(np.full((count, size), 12.0)),
        view=torch.zeros(count, dtype=torch.long), start_high=torch.zeros(count, dtype=torch.bool),
        end_high=torch.zeros(count, dtype=torch.bool))


def test_cut_batch_hides_the_future_and_the_end():
    flights = toy_flights()
    index = torch.arange(3)
    full = build_batch3(flights, index)
    cut = build_batch3(flights, index, cut_at(flights.n[index], 0.5))
    assert bool(full.end_known.all()) and not bool(cut.end_known.any())
    assert not bool(cut.has_pitch_end.any())
    assert int(cut.present.sum(dim=1).max()) == 10
    # The solver sees only the seen frames: its flight length is the cut, never the true length.
    assert pack(cut)["end_frame"].tolist() == [9, 9, 9]
    assert pack(full)["end_frame"].tolist() == [19, 19, 19]
    # Targets still span the whole flight for the training loss.
    assert int(cut.has_gt.sum(dim=1)[0]) == 20


def test_sample_cuts_range():
    n = torch.full((1000,), 40)
    cuts = sample_cuts(n, torch.Generator().manual_seed(0))
    share_full = (cuts == 40).float().mean()
    assert 0.4 < share_full < 0.6
    assert int(cuts.min()) >= 12 and int(cuts.max()) <= 40


def test_arcprior_shapes_and_losses():
    flights = toy_flights()
    batch = build_batch3(flights, torch.arange(3))
    model = ArcPrior(4, d_model=32, n_layers=2)
    out = decode3(model(batch.tokens, batch.padding), batch)
    assert out["theta"].shape == (3, 4, 9) and out["path"].shape == (3, 4, batch.n_frames, 3)
    assert (out["std"] > 0).all()
    loss, parts = compute_losses3(out, batch, {"path": 1.0, "nll": 0.05, "select": 0.1})
    assert torch.isfinite(loss) and set(parts) >= {"path", "nll", "select"}
    loss.backward()


def test_prior_kinds():
    theta = torch.randn(5, 10, dtype=torch.float64)
    net = torch.randn(5, 10, dtype=torch.float64)
    std = torch.rand(5, N_PRIOR, dtype=torch.float64) + 0.5
    eye = torch.eye(N_PRIOR, dtype=torch.float64).expand(5, -1, -1)
    gauss = prior_residual(theta, pack_prior(net, std, eye), "gauss")
    sub = prior_residual(theta, pack_prior(net, std, eye), "subspace")
    torch.testing.assert_close(gauss, sub)
    zero = prior_residual(theta, pack_prior(net, std, torch.zeros_like(eye)), "subspace")
    assert torch.count_nonzero(zero) == 0
    assert prior_residual(theta, net, "old").shape == (5, 4)


def test_weak_projection_is_a_projection():
    flights = toy_flights()
    batch = build_batch3(flights, torch.arange(3))
    theta = torch.tensor([[0.0, 0.0, 0.11, 12.0, 3.0, 8.0, 0.01, 0.0, 0.0, 9.806]] * 3,
                         dtype=torch.float64)
    std = torch.ones(3, N_PRIOR, dtype=torch.float64)
    proj = weak_projection(theta, std, batch, RefineConfig(), 1.0)
    torch.testing.assert_close(proj @ proj, proj, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(proj, proj.transpose(1, 2), atol=1e-9, rtol=1e-9)


def test_omega_falls_with_more_data():
    flights = toy_flights()
    index = torch.arange(3)
    cfg = RefineConfig(has_spin=False, has_free_g=True, k3_max=1.0)
    omegas = []
    for share in (0.3, 1.0):
        batch = build_batch3(flights, index, cut_at(flights.n[index], share))
        guess = torch.tensor([[0.0, 0.0, 0.11, 11.0, 2.0, 9.0, 0.02, 0.0, 0.0, 9.806]] * 3,
                             dtype=torch.float64)
        theta, _ = refine(guess, batch, cfg)
        omegas.append(observability(theta, batch, cfg, None, batch.n_frames,
                                    batch.in_flight)["omega"].mean())
    assert torch.isfinite(omegas[1]) and omegas[1] < omegas[0]


# ---------------------------------------------------------------- Metrics


def toy_table() -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(0)
    for group in range(30):
        for view in (0, 1, 2):
            for share in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
                for method, offset in (("a", 0.0), ("b", 0.5)):
                    rows.append({"method": method, "group": group, "view": view, "share": share,
                                 "rmse_cap": 1.0 + offset + rng.random() * 0.1})
    return pd.DataFrame(rows)


def test_e_full_averages_views_per_flight():
    table = toy_table()
    static = e_full(table, "a", "static")
    assert len(static) == 30
    expected = table[(table.method == "a") & (table.view > 0) & (table.share == 1.0)].groupby(
        "group")["rmse_cap"].mean()
    pd.testing.assert_series_equal(static, expected)


def test_e_part_needs_every_share(monkeypatch):
    import arcprior.metrics as report
    monkeypatch.setattr(report, "part_population", lambda: set(range(30)))
    table = toy_table()
    assert len(e_part(table, "a", "broadcast")) == 30
    missing = table[~((table.group == 0) & np.isclose(table.share, 0.5))]
    assert len(e_part(missing, "a", "broadcast")) == 29


def test_bootstrap_and_holm():
    table = toy_table()
    out = bootstrap(e_full(table, "a", "broadcast"), e_full(table, "b", "broadcast"), draws=500)
    assert out["gap"] < 0 and out["high"] < 0 and out["p"] < 0.01
    adjusted = holm({"x": 0.01, "y": 0.04, "z": 0.03})
    assert adjusted["x"] == pytest.approx(0.03) and adjusted["z"] == pytest.approx(0.06)
    assert adjusted["y"] == pytest.approx(0.06)


