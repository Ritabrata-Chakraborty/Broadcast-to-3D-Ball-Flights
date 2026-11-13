"""Contact-height draws, test cells, anchors, corruption, fill rule, robust prior, tokens."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")   # snv3d env

from arcprior.contacts import (CONTACT_MAX_M, GOOD_DELTA_M, TEST_WRONG_M,  # noqa: E402
                               TRAIN_WRONG_M, assign_cells, corrupt, fill_short_gaps,
                               ray_at_height, sample_contacts)
from arcprior.contacts import test_contact_values as contact_values  # noqa: E402
from arcprior.network import N_FEATURES, N_FEATURES_V5, build_batch3  # noqa: E402
from arcprior.rays import project  # noqa: E402
from arcprior.solver import robust  # noqa: E402


def test_sample_contacts_levels_and_cut_end():
    g = torch.Generator().manual_seed(0)
    n = 40_000
    z = torch.full((n,), 1.0)
    end_known = torch.arange(n) % 2 == 0
    out = sample_contacts(z, z, end_known, g)
    given = out["start_given"]
    assert abs(1 - given.float().mean().item() - 0.25) < 0.01
    delta = (out["h_start"] - 1.0)[given].abs()
    good = delta <= GOOD_DELTA_M + 1e-6
    assert abs(good.float().mean().item() - 0.45 / 0.75) < 0.02
    assert delta.max() <= TRAIN_WRONG_M[1] + 1e-6
    assert not out["end_given"][~end_known].any()


def test_sample_contacts_missing_gt_is_none():
    g = torch.Generator().manual_seed(1)
    z = torch.full((100,), float("nan"))
    out = sample_contacts(z, z, torch.ones(100, dtype=torch.bool), g)
    assert not out["start_given"].any() and not out["end_given"].any()


def test_test_contacts_shared_by_views_and_levels():
    group = np.array([5, 5, 5, 9, 9])
    z = np.array([1.0, 1.0, 1.0, 0.5, 0.5])
    level = np.array(["wrong", "wrong", "wrong", "correct", "correct"])
    value, given = contact_values(z, level, group, "start", seed=3)
    assert given.all()
    assert np.allclose(value[:3], value[0]) and np.allclose(value[3:], value[3])
    assert TEST_WRONG_M[0] - 1e-6 <= abs(value[0] - 1.0) <= TEST_WRONG_M[1] + 1e-6
    assert abs(value[3] - 0.5) <= GOOD_DELTA_M + 1e-6
    value, given = contact_values(z, np.array(["none"] * 5), group, "start", seed=3)
    assert not given.any()
    assert (contact_values(np.array([2.6]), np.array(["wrong"]), np.array([1]), "end",
                                0)[0] <= CONTACT_MAX_M + 1e-6).all()


def test_assign_cells_balanced_and_per_flight():
    rng = np.random.default_rng(0)
    groups = np.arange(360)
    group = np.repeat(groups, 3)
    apex = np.repeat(rng.uniform(1, 8, 360), 3)
    cells = assign_cells(group, apex, partial=False, seed=1)
    keys = list(zip(cells["track"], cells["start_level"], cells["end_level"]))
    per_group = {}
    for g, k in zip(group, keys):
        per_group.setdefault(g, set()).add(k)
    assert all(len(v) == 1 for v in per_group.values())
    counts = np.unique([next(iter(v)) for v in per_group.values()], axis=0, return_counts=True)[1]
    assert len(counts) == 18 and counts.max() - counts.min() <= 2
    part = assign_cells(group, apex, partial=True, seed=1)
    assert set(part["end_level"]) == {"none"}


def camera() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A broadcast-like camera 30 m high and 40 m back, looking at the pitch (look-at, in the
    height-down frame that arcprior.rays.project uses)."""
    sign = torch.tensor([1.0, 1.0, -1.0])
    centre = torch.tensor([0.0, -40.0, 30.0]) * sign
    target = torch.tensor([0.0, 20.0, 0.0]) * sign
    z = (target - centre) / (target - centre).norm()
    x = torch.tensor([1.0, 0.0, 0.0])
    y = torch.linalg.cross(z, x)
    rot = torch.stack([x, y, z])
    trans = -rot @ centre
    intr = torch.tensor([1500.0, 1500.0, 960.0, 540.0])
    return rot, trans, intr


def test_ray_at_height_inverts_projection():
    rot, trans, intr = camera()
    point = torch.tensor([[3.0, 12.0, 1.4]])
    uv, depth = project(point, rot, trans, intr)
    assert depth.item() > 0
    xy = ray_at_height(uv, rot, trans, intr, torch.tensor([1.4]))
    assert torch.allclose(xy, point[:, :2], atol=1e-3)


def test_fill_short_gaps():
    uv = torch.arange(40, dtype=torch.float32)[None, :, None].repeat(1, 1, 2)
    uv[0, 5:8] = float("nan")          # 3 missing: filled
    uv[0, 15:27] = float("nan")        # 12 missing: kept
    uv[0, :2] = float("nan")           # at the start: kept
    out, filled = fill_short_gaps(uv, torch.tensor([40]))
    assert filled[0, 5:8].all() and torch.allclose(out[0, 5:8, 0], torch.tensor([5.0, 6.0, 7.0]))
    assert torch.isnan(out[0, 15:27, 0]).all() and not filled[0, 15:27].any()
    assert torch.isnan(out[0, :2, 0]).all()


def test_corrupt_clean_min_missing_and_size():
    g = torch.Generator().manual_seed(2)
    uv = torch.rand(4, 60, 2)
    size = torch.full((4, 60), 7.0)
    donor_size = torch.full((4, 60), 99.0)
    n = torch.full((4,), 50)
    view = torch.zeros(4, dtype=torch.long)
    gaps = {0: np.array([3, 5]), 1: np.array([10])}
    same, same_size = corrupt(uv, size, n, view, gaps, uv, size, view, n, g, p_clean=1.0,
                              p_wrong=0.0)
    assert torch.equal(same, uv) and torch.equal(same_size, size)
    out, out_size = corrupt(uv, size, n, view, gaps, uv, size, view, n, g, p_clean=0.0,
                            p_wrong=0.0, min_missing=0.2, max_missing=0.3)
    missing = torch.isnan(out[:, :50, 0])
    assert (missing.float().mean(dim=1) >= 0.1).all()   # overlapping runs remove a little less
    assert torch.isnan(out_size[:, :50][missing]).all()
    donors = torch.rand(4, 60, 2) + 5.0
    wrong, wrong_size = corrupt(uv, size, n, view, gaps, donors, donor_size, view, n, g,
                                p_clean=1.0, p_wrong=1.0)
    moved = (wrong != uv).any(dim=-1)
    assert moved.any() and (wrong_size[moved] == 99.0).all()


def test_robust_residual():
    r = torch.tensor([-3.0, -1.0, 0.5, 1.0, 4.0])
    out = robust(r)
    assert torch.allclose(out[1:4], r[1:4])
    assert torch.allclose(out[[0, 4]] ** 2, 2 * r[[0, 4]].abs() - 1)


class Flights:
    """A minimal FlightSet for build_batch3."""

    def __init__(self, count: int = 3, size: int = 20) -> None:
        rot, trans, intr = camera()
        t = torch.linspace(0, 1, size)
        xyz = torch.stack([5 * t, 10 + 20 * t, 4 * t * (1 - t) * 4 + 0.2], dim=-1)
        uv, _ = project(xyz, rot, trans, intr)
        self.device = "cpu"
        self.n = torch.full((count,), size)
        self.uv_det = uv[None].repeat(count, 1, 1)
        self.rot = rot.expand(count, size, 3, 3).clone()
        self.trans = trans.expand(count, size, 3).clone()
        self.intr = intr.expand(count, size, 4).clone()
        self.xyz = xyz[None].repeat(count, 1, 1)
        self.pitch_xy = torch.full((count, size, 2), float("nan"))
        self.oracle_params = torch.zeros(count, 9)
        self.size_det = torch.full((count, size), 10.0)
        self.view = torch.zeros(count, dtype=torch.long)
        self.start_high = torch.zeros(count, dtype=torch.bool)
        self.end_high = torch.zeros(count, dtype=torch.bool)


def test_build_batch_token_layouts():
    flights = Flights()
    index = torch.arange(3)
    legacy = build_batch3(flights, index)
    assert legacy.tokens.shape[-1] == N_FEATURES
    contact = {"h_start": torch.full((3,), 0.3), "h_end": torch.full((3,), 0.2),
               "start_given": torch.tensor([True, False, True]),
               "end_given": torch.tensor([True, True, False])}
    batch = build_batch3(flights, index, contact=contact)
    assert batch.tokens.shape[-1] == N_FEATURES_V5
    assert batch.has_pitch_start.tolist() == [True, False, True]
    cut = build_batch3(flights, index, torch.full((3,), 10), contact=contact)
    assert not cut.end_given.any()
