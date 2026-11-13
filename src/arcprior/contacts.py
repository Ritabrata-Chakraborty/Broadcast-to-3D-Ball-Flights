"""The user's contact heights (an input), input corruption and the mixed test sets.

A user gives each end's contact height as one number (ball centre, m) or nothing. Training draws,
per end, none / good / wrong around the GT height; the test sets give every physical flight one
balanced combination of track (clean / masked) and height levels (correct / wrong / none per end;
partial flights: start only). Anchors are the input's own ball ray at the contact frame (or the
nearest detection within a few frames), cut at the given height minus the ball radius. Corruption
mimics the video tracker: runs of missing frames with lengths from the fold's training gaps, and a
run of wrong points taken from another flight of the same view; then short gaps are filled.
"""
import numpy as np
import torch

import config as Config
from arcprior.rays import rays_from_pixels

CONTACT_MAX_M = 2.7                     # contact heights in training: p99.9 2.94, 0.13 % above
GOOD_DELTA_M = 0.15                     # good / correct: delta ~ U[-0.15, 0.15]
TRAIN_WRONG_M = (0.15, 0.80)            # wrong in training: abs(delta) ~ U[0.15, 0.80]
TEST_WRONG_M = (0.40, 0.80)             # wrong at test: abs(delta) ~ U[0.40, 0.80]
TRAIN_LEVELS = (("none", 0.25), ("good", 0.45), ("wrong", 0.30))
LEVELS = ("correct", "wrong", "none")
MAX_FILL = 10                           # interior gaps up to this length are interpolated
TEST_MISSING = (0.10, 0.40)             # masked test track: missing share ~ U[0.1, 0.4]
TEST_P_WRONG = 0.5                      # masked test track: a wrong-point run with this chance
SEEDS = {"val": 7101, "test": 7202, "train": 7303}


# ---------------------------------------------------------------- Contact heights


def contact_value(z, level_is_good, level_is_wrong, u, sign, wrong: tuple[float, float]):
    """GT + delta for good (U[-0.15, 0.15]) or wrong (sign * U[wrong]) levels, clipped."""
    good = (2.0 * u - 1.0) * GOOD_DELTA_M
    bad = sign * (wrong[0] + (wrong[1] - wrong[0]) * u)
    delta = good * level_is_good + bad * level_is_wrong
    return (z + delta).clip(0.0, CONTACT_MAX_M)


def sample_contacts(z_start: torch.Tensor, z_end: torch.Tensor, end_known: torch.Tensor,
                    generator: torch.Generator) -> dict:
    """Training draw per end: none .25 / good .45 / wrong .30; never the end of a cut flight."""
    out = {}
    for name, z, allowed in (("start", z_start, torch.ones_like(end_known)),
                             ("end", z_end, end_known)):
        pick = torch.rand(len(z), generator=generator)
        u = torch.rand(len(z), generator=generator)
        sign = torch.where(torch.rand(len(z), generator=generator) < 0.5, -1.0, 1.0)
        p_none, p_good = TRAIN_LEVELS[0][1], TRAIN_LEVELS[1][1]
        is_good = ((pick >= p_none) & (pick < p_none + p_good)).float()
        is_wrong = (pick >= p_none + p_good).float()
        given = (pick >= p_none) & torch.isfinite(z) & allowed
        value = contact_value(torch.nan_to_num(z), is_good, is_wrong, u, sign, TRAIN_WRONG_M)
        out[f"h_{name}"] = torch.where(given, value, torch.zeros_like(z))
        out[f"{name}_given"] = given
    return out


def assign_cells(group: np.ndarray, apex: np.ndarray, partial: bool, seed: int) -> dict:
    """One balanced cell per physical flight: track (clean / masked) x start x end contact level.

    Cells are dealt round-robin over a seeded order within apex strata (below / above 3 m); the
    partial set has no end factor (the end is never given). Returns per-flight level arrays.
    """
    tracks, starts = ("clean", "masked"), LEVELS
    ends = ("none",) if partial else LEVELS
    cells = [(t, s, e) for t in tracks for s in starts for e in ends]
    groups = np.unique(group)
    apex_of = {g: np.nanmax(apex[group == g]) for g in groups}
    rng = np.random.default_rng(seed)
    choice = {}
    for stratum in (False, True):
        members = np.array([g for g in groups if (apex_of[g] >= 3.0) == stratum])
        members = members[rng.permutation(len(members))]
        offset = int(rng.integers(len(cells)))
        for k, g in enumerate(members):
            choice[g] = cells[(k + offset) % len(cells)]
    return {"track": np.array([choice[g][0] for g in group]),
            "start_level": np.array([choice[g][1] for g in group]),
            "end_level": np.array([choice[g][2] for g in group])}


def test_contact_values(z: np.ndarray, level: np.ndarray, group: np.ndarray, end: str,
                        seed: int) -> tuple[np.ndarray, np.ndarray]:
    """The user's number per view-flight for its level; one draw per physical flight and end."""
    draws = np.array([np.random.default_rng([seed, int(g), end == "end"]).random(2)
                      for g in group]).reshape(-1, 2)
    sign = np.where(draws[:, 1] < 0.5, -1.0, 1.0)
    value = contact_value(np.nan_to_num(z), (level == "correct").astype(float),
                          (level == "wrong").astype(float), draws[:, 0], sign, TEST_WRONG_M)
    given = (level != "none") & np.isfinite(z)
    return np.where(given, value, 0.0).astype(np.float32), given


def ray_at_height(uv: torch.Tensor, rot: torch.Tensor, trans: torch.Tensor, intr: torch.Tensor,
                  height: torch.Tensor) -> torch.Tensor:
    """Pitch xy (m) where each pixel's ray reaches a height (m); NaN if it never does."""
    centre, ray = rays_from_pixels(uv, rot, trans, intr)
    down = ray[..., 2]
    along = (height - centre[..., 2]) / torch.where(down < 0, down, torch.full_like(down, -1.0))
    ok = (down < 0) & (along > 0) & (along <= Config.MAX_RAY_M)
    xy = centre[..., :2] + ray[..., :2] * along[..., None]
    return torch.where(ok[..., None], xy, torch.full_like(xy, float("nan")))


def contact_anchors(uv: torch.Tensor, rot: torch.Tensor, trans: torch.Tensor, intr: torch.Tensor,
                    present: torch.Tensor, n_full: torch.Tensor, contact: dict) -> dict:
    """Start / end anchors: the nearest present frame inside the search window, cut at the height.

    Returns pitch xy per end, whether it exists, and its frame (for the start-box width).
    """
    rows = torch.arange(len(n_full))
    out = {}
    search = Config.ANCHOR_SEARCH_FRAMES
    for name in ("start", "end"):
        xy = torch.full((len(n_full), 2), float("nan"))
        frame_used = torch.full((len(n_full),), -1, dtype=torch.long)
        for k in range(search, -1, -1):          # nearest frame wins: it is written last
            frame = k if name == "start" else n_full - 1 - k
            frame = torch.as_tensor(frame).expand(len(n_full)).clamp(min=0)
            ok = present[rows, frame] & contact[f"{name}_given"]
            height = (contact[f"h_{name}"] - Config.BALL_RADIUS_M).clamp(min=0.0)
            hit = ray_at_height(uv[rows, frame], rot[rows, frame], trans[rows, frame],
                                intr[rows, frame], height)
            ok = ok & torch.isfinite(hit).all(dim=1)
            xy = torch.where(ok[:, None], hit, xy)
            frame_used = torch.where(ok, frame, frame_used)
        out[f"{name}_xy"] = xy
        out[f"has_{name}"] = frame_used >= 0
        out[f"{name}_frame"] = frame_used
    return out


# ---------------------------------------------------------------- Input corruption


def gap_table(uv: np.ndarray, n: np.ndarray, view: np.ndarray) -> dict:
    """Lengths of runs of missing frames inside flights, per view type (0 broadcast, 1 static)."""
    runs = {0: [], 1: []}
    for i in range(len(n)):
        missing = ~np.isfinite(uv[i, :n[i], 0])
        edges = np.diff(np.r_[0, missing.astype(int), 0])
        starts, stops = np.where(edges == 1)[0], np.where(edges == -1)[0]
        runs[int(view[i] != 0)] += (stops - starts).tolist()
    return {k: np.asarray(v if v else [1], dtype=np.int64) for k, v in runs.items()}


def fill_short_gaps(uv: torch.Tensor, n: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Interpolate interior gaps of at most MAX_FILL frames (the video rule); flag those frames."""
    out = uv.clone()
    filled = torch.zeros(uv.shape[:2], dtype=torch.bool)
    for i in range(len(n)):
        m = int(n[i])
        known = torch.isfinite(uv[i, :m, 0]).numpy()
        idx = np.where(known)[0]
        if len(idx) < 2:
            continue
        for a, b in zip(idx[:-1], idx[1:]):
            if 1 < b - a <= MAX_FILL + 1:
                w = torch.linspace(0, 1, b - a + 1)[1:-1, None]
                out[i, a + 1:b] = uv[i, a] * (1 - w) + uv[i, b] * w
                filled[i, a + 1:b] = True
    return out, filled


def corrupt(uv: torch.Tensor, size: torch.Tensor, n: torch.Tensor, view: torch.Tensor,
            gaps: dict, donors: torch.Tensor, donor_size: torch.Tensor, donor_view: torch.Tensor,
            donor_n: torch.Tensor, generator: torch.Generator, p_clean: float = 0.3,
            max_missing: float = 0.4, p_wrong: float = 0.3, min_missing: float = 0.0
            ) -> tuple[torch.Tensor, torch.Tensor]:
    """Runs of missing frames (share ~ U[min, max]) and a run of wrong points (another flight).

    The box size moves with the point: missing frames lose it, wrong points carry the donor's, so
    no frame keeps the true ball's size where its pixel is not the ball's.
    """
    out = uv.clone()
    out_size = size.clone()
    for i in range(len(n)):
        m = int(n[i])
        if float(torch.rand(1, generator=generator)) >= p_clean:
            share = min_missing + float(torch.rand(1, generator=generator)) * \
                (max_missing - min_missing)
            target = share * m
            lengths = gaps[int(view[i] != 0)]
            removed, at_edge = 0, float(torch.rand(1, generator=generator)) < 0.3
            while removed < target:
                length = int(lengths[int(torch.randint(len(lengths), (1,), generator=generator))])
                length = max(1, min(length, m - 1))
                if at_edge:
                    begin = 0 if float(torch.rand(1, generator=generator)) < 0.5 else m - length
                    at_edge = False
                else:
                    begin = int(torch.randint(max(1, m - length + 1), (1,), generator=generator))
                out[i, begin:begin + length] = float("nan")
                out_size[i, begin:begin + length] = float("nan")
                removed += length
        if float(torch.rand(1, generator=generator)) < p_wrong:
            pool = torch.where(donor_view == view[i])[0]
            pick = pool[int(torch.randint(len(pool), (1,), generator=generator))]
            donor, donor_len = donors[pick], int(donor_n[pick])
            share = 0.1 + 0.2 * float(torch.rand(1, generator=generator))
            length = max(1, min(int(round(share * m)), donor_len))
            begin = int(torch.randint(max(1, m - length + 1), (1,), generator=generator))
            shift = int(torch.randint(donor_len - length + 1, (1,), generator=generator))
            piece = donor[shift:shift + length]
            piece_size = donor_size[pick, shift:shift + length]
            keep = torch.isfinite(piece[:, 0])
            out[i, begin:begin + length][keep] = piece[keep]
            out_size[i, begin:begin + length][keep] = piece_size[keep]
    return out, out_size


# ---------------------------------------------------------------- Test sets


def build_test_set(arrays: dict, kind: str, split: str, gaps: dict,
                   fixed: str | None = None) -> dict:
    """One split of a mixed test set from an input setting's arrays.

    kind "full" or "part"; the train split is clean with correct contacts (in-sample check).
    fixed (a LEVELS name): every flight clean with that contact level at both ends (the paired
    measurement of what the heights are worth, results 5.1).
    Adds: uv_det (masked where assigned, then filled), filled, h_start / h_end, start_given /
    end_given, pitch_xy (anchors at the given heights), start_slope, and the assigned levels
    (track_masked; start_level / end_level as indices into LEVELS).
    """
    seed = SEEDS[split]
    group, n = arrays["group"], arrays["n"]
    frames = np.arange(arrays["xyz"].shape[1])[None]
    apex = np.nanmax(np.where(frames < n[:, None], arrays["xyz"][..., 2], np.nan), axis=1)
    if split == "train" or fixed is not None:
        level = fixed or "correct"
        cells = {"track": np.full(len(n), "clean"), "start_level": np.full(len(n), level),
                 "end_level": np.full(len(n), level)}
    else:
        cells = assign_cells(group, apex, kind == "part", seed)
    uv = torch.as_tensor(arrays["uv_det"]).float()
    size = torch.as_tensor(arrays["size_det"]).float()
    clean_uv, clean_size = uv.clone(), size.clone()     # donors: the unmasked tracks
    n_t = torch.as_tensor(n).long()
    view = torch.as_tensor(arrays["view"]).long()
    for i in np.where(cells["track"] == "masked")[0]:
        generator = torch.Generator().manual_seed(seed * 1_000_000 + int(group[i]) * 10 +
                                                  int(view[i]))
        uv[i:i + 1], size[i:i + 1] = corrupt(
            uv[i:i + 1], size[i:i + 1], n_t[i:i + 1], view[i:i + 1], gaps, clean_uv, clean_size,
            view, n_t, generator, p_clean=0.0, max_missing=TEST_MISSING[1], p_wrong=TEST_P_WRONG,
            min_missing=TEST_MISSING[0])
    uv, filled = fill_short_gaps(uv, n_t)
    contact = {}
    for end in ("start", "end"):
        value, given = test_contact_values(arrays[f"z_{end}"], cells[f"{end}_level"], group, end,
                                           seed)
        contact[f"h_{end}"], contact[f"{end}_given"] = value, given
    tensors = {k: torch.as_tensor(v) for k, v in contact.items()}
    present = torch.isfinite(uv[..., 0]) & (torch.as_tensor(frames) < n_t[:, None])
    rot, trans, intr = (torch.as_tensor(arrays[k]).float() for k in ("rot", "trans", "intr"))
    anchors = contact_anchors(uv, rot, trans, intr, present, n_t, tensors)
    pitch = np.full(arrays["pitch_xy"].shape, np.nan, dtype=np.float32)
    rows = np.arange(len(n))
    pitch[rows, 0] = anchors["start_xy"].numpy()
    pitch[rows, n - 1] = anchors["end_xy"].numpy()
    start = anchors["start_frame"].clamp(min=0)
    _, ray = rays_from_pixels(uv[rows, start].nan_to_num(), rot[rows, start], trans[rows, start],
                              intr[rows, start])
    slope = ray[:, :2].norm(dim=1) / ray[:, 2].abs().clamp(min=1e-3)
    out = dict(arrays)
    out.update(uv_det=uv.numpy().astype(np.float32), size_det=size.numpy().astype(np.float32),
               filled=filled.numpy(), pitch_xy=pitch,
               start_slope=slope.numpy().astype(np.float32),
               start_given=contact["start_given"] & anchors["has_start"].numpy(),
               end_given=contact["end_given"] & anchors["has_end"].numpy(),
               h_start=contact["h_start"], h_end=contact["h_end"],
               track_masked=cells["track"] == "masked",
               start_level=np.array([LEVELS.index(x) for x in cells["start_level"]], np.int8),
               end_level=np.array([LEVELS.index(x) for x in cells["end_level"]], np.int8))
    return out
