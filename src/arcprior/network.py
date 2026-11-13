"""ArcPrior: ray tokens with contact heights, ball size, end flags and view; K flight hypotheses,
each with a predicted spread, for full or cut flights. Time is seconds from the start contact.

The flight frame has its origin at the start anchor and +x towards the end anchor, or towards the
last seen ray's pitch hit when the end is hidden. Physics follows arcprior.flight.
"""
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

import config as Config
from arcprior.contacts import contact_anchors
from arcprior.rays import ground_hits, rays_from_pixels
from arcprior.flight import integrate, limit_speed, reprojection_px

N_VIEWS = len(Config.VIEWS)
N_FOURIER = 4
# Base token: centre 3, ray 3, delta 3, pitch hit 2, time, first, last, end contact, start high,
# end high, end anchor 2, has start, has end, end known, log focal, size depth, has size, view 6,
# Fourier 8 (39 features).
N_FEATURES = 39
# ArcPrior's token replaces the two contact-type flags by the user's contact heights (given,
# height / 2 for each end) and adds a per-frame flag for interpolated (filled) track points.
N_FEATURES_V5 = 42
SIZE_FEATURES = slice(23, 25)
N_PARAMS = 9          # start x, y, z (flight frame), velocity (3), k3, kl, ks
PARAM_SCALE = (1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 0.02, 0.01, 0.01)
LOG_STD_RANGE = (-4.0, 2.0)
LOSS_NAMES = ("path", "reproj", "end", "height", "params", "nll", "select")


@dataclass
class Batch3:
    """One batch in each flight's own frame. Arrays run over the full flight; tokens to the cut."""
    tokens: torch.Tensor
    padding: torch.Tensor
    centre: torch.Tensor
    ray: torch.Tensor
    focal_px: torch.Tensor
    present: torch.Tensor
    in_flight: torch.Tensor          # frames of the full flight
    seen: torch.Tensor               # frames up to the cut
    xyz: torch.Tensor
    has_gt: torch.Tensor
    labels: torch.Tensor             # oracle k3, kl, ks
    label_weight: torch.Tensor
    oracle: torch.Tensor             # oracle 9 parameters in the flight frame
    anchor_centre: torch.Tensor
    anchor_ray: torch.Tensor
    anchor_time: torch.Tensor
    hit_end: torch.Tensor
    v_init: torch.Tensor
    end_high: torch.Tensor
    end_known: torch.Tensor
    index: torch.Tensor
    n_frames: int
    n_full: torch.Tensor
    cut: torch.Tensor
    origin: torch.Tensor
    to_flight: torch.Tensor
    pitch_start: torch.Tensor
    has_pitch_start: torch.Tensor
    has_pitch_end: torch.Tensor
    mirrored: torch.Tensor
    has_obs: torch.Tensor
    # The user's contact heights (m, ball centre) and whether each is given.
    h_start: torch.Tensor | None = None
    h_end: torch.Tensor | None = None
    start_given: torch.Tensor | None = None
    end_given: torch.Tensor | None = None
    start_slope: torch.Tensor | None = None   # horizontal m per vertical m on the anchor ray


def contact_of(flights, index: torch.Tensor) -> dict | None:
    """The user's contact heights stored with a test condition's data, if any."""
    if not hasattr(flights, "h_start"):
        return None
    return {"h_start": flights.h_start[index].float(), "h_end": flights.h_end[index].float(),
            "start_given": flights.start_given[index].bool(),
            "end_given": flights.end_given[index].bool()}


def build_batch3(flights, index: torch.Tensor, cut: torch.Tensor | None = None,
                 mirrored: torch.Tensor | None = None, label_weight: float = 1.0,
                 uv: torch.Tensor | None = None, filled: torch.Tensor | None = None,
                 contact: dict | None = None, size_override: torch.Tensor | None = None
                 ) -> Batch3:
    """Tokens and targets of chosen view-flights; cut = frames seen (the whole flight if None).

    With heights (`contact` given, or stored in `flights`): anchors come from the user's contact
    heights instead of the stored pitch_xy, and the tokens carry those heights and `filled`.
    `uv`, `size_override` and `filled` override the stored track (training corruption).
    """
    device = flights.device
    n_full = flights.n[index]
    cut = n_full.clone() if cut is None else torch.minimum(cut, n_full)
    end_known = cut == n_full
    size = int(n_full.max())
    if contact is None:
        contact = contact_of(flights, index)
    is_v5 = contact is not None
    uv = flights.uv_det[index, :size] if uv is None else uv[:, :size]
    if filled is None:
        filled = flights.filled[index, :size].bool() if hasattr(flights, "filled") else \
            torch.zeros(uv.shape[:2], dtype=torch.bool, device=device)
    filled = filled[:, :size]
    rot, trans, intr = (x[index, :size] for x in (flights.rot, flights.trans, flights.intr))
    xyz = flights.xyz[index, :size]
    frame = torch.arange(size, device=device)[None]
    in_flight = frame < n_full[:, None]
    seen = frame < cut[:, None]
    present = torch.isfinite(uv[..., 0]) & seen
    has_any = present.any(dim=1)
    # The transformer needs one token; a flight without any detection gets a blank one, which the
    # solver never sees (it reads `present`), and such flights are not answered.
    padding = ~(present | (~has_any[:, None] & (frame == 0)))
    centre, ray = rays_from_pixels(torch.nan_to_num(uv), rot, trans, intr)
    hits = ground_hits(centre, ray)

    rows = torch.arange(len(index), device=device)
    first = present.float().argmax(dim=1)
    last = size - 1 - present.flip(1).float().argmax(dim=1)
    if is_v5:
        contact = {**contact, "end_given": contact["end_given"] & end_known}
        anchors = contact_anchors(uv, rot, trans, intr, present, n_full, contact)
        start_xy, end_xy = anchors["start_xy"], anchors["end_xy"]
    else:
        pitch = flights.pitch_xy[index, :size]
        start_xy = pitch[rows, 0]
        end_xy = pitch[rows, n_full - 1]
    has_start = torch.isfinite(start_xy).all(dim=1)
    has_end = torch.isfinite(end_xy).all(dim=1) & end_known
    start_world = torch.where(has_start[:, None], torch.nan_to_num(start_xy), hits[rows, first, :2])
    end_world = torch.where(has_end[:, None], torch.nan_to_num(end_xy), hits[rows, last, :2])
    origin = torch.cat([start_world, torch.zeros_like(start_world[:, :1])], dim=1)
    heading = end_world - start_world
    yaw = torch.atan2(heading[:, 1], heading[:, 0])
    cos, sin = torch.cos(yaw), torch.sin(yaw)
    zero, one = torch.zeros_like(cos), torch.ones_like(cos)
    to_flight = torch.stack([cos, sin, zero, -sin, cos, zero, zero, zero, one], 1).view(-1, 3, 3)

    def place(points: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bij,btj->bti", to_flight, points - origin[:, None])

    centre = place(centre)
    ray = torch.einsum("bij,btj->bti", to_flight, ray)
    target = place(torch.nan_to_num(xyz))
    hits = place(hits)
    end_point = torch.cat([end_world, torch.zeros_like(end_world[:, :1])], dim=1)
    end_point = torch.einsum("bij,bj->bi", to_flight, end_point - origin)
    oracle = flights.oracle_params[index].clone()
    oracle_start = torch.einsum("bij,bj->bi", to_flight, oracle[:, :3] - origin)
    oracle_velocity = torch.einsum("bij,bj->bi", to_flight, oracle[:, 3:6])
    oracle = torch.cat([oracle_start, oracle_velocity, oracle[:, 6:9]], dim=1)
    labels = oracle[:, 6:9].clone()
    has_gt = torch.isfinite(xyz[..., 2]) & in_flight
    if mirrored is None:
        mirrored = torch.zeros(len(index), dtype=torch.bool, device=device)
    flip = torch.where(mirrored, -1.0, 1.0)
    for tensor in (centre, ray, target, hits):
        tensor[..., 1] *= flip[:, None]
    end_point[:, 1] *= flip
    for column in (1, 4, 8):
        oracle[:, column] *= flip
    labels[:, 2] *= flip

    fps = Config.FPS
    time = (frame.float() / fps).expand(len(index), -1)
    anchor_ray = ray[rows, first]
    anchor_centre = centre[rows, first]
    anchor_time = first.float() / fps
    frame_start = torch.where(has_start, torch.zeros_like(first), first)
    frame_end = torch.where(has_end, n_full - 1, last)
    elapsed = ((frame_end - frame_start).float() / fps).clamp(min=2.0 / fps)
    closed = torch.stack([end_point[:, 0] / elapsed, end_point[:, 1] / elapsed,
                          Config.G_MPS2 * elapsed / 2.0], dim=1)
    seen_time = ((cut - 1).clamp(min=1).float() / fps)
    open_end = torch.stack([zero, zero, Config.G_MPS2 * seen_time], dim=1)
    v_init = torch.where(end_known[:, None], closed, open_end)

    focal = intr[..., 0]
    size_px = flights.size_det[index, :size] if size_override is None else \
        size_override[:, :size]
    has_size = torch.isfinite(size_px) & (size_px > 0) & present
    depth = focal * Config.BALL_DIAMETER_M / torch.nan_to_num(size_px, nan=1.0).clamp(min=1.0)
    size_feature = torch.where(has_size, torch.log(depth / 50.0), torch.zeros_like(depth))
    view = F.one_hot(flights.view[index].long().clamp(max=N_VIEWS - 1), N_VIEWS).float()
    delta = (ray - anchor_ray[:, None]) * 10.0
    freq = 2.0 ** torch.arange(N_FOURIER, device=device).float() * math.pi
    wave = time[..., None] * freq
    is_last = frame == last[:, None]
    end_high = flights.end_high[index] & end_known

    def per_flight(values: torch.Tensor) -> torch.Tensor:
        return values[:, None, None].expand(-1, size, -1).float()

    if is_v5:
        contact_tokens = [per_flight(contact["start_given"]), per_flight(contact["h_start"] / 2.0),
                          per_flight(contact["end_given"]), per_flight(contact["h_end"] / 2.0)]
        frame_tokens = [filled[..., None].float()]
    else:
        contact_tokens = [per_flight(flights.start_high[index]), per_flight(end_high)]
        frame_tokens = []
    tokens = torch.cat([
        centre / 100.0, ray, delta, hits[..., :2] / 50.0, time[..., None] / 5.0,
        (frame == first[:, None])[..., None].float(), is_last[..., None].float(),
        (is_last & end_known[:, None])[..., None].float(),
        *contact_tokens,
        (end_point[:, :2] * end_known[:, None] / 50.0)[:, None].expand(-1, size, -1),
        per_flight(has_start), per_flight(has_end), per_flight(end_known),
        torch.log(focal / 2000.0)[..., None], size_feature[..., None], has_size[..., None].float(),
        view[:, None].expand(-1, size, -1), torch.sin(wave), torch.cos(wave), *frame_tokens],
        dim=-1)
    tokens = torch.nan_to_num(tokens).clamp(-20.0, 20.0)
    weight = torch.full((len(index),), label_weight, device=device)
    extra = {}
    if is_v5:
        slope = anchor_ray[:, :2].norm(dim=1) / anchor_ray[:, 2].abs().clamp(min=1e-3)
        extra = {"h_start": contact["h_start"], "h_end": contact["h_end"],
                 "start_given": contact["start_given"] & has_start,
                 "end_given": contact["end_given"] & has_end, "start_slope": slope}
    return Batch3(tokens=tokens, padding=padding, centre=centre, ray=ray, focal_px=focal,
                  present=present, in_flight=in_flight, seen=seen, xyz=target, has_gt=has_gt,
                  labels=labels, label_weight=weight, oracle=oracle, anchor_centre=anchor_centre,
                  anchor_ray=anchor_ray, anchor_time=anchor_time, hit_end=end_point, v_init=v_init,
                  end_high=end_high, end_known=end_known, index=index, n_frames=size,
                  n_full=n_full, cut=cut, origin=origin, to_flight=to_flight,
                  pitch_start=torch.zeros_like(end_point[:, :2]), has_pitch_start=has_start,
                  has_pitch_end=has_end, mirrored=mirrored, has_obs=has_any, **extra)


def to_world3(path: torch.Tensor, batch: Batch3) -> torch.Tensor:
    """Undo mirroring and the flight frame."""
    path = path.clone()
    path[..., 1] *= torch.where(batch.mirrored, -1.0, 1.0)[:, None]
    return torch.einsum("bij,btj->bti", batch.to_flight.transpose(1, 2), path) + \
        batch.origin[:, None]


class ArcPrior(nn.Module):
    """Transformer over ray tokens; a query token predicts K hypotheses, spreads and logits."""

    def __init__(self, n_hypotheses: int = 4, d_model: int = 128, n_layers: int = 6,
                 n_heads: int = 4, dropout: float = 0.1, has_size: bool = True,
                 n_features: int = N_FEATURES) -> None:
        super().__init__()
        self.k = n_hypotheses
        self.has_size = has_size
        self.embed = nn.Sequential(nn.Linear(n_features, d_model), nn.GELU(),
                                   nn.Linear(d_model, d_model))
        self.query = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(),
                                  nn.Linear(d_model, n_hypotheses * (2 * N_PARAMS + 1)))
        nn.init.normal_(self.head[-1].weight, std=1e-3)
        nn.init.zeros_(self.head[-1].bias)
        nn.init.normal_(self.query, std=0.02)
        # Spread the hypotheses at the start: different launch heights and vertical speeds.
        with torch.no_grad():
            bias = self.head[-1].bias.view(n_hypotheses, 2 * N_PARAMS + 1)
            for k in range(n_hypotheses):
                bias[k, 5] = 0.15 * (k - (n_hypotheses - 1) / 2)

    def forward(self, tokens: torch.Tensor, padding: torch.Tensor) -> torch.Tensor:
        """Raw outputs (batch, K, 2 * N_PARAMS + 1)."""
        if not self.has_size:
            tokens = tokens.clone()
            tokens[..., SIZE_FEATURES] = 0.0
        x = self.embed(tokens)
        query = self.query.expand(len(x), -1, -1)
        x = torch.cat([query, x], dim=1)
        pad = torch.cat([torch.zeros(len(x), 1, dtype=torch.bool, device=x.device), padding], dim=1)
        out = self.head(self.norm(self.encoder(x, src_key_padding_mask=pad)[:, 0]))
        return out.view(len(x), self.k, 2 * N_PARAMS + 1)


def decode3(raw: torch.Tensor, batch: Batch3) -> dict:
    """Bounded parameters, spreads, logits and full-flight paths of every hypothesis."""
    count, k = raw.shape[:2]
    flat = raw.reshape(count * k, -1)

    def rep(x: torch.Tensor) -> torch.Tensor:
        return x.repeat_interleave(k, dim=0)

    z0 = Config.Z0_MAX_M * torch.sigmoid(flat[:, 0] - 1.5)
    offset = torch.tanh(flat[:, 1:3])
    velocity = limit_speed(rep(batch.v_init) + 25.0 * flat[:, 3:6])
    down = rep(batch.anchor_ray)[:, 2]
    safe = torch.where(down.abs() < 1e-3, torch.full_like(down, -1e-3), down)
    along = ((z0 - rep(batch.anchor_centre)[:, 2]) / safe).clamp(1.0, Config.MAX_RAY_M)
    on_ray = rep(batch.anchor_centre) + rep(batch.anchor_ray) * along[:, None]
    on_ray = on_ray - velocity * rep(batch.anchor_time)[:, None]
    at_pitch = torch.cat([rep(batch.pitch_start), z0[:, None]], dim=1)
    start = torch.where(rep(batch.has_pitch_start)[:, None], at_pitch, on_ray)
    start = start + torch.cat([offset, torch.zeros_like(offset[:, :1])], dim=1)
    prior = math.log(0.3 / 0.7)
    k3 = Config.K3_MAX * torch.sigmoid(flat[:, 6] + prior)
    kl = Config.KL_KS_MAX * torch.tanh(flat[:, 7])
    ks = Config.KL_KS_MAX * torch.tanh(flat[:, 8])
    path = integrate(start, velocity, k3, kl, ks, batch.n_frames)
    log_std = flat[:, N_PARAMS:2 * N_PARAMS].clamp(*LOG_STD_RANGE)
    std = torch.exp(log_std) * torch.tensor(PARAM_SCALE, device=raw.device)
    theta = torch.cat([start, velocity, k3[:, None], kl[:, None], ks[:, None]], dim=1)
    return {"theta": theta.view(count, k, N_PARAMS), "std": std.view(count, k, N_PARAMS),
            "logit": raw[..., -1], "path": path.view(count, k, batch.n_frames, 3)}


def path_error(path: torch.Tensor, batch: Batch3) -> torch.Tensor:
    """Mean 3D distance (m) to the GT over the full flight, per flight and hypothesis."""
    distance = torch.sqrt(((path - batch.xyz[:, None]) ** 2).sum(dim=-1) + 1e-6)
    has_gt = batch.has_gt[:, None].float()
    return (distance * has_gt).sum(dim=-1) / has_gt.sum(dim=-1).clamp(min=1.0)


def compute_losses3(out: dict, batch: Batch3, weights: dict) -> tuple[torch.Tensor, dict]:
    """Winner-takes-most path loss, the physics terms on the best hypothesis, spread NLL, select."""
    errors = path_error(out["path"], batch)
    best = errors.detach().argmin(dim=1)
    rows = torch.arange(len(best), device=best.device)
    path_loss = (0.95 * errors.min(dim=1).values + 0.05 * errors.mean(dim=1)).mean()
    path = out["path"][rows, best]

    present = batch.present.float()
    pixels = F.smooth_l1_loss(reprojection_px(path, batch) / 5.0, torch.zeros_like(present),
                              reduction="none")
    reproj_loss = (pixels * present).sum() / present.sum().clamp(min=1.0)
    end = path[rows, batch.n_full - 1]
    closed = (batch.end_known & batch.has_pitch_end).float()
    end_gap = (end[:, :2] - batch.hit_end[:, :2]).norm(dim=1)
    end_loss = (end_gap * closed).sum() / closed.sum().clamp(min=1.0)
    below = F.relu(-path[..., 2]) * batch.in_flight.float()
    height_loss = below.sum() / batch.in_flight.float().sum().clamp(min=1.0)
    height_loss = height_loss + F.relu(end[:, 2] - 5.0).mean()

    theta = out["theta"][rows, best]
    std = out["std"][rows, best]
    scale = torch.tensor([0.02, 0.01, 0.01], device=path.device)
    params_loss = (((theta[:, 6:9] - batch.labels) / scale).abs().mean(dim=1) *
                   batch.label_weight).mean()
    z = (theta.detach() - batch.oracle) / std
    nll = (0.5 * z ** 2 + torch.log(std / torch.tensor(PARAM_SCALE, device=std.device)))
    nll_loss = nll.clamp(max=50.0).mean()
    select_loss = F.cross_entropy(out["logit"], best)
    parts = {"path": path_loss, "reproj": reproj_loss, "end": end_loss, "height": height_loss,
             "params": params_loss, "nll": nll_loss, "select": select_loss}
    total = sum(weights.get(name, 0.0) * parts[name] for name in LOSS_NAMES)
    return total, {name: float(value.detach()) for name, value in parts.items()}


def sample_cuts(n: torch.Tensor, generator: torch.Generator, full_share: float = 0.5,
                low: float = 0.3) -> torch.Tensor:
    """Training cuts: the whole flight with probability full_share, else a share in [low, 1)."""
    u = torch.rand(len(n), generator=generator, device=n.device)
    share = low + (1.0 - low) * torch.rand(len(n), generator=generator, device=n.device)
    frames = torch.ceil(share * n.float()).long().clamp(min=3)
    frames = torch.minimum(frames, n - 1)
    return torch.where(u < full_share, n, frames)


def cut_at(n: torch.Tensor, share: float) -> torch.Tensor:
    """Frames seen at a test share; share 1 is the whole flight with its end contact."""
    if share >= 1.0:
        return n.clone()
    return torch.minimum(torch.ceil(share * n.double()).long(), n - 1)


def repeat_batch(batch: Batch3, k: int) -> Batch3:
    """The same flights k times in a row (for one solver run per hypothesis)."""
    fields = {}
    for name, value in batch.__dict__.items():
        if torch.is_tensor(value) and value.dim() > 0 and len(value) == len(batch.index):
            fields[name] = value.repeat_interleave(k, dim=0)
        else:
            fields[name] = value
    return Batch3(**fields)
