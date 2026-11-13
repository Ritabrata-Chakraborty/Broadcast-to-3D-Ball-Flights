"""Per-flight refinement: batched Levenberg-Marquardt on the arc losses of Grad et al., any start.

Reimplemented from the published loss and bounds, not copied. Errors are metres at the ball
(point-to-ray distance), which is what dividing pixels by the ball's size measures.
"""
from dataclasses import dataclass

import torch

import config as Config
from arcprior.flight import integrate

N_THETA = 10
START, VELOCITY, K3, KL, KS, G = (0, 1, 2), (3, 4, 5), 6, 7, 8, 9
SCALE = (1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 0.02, 0.01, 0.01, 1.0)
# Spread of the network prior: start height (m), then k3, kl, ks (1/m).
PRIOR_SCALE = (0.5, 0.02, 0.01, 0.01)
# ball3d's own bounds and term scales; see conf/trajectory/base_trajectory.yaml.
G_VARIATION_MPS2 = 3.5
START_SHIFT_M = 0.1
Z0_MAX_M = 2.0
BALL3D_K3_MAX = 1.0
# ball3d divides both by ball size, but measures L_end in world units and L_traj in pixels. For
# a typical 11 px box, a metre at the end point weighs this share of a metre on the ray.
END_SCALE = 0.28
HEIGHT_SLOPE = 0.11
HEIGHT_WALL = 22.0
FD_STEP = 1e-5
# A user's contact height is a robust soft prior (spread of a careful
# guess, Huber beyond it); the start may move CONTACT_BOUND_M around it (a wrong number is at most
# 0.8 m off) and along the anchor ray by as much; heights stay in [0, CONTACT_Z_MAX_M].
CONTACT_SIGMA_M = 0.15
CONTACT_BOUND_M = 0.8
CONTACT_Z_MAX_M = 2.7


@dataclass
class RefineConfig:
    """What may vary and how strongly each ball3d term counts."""
    has_spin: bool = True
    has_free_g: bool = False
    k3_max: float = Config.K3_MAX
    spin_max: float = Config.KL_KS_MAX
    prior_weight: float = 0.0
    n_iters: int = 40
    huber_m: float = 1.0
    w_end: float = 0.35
    w_height: float = 0.3
    start_shift_m: float = START_SHIFT_M   # half-width of the start box around a known contact
    # "old": z0, k3, kl, ks with PRIOR_SCALE; "gauss": all nine network parameters with the
    # network's spread; "subspace": the same, projected on the weak directions (pack_prior).
    prior_kind: str = "old"


# LM-fitg: Grad et al.'s fitg model (drag, free gravity) with its drag bound, no lift.
LM_FITG_CFG = RefineConfig(has_spin=False, has_free_g=True, k3_max=BALL3D_K3_MAX)
# ArcPrior's flight model: fixed gravity, drag, lift and sidespin within the network's bounds.
LIFT_CFG = RefineConfig()


def ball3d_guess(batch) -> torch.Tensor:
    """ball3d's initial guess: on the ground at the start contact, symmetric arc to the end."""
    count = len(batch.v_init)
    start = torch.cat([batch.pitch_start, torch.zeros(count, 1, device=batch.v_init.device)], dim=1)
    k3 = torch.full((count, 1), 0.05 * 0.2, device=batch.v_init.device)
    g = torch.full((count, 1), Config.G_MPS2, device=batch.v_init.device)
    spin = torch.zeros(count, 2, device=batch.v_init.device)
    return torch.cat([start, batch.v_init, k3, spin, g], dim=1).double()


def make_bounds(batch, cfg: RefineConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-flight lower and upper bounds; the start is boxed around a known contact point."""
    count = len(batch.v_init)
    low = torch.full((count, N_THETA), -1e9, dtype=torch.float64)
    high = torch.full((count, N_THETA), 1e9, dtype=torch.float64)
    known = batch.has_pitch_start
    pitch = batch.pitch_start.double()
    low[:, :2] = torch.where(known[:, None], pitch - cfg.start_shift_m, low[:, :2])
    high[:, :2] = torch.where(known[:, None], pitch + cfg.start_shift_m, high[:, :2])
    low[:, 2], high[:, 2] = 0.0, Z0_MAX_M
    if getattr(batch, "h_start", None) is not None:
        # The user's number replaces ball3d's 0-2 m start band; a height error moves the anchor
        # along its ray, so the box grows with the ray's slope.
        given = batch.start_given
        h = batch.h_start.double()
        low[:, 2] = torch.where(given, (h - CONTACT_BOUND_M).clamp(min=0.0), low[:, 2])
        high[:, 2] = torch.where(given, (h + CONTACT_BOUND_M).clamp(max=CONTACT_Z_MAX_M),
                                 torch.full_like(h, CONTACT_Z_MAX_M))
        box = cfg.start_shift_m + CONTACT_BOUND_M * batch.start_slope.double().clamp(max=10.0)
        low[:, :2] = torch.where(known[:, None], pitch - box[:, None], low[:, :2])
        high[:, :2] = torch.where(known[:, None], pitch + box[:, None], high[:, :2])
    low[:, K3], high[:, K3] = 0.0, cfg.k3_max
    low[:, [KL, KS]], high[:, [KL, KS]] = -cfg.spin_max, cfg.spin_max
    low[:, G] = Config.G_MPS2 - G_VARIATION_MPS2
    high[:, G] = Config.G_MPS2 + G_VARIATION_MPS2
    return low, high


def project_bounds(theta: torch.Tensor, low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    """Keep parameters inside their box and under the speed limit."""
    out = torch.minimum(torch.maximum(theta, low), high)
    speed = out[:, 3:6].norm(dim=1, keepdim=True).clamp(min=1e-6)
    out[:, 3:6] = out[:, 3:6] * (Config.MAX_PRED_SPEED_MPS / speed).clamp(max=1.0)
    return out


def pack(batch) -> dict:
    """The observed data one flight's residuals need, in double precision.

    A cut flight (cut batches carry `seen`) ends at its last seen frame: the solver never
    learns how long the flight really is.
    """
    present = batch.present
    rows = torch.arange(len(present), device=present.device)
    last_seen = present.shape[1] - 1 - present.flip(1).float().argmax(dim=1)
    n_present = present.sum(dim=1).double().clamp(min=1.0)
    n = getattr(batch, "seen", batch.in_flight).sum(dim=1)
    return {"centre": batch.centre.double(), "ray": batch.ray.double(), "present": present,
            "rows": rows, "end_frame": n - 1, "last_seen": last_seen, "n_present": n_present,
            "has_end": batch.has_pitch_end, "end_xy": batch.hit_end[:, :2].double(),
            "in_flight": batch.in_flight, "n_frames": batch.n_frames,
            "h_start": getattr(batch, "h_start", None), "h_end": getattr(batch, "h_end", None),
            "start_given": getattr(batch, "start_given", None),
            "end_given": getattr(batch, "end_given", None)}


def repeat(data: dict, count: int) -> dict:
    """Stack a flight's data once per parameter, for the finite-difference Jacobian."""
    out = {}
    for key, value in data.items():
        out[key] = value.repeat_interleave(count, dim=0) if torch.is_tensor(value) else value
    out["rows"] = torch.arange(len(out["present"]), device=out["present"].device)
    return out


def residuals(theta: torch.Tensor, data: dict, cfg: RefineConfig, weights: torch.Tensor | None,
              prior: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    """Weighted residual vector per flight, and each frame's distance to its observed ray."""
    path = integrate(theta[:, :3], theta[:, 3:6], theta[:, K3], theta[:, KL], theta[:, KS],
                     data["n_frames"], theta[:, G])
    across = torch.cross(path - data["centre"], data["ray"], dim=-1)
    present = data["present"]
    distance = torch.sqrt((across ** 2).sum(dim=-1) + 1e-12) * present
    frame_weight = present.double() if weights is None else weights
    traj = across * torch.sqrt(frame_weight)[..., None]

    rows, n_present = data["rows"], data["n_present"]
    end_weight = torch.sqrt(cfg.w_end * n_present)
    # L_end: world distance at a known contact, which ball3d also applies to high contacts after
    # moving them to the player's feet. Image distance only when the end point is unknown.
    closed = (path[rows, data["end_frame"], :2] - data["end_xy"]) * END_SCALE
    closed = closed * (end_weight * data["has_end"])[:, None]
    open_end = across[rows, data["last_seen"]] * (end_weight * ~data["has_end"])[:, None]
    # L_z: ball3d's height band is centred at 1.2 m at a contact and 5 m on an open end.
    z_end = path[rows, data["end_frame"], 2]
    band = torch.where(data["has_end"], 1.2, 5.0)
    wall = HEIGHT_WALL * ((z_end - band).abs() - band)
    height = torch.maximum(HEIGHT_SLOPE * z_end, wall) * torch.sqrt(cfg.w_height * n_present)
    parts = [traj.flatten(1), closed, open_end]
    if data.get("h_start") is not None:
        # The user's heights as soft terms; the band stays where none is given.
        scale = torch.sqrt(cfg.w_height * n_present)
        end_term = robust((z_end - data["h_end"].double()) / CONTACT_SIGMA_M) * scale
        height = torch.where(data["end_given"], end_term, height)
        start_term = robust((theta[:, 2] - data["h_start"].double()) / CONTACT_SIGMA_M) * scale
        parts.append((start_term * data["start_given"])[:, None])
    parts.append(height[:, None])
    if prior is not None and cfg.prior_weight > 0:
        parts.append(cfg.prior_weight * prior_residual(theta, prior, cfg.prior_kind))
    return torch.cat(parts, dim=1), distance


N_PRIOR = 9   # start (3), velocity (3), k3, kl, ks


def robust(r: torch.Tensor) -> torch.Tensor:
    """Residual whose square is Huber's loss (x2): r inside one unit, linear growth beyond."""
    return torch.where(r.abs() <= 1.0, r, torch.sign(r) * torch.sqrt(2.0 * r.abs() - 1.0))


def pack_prior(theta: torch.Tensor, std: torch.Tensor | None = None,
               proj: torch.Tensor | None = None) -> torch.Tensor:
    """One tensor per flight (so it repeats with the Jacobian stack): theta, spread, projection."""
    count = len(theta)
    if std is None:
        std = theta.new_ones(count, N_PRIOR)
    if proj is None:
        proj = torch.eye(N_PRIOR, dtype=theta.dtype, device=theta.device).expand(count, -1, -1)
    return torch.cat([theta, std.to(theta.dtype), proj.reshape(count, -1).to(theta.dtype)], dim=1)


def prior_residual(theta: torch.Tensor, prior: torch.Tensor, kind: str) -> torch.Tensor:
    """Prior residual for one of the prior kinds (see RefineConfig.prior_kind)."""
    if kind == "old":
        scale = theta.new_tensor(PRIOR_SCALE)
        chosen = torch.stack([theta[:, 2], theta[:, K3], theta[:, KL], theta[:, KS]], dim=1)
        anchor = torch.stack([prior[:, 2], prior[:, K3], prior[:, KL], prior[:, KS]], dim=1)
        return (chosen - anchor) / scale
    std = prior[:, N_THETA:N_THETA + N_PRIOR]
    white = (theta[:, :N_PRIOR] - prior[:, :N_PRIOR]) / std
    if kind == "subspace":
        proj = prior[:, N_THETA + N_PRIOR:].reshape(-1, N_PRIOR, N_PRIOR)
        white = torch.einsum("bij,bj->bi", proj, white)
    elif kind != "gauss":
        raise ValueError(f"unknown prior kind {kind}")
    return white


def huber_weights(distance: torch.Tensor, present: torch.Tensor, huber_m: float) -> torch.Tensor:
    """IRLS weights: frames far from the path, such as false detections, count less."""
    return (huber_m / distance.clamp(min=1e-9)).clamp(max=1.0) * present


@torch.no_grad()
def refine(theta: torch.Tensor, batch, cfg: RefineConfig, prior: torch.Tensor | None = None
           ) -> tuple[torch.Tensor, torch.Tensor]:
    """Levenberg-Marquardt per flight. Returns parameters and a fit score in metres."""
    varying = [*START, *VELOCITY, K3]
    if cfg.has_spin:
        varying += [KL, KS]
    if cfg.has_free_g:
        varying.append(G)
    theta = theta.clone()
    if not cfg.has_spin:
        theta[:, [KL, KS]] = 0.0
    count = len(varying) + 1
    data = pack(batch)
    wide = repeat(data, count)
    wide_prior = None if prior is None else prior.repeat_interleave(count, dim=0)
    scale = theta.new_tensor(SCALE)
    n_batch = len(theta)
    damping = theta.new_full((n_batch,), 1e-3)
    low, high = make_bounds(batch, cfg)
    theta = project_bounds(theta, low, high)
    eye = torch.eye(len(varying), dtype=theta.dtype, device=theta.device)

    for _ in range(cfg.n_iters):
        _, distance = residuals(theta, data, cfg, None, prior)
        weights = huber_weights(distance, data["present"], cfg.huber_m)
        # Forward differences in scaled units, stepping into the box at an upper bound.
        stack = theta.repeat_interleave(count, dim=0)
        steps = torch.zeros(n_batch, len(varying), dtype=theta.dtype, device=theta.device)
        for position, column in enumerate(varying):
            step = FD_STEP * SCALE[column]
            is_at_top = theta[:, column] + step > high[:, column]
            steps[:, position] = torch.where(is_at_top, -step, step)
            stack[position + 1::count, column] += steps[:, position]
        wide_weights = weights.repeat_interleave(count, dim=0)
        values, _ = residuals(stack, wide, cfg, wide_weights, wide_prior)
        values = values.view(n_batch, count, -1)
        now = values[:, 0]
        in_scaled = steps / scale[varying]
        jac = ((values[:, 1:] - now[:, None]) / in_scaled[..., None]).transpose(1, 2)
        cost = (now ** 2).sum(dim=1)
        grad = (jac.transpose(1, 2) * now[:, None]).sum(dim=2)
        hess = jac.transpose(1, 2) @ jac
        # Without this, projecting a step back onto a bound can stop it being a descent step.
        chosen = theta[:, varying]
        margin = 1e-9
        blocked = ((chosen <= low[:, varying] + margin) & (grad > 0)) | \
            ((chosen >= high[:, varying] - margin) & (grad < 0))
        free = (~blocked).double()
        grad = grad * free
        hess = hess * free[:, None] * free[:, :, None]
        diagonal = torch.diagonal(hess, dim1=1, dim2=2)
        system = hess + (damping[:, None] * diagonal + 1e-9 + blocked.double())[..., None] * eye
        move = torch.linalg.solve(system, -grad[..., None])[..., 0]
        trial = theta.clone()
        trial[:, varying] = theta[:, varying] + move * scale[varying]
        trial = project_bounds(trial, low, high)
        values_new, _ = residuals(trial, data, cfg, weights, prior)
        cost_new = (values_new ** 2).sum(dim=1)
        is_better = torch.isfinite(cost_new) & (cost_new < cost)
        theta = torch.where(is_better[:, None], trial, theta)
        damping = torch.where(is_better, damping / 3.0, damping * 4.0).clamp(1e-9, 1e6)

    _, distance = residuals(theta, data, cfg, None, prior)
    return theta, distance.sum(dim=1) / data["n_present"]
