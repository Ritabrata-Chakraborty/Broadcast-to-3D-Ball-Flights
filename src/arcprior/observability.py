"""Per-flight observability: what the image fixes, from the solver's own Gauss-Newton model.

At a fitted parameter vector the residual Jacobian J gives Sigma = s^2 (J'J)^-1 (scaled units);
the path Jacobian carries it to a 3x3 covariance per frame. Omega = sqrt(mean trace) is the
predicted 3D error, and the top eigenvector of the mean covariance is the weakest direction.
The same Jacobian, whitened by the network's spread, gives the weak subspace for the prior.
"""
import torch

from arcprior.flight import integrate
from arcprior.solver import (FD_STEP, G, K3, KL, KS, N_PRIOR, SCALE, START, VELOCITY, RefineConfig,
                         huber_weights, make_bounds, pack, repeat, residuals)


def varying_columns(cfg: RefineConfig) -> list[int]:
    """The parameters the solver moves, in refine()'s order."""
    columns = [*START, *VELOCITY, K3]
    if cfg.has_spin:
        columns += [KL, KS]
    if cfg.has_free_g:
        columns.append(G)
    return columns


@torch.no_grad()
def residual_jacobian(theta: torch.Tensor, batch, cfg: RefineConfig, prior: torch.Tensor | None,
                      steps: torch.Tensor, columns: list[int]
                      ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Residuals (B, m), their Jacobian (B, m, p) per unit step, and the Huber weights."""
    data = pack(batch)
    count = len(columns) + 1
    wide = repeat(data, count)
    wide_prior = None if prior is None else prior.repeat_interleave(count, dim=0)
    _, distance = residuals(theta, data, cfg, None, prior)
    weights = huber_weights(distance, data["present"], cfg.huber_m)
    stack = theta.repeat_interleave(count, dim=0)
    for position, column in enumerate(columns):
        stack[position + 1::count, column] += steps[:, position]
    values, _ = residuals(stack, wide, cfg, weights.repeat_interleave(count, dim=0), wide_prior)
    values = values.view(len(theta), count, -1)
    jac = (values[:, 1:] - values[:, :1]) / steps[..., None]
    return values[:, 0], jac.transpose(1, 2), weights


@torch.no_grad()
def weak_projection(theta: torch.Tensor, std: torch.Tensor, batch, cfg: RefineConfig,
                    prior_weight: float) -> torch.Tensor:
    """Projection (B, 9, 9) onto whitened directions where the data are weaker than the prior.

    Whitened units divide each parameter by the network's spread, so the prior Hessian there is
    prior_weight^2 times the identity; data eigenvalues below that leave the prior in charge.
    """
    data_cfg = RefineConfig(**{**cfg.__dict__, "prior_weight": 0.0})
    columns = varying_columns(data_cfg)[:N_PRIOR]
    steps = (FD_STEP * 10.0 * std[:, :len(columns)]).to(theta.dtype)
    _, jac, _ = residual_jacobian(theta, batch, data_cfg, None, steps, columns)
    jac = jac * std[:, None, :len(columns)].to(theta.dtype)   # per whitened unit
    hess = jac.transpose(1, 2) @ jac
    values, vectors = torch.linalg.eigh(hess)
    weak = (values < prior_weight ** 2).to(theta.dtype)
    proj = vectors @ torch.diag_embed(weak) @ vectors.transpose(1, 2)
    out = torch.zeros(len(theta), N_PRIOR, N_PRIOR, dtype=theta.dtype, device=theta.device)
    out[:, :len(columns), :len(columns)] = proj
    return out


@torch.no_grad()
def observability(theta: torch.Tensor, batch, cfg: RefineConfig, prior: torch.Tensor | None,
                  horizon: int, mask: torch.Tensor) -> dict:
    """Omega (m) over the masked frames, the weakest direction (flight frame), the covariance."""
    columns = varying_columns(cfg)
    scale = theta.new_tensor(SCALE)[columns]
    steps = (FD_STEP * scale).expand(len(theta), -1).clone()
    values, jac, weights = residual_jacobian(theta, batch, cfg, prior, steps, columns)
    jac = jac * scale            # per scaled unit
    data = pack(batch)
    n_present = data["n_present"]
    n_traj = 3 * data["present"].shape[1]
    traj_cost = (values[:, :n_traj] ** 2).sum(dim=1)
    dof = (2.0 * n_present - len(columns)).clamp(min=1.0)
    s2 = (traj_cost / dof).clamp(min=1e-6)
    hess = jac.transpose(1, 2) @ jac
    eye = torch.eye(len(columns), dtype=theta.dtype, device=theta.device)
    # Parameters held at a bound are not free: the constrained covariance leaves them out
    # (on validation this lifts Omega's rank correlation on full flights from .39 to .72).
    low, high = make_bounds(batch, cfg)
    tol = 1e-6 * scale
    chosen = theta[:, columns]
    free = ~((chosen <= low[:, columns] + tol) | (chosen >= high[:, columns] - tol))
    free = free.to(theta.dtype)
    hess = hess * free[:, None] * free[:, :, None] + torch.diag_embed(1.0 - free)
    cov = s2[:, None, None] * torch.linalg.inv(hess + 1e-9 * eye)
    cov = cov * free[:, None] * free[:, :, None]

    def path_of(params: torch.Tensor) -> torch.Tensor:
        return integrate(params[:, :3], params[:, 3:6], params[:, K3], params[:, KL],
                         params[:, KS], horizon, params[:, G])

    base = path_of(theta)
    grads = []
    for position, column in enumerate(columns):
        moved = theta.clone()
        moved[:, column] += FD_STEP * 10.0 * scale[position]
        grads.append((path_of(moved) - base) / (FD_STEP * 10.0))
    path_jac = torch.stack(grads, dim=-1)                       # (B, T, 3, p)
    frame_cov = path_jac @ cov[:, None] @ path_jac.transpose(-1, -2)
    trace = torch.diagonal(frame_cov, dim1=-2, dim2=-1).sum(dim=-1)
    weight = mask.to(theta.dtype)
    count = weight.sum(dim=1).clamp(min=1.0)
    omega = torch.sqrt(((trace * weight).sum(dim=1) / count).clamp(min=0.0))
    mean_cov = (frame_cov * weight[..., None, None]).sum(dim=1) / count[:, None, None]
    # Flights without usable data give non-finite covariances: no Omega, no direction.
    is_finite = torch.isfinite(mean_cov).all(dim=(1, 2)) & torch.isfinite(omega)
    safe = torch.where(is_finite[:, None, None], mean_cov, torch.zeros_like(mean_cov))
    safe = safe + 1e-12 * torch.eye(3, dtype=safe.dtype, device=safe.device)
    _, vectors = torch.linalg.eigh(safe)
    nan = torch.full_like(omega, float("nan"))
    return {"omega": torch.where(is_finite, omega, nan), "weak_dir": vectors[..., -1],
            "cov": cov, "s2": s2}


@torch.no_grad()
def objective(theta: torch.Tensor, batch, cfg: RefineConfig, prior: torch.Tensor | None
              ) -> torch.Tensor:
    """The solver's final weighted cost per flight (data, end, height and prior terms)."""
    data = pack(batch)
    _, distance = residuals(theta, data, cfg, None, prior)
    weights = huber_weights(distance, data["present"], cfg.huber_m)
    values, _ = residuals(theta, data, cfg, weights, prior)
    return (values ** 2).sum(dim=1)
