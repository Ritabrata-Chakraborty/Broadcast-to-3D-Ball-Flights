"""Differentiable flight model: batched RK4 of the arc equations of Grad et al., with lift."""
import torch

import config as Config


def integrate(start: torch.Tensor, velocity: torch.Tensor, k3: torch.Tensor, kl: torch.Tensor,
              ks: torch.Tensor, n_frames: int, g: float = Config.G_MPS2) -> torch.Tensor:
    """Batched RK4 of the arc equations; returns (batch, n_frames, 3) in metres."""
    h = 1.0 / (Config.FPS * Config.SUBSTEPS)
    eps = 1e-6

    def acc(v: torch.Tensor) -> torch.Tensor:
        vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]
        speed = torch.sqrt(vx * vx + vy * vy + vz * vz + eps)
        ground = torch.sqrt(vx * vx + vy * vy + eps)
        ax = -k3 * speed * vx + kl * speed * vx * vz / ground + ks * speed * speed * vy / ground
        ay = -k3 * speed * vy + kl * speed * vy * vz / ground - ks * speed * speed * vx / ground
        az = -g - k3 * speed * vz - kl * speed * ground
        return torch.stack([ax, ay, az], dim=1)

    position, speed_now = start, velocity
    path = [position]
    for _ in range(n_frames - 1):
        for _ in range(Config.SUBSTEPS):
            a1 = acc(speed_now)
            v2 = speed_now + 0.5 * h * a1
            a2 = acc(v2)
            v3 = speed_now + 0.5 * h * a2
            a3 = acc(v3)
            v4 = speed_now + h * a3
            a4 = acc(v4)
            position = position + h * (speed_now + 2 * v2 + 2 * v3 + v4) / 6.0
            speed_now = speed_now + h * (a1 + 2 * a2 + 2 * a3 + a4) / 6.0
        path.append(position)
    return torch.stack(path, dim=1)


def limit_speed(velocity: torch.Tensor) -> torch.Tensor:
    """Scale down any velocity above the physical limit, which also keeps RK4 stable."""
    speed = velocity.norm(dim=1, keepdim=True).clamp(min=1e-6)
    return velocity * (Config.MAX_PRED_SPEED_MPS / speed).clamp(max=1.0)


def reprojection_px(path: torch.Tensor, batch) -> torch.Tensor:
    """Angular distance from each point to its observed ray, as pixels at the camera's focal."""
    offset = path - batch.centre
    across = torch.cross(offset, batch.ray, dim=-1).norm(dim=-1)
    along = (offset * batch.ray).sum(dim=-1).abs().clamp(min=1.0)
    return batch.focal_px * across / along
