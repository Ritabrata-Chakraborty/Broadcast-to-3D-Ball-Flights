"""Flight datasets (one fold and split of view-flights), viewing rays and the pinhole projection."""
from pathlib import Path

import numpy as np
import torch

import config as Config

SIGN = torch.tensor([1.0, 1.0, -1.0])


def project(points: torch.Tensor, rot: torch.Tensor, trans: torch.Tensor, intr: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor]:
    """World metres (height up) to pixels at the common width; also returns camera depth."""
    flipped = points * SIGN.to(points.device)
    x_cam = torch.einsum("...ij,...j->...i", rot, flipped) + trans
    depth = x_cam[..., 2]
    safe = depth.clamp(min=1e-3)
    u = intr[..., 0] * x_cam[..., 0] / safe + intr[..., 2]
    v = intr[..., 1] * x_cam[..., 1] / safe + intr[..., 3]
    return torch.stack([u, v], dim=-1), depth


def rays_from_pixels(uv: torch.Tensor, rot: torch.Tensor, trans: torch.Tensor, intr: torch.Tensor
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    """Camera centre and unit ray in world metres (height up) for each pixel."""
    x = (uv[..., 0] - intr[..., 2]) / intr[..., 0]
    y = (uv[..., 1] - intr[..., 3]) / intr[..., 1]
    direction = torch.stack([x, y, torch.ones_like(x)], dim=-1)
    sign = SIGN.to(uv.device)
    world_dir = torch.einsum("...ji,...j->...i", rot, direction) * sign
    centre = -torch.einsum("...ji,...j->...i", rot, trans) * sign
    return centre, world_dir / world_dir.norm(dim=-1, keepdim=True)


def ground_hits(centre: torch.Tensor, ray: torch.Tensor) -> torch.Tensor:
    """Where each ray meets the pitch plane, capped for rays near or above the horizon."""
    down = (-ray[..., 2]).clamp(min=1e-3)
    distance = (centre[..., 2] / down).clamp(max=Config.MAX_RAY_M)
    return centre + ray * distance[..., None]


class FlightSet:
    """One split on one device. Test flights are only read when asked for by name."""

    def __init__(self, fold: str, split: str, device: str, views: tuple[int, ...] | None = None,
                 root: Path = Config.FLIGHTS_DIR) -> None:
        with np.load(root / f"fold_{fold}" / f"{split}.npz") as archive:
            keep = np.arange(len(archive["n"]))
            if views is not None:
                keep = keep[np.isin(archive["view"], views)]
            self.split = split
            self.device = device
            for key in archive.files:
                setattr(self, key, torch.as_tensor(archive[key][keep], device=device))
        self.n = self.n.long()
        self.size = len(self.n)
        self.rot = self.rot.float()
