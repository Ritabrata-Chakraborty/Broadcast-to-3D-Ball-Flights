"""NumPy flight model (the arc equations of Grad et al., metres) and the oracle 3D fit."""
import math

import numpy as np
from scipy.optimize import least_squares


def simulate(params: np.ndarray, n_frames: int, fps: float, g: float, substeps: int) -> np.ndarray:
    """RK4 path of shape (n_frames, 3) for params x, y, z, vx, vy, vz, k3[, kl, ks]."""
    x, y, z, vx, vy, vz, k3 = params[:7]
    kl = params[7] if len(params) > 7 else 0.0
    ks = params[8] if len(params) > 8 else 0.0
    h = 1.0 / (fps * substeps)
    out = np.empty((n_frames, 3))

    def acc(vx: float, vy: float, vz: float) -> tuple[float, float, float]:
        v = math.sqrt(vx * vx + vy * vy + vz * vz)
        vp = math.sqrt(vx * vx + vy * vy) + 1e-6
        ax = -k3 * v * vx + kl * v * vx * vz / vp + ks * v * v * vy / vp
        ay = -k3 * v * vy + kl * v * vy * vz / vp - ks * v * v * vx / vp
        az = -g - k3 * v * vz - kl * v * vp
        return ax, ay, az

    for i in range(n_frames):
        out[i] = x, y, z
        for _ in range(substeps):
            a1 = acc(vx, vy, vz)
            v2 = vx + .5 * h * a1[0], vy + .5 * h * a1[1], vz + .5 * h * a1[2]
            a2 = acc(*v2)
            v3 = vx + .5 * h * a2[0], vy + .5 * h * a2[1], vz + .5 * h * a2[2]
            a3 = acc(*v3)
            v4 = vx + h * a3[0], vy + h * a3[1], vz + h * a3[2]
            a4 = acc(*v4)
            x += h * (vx + 2 * v2[0] + 2 * v3[0] + v4[0]) / 6
            y += h * (vy + 2 * v2[1] + 2 * v3[1] + v4[1]) / 6
            z += h * (vz + 2 * v2[2] + 2 * v3[2] + v4[2]) / 6
            vx += h * (a1[0] + 2 * a2[0] + 2 * a3[0] + a4[0]) / 6
            vy += h * (a1[1] + 2 * a2[1] + 2 * a3[1] + a4[1]) / 6
            vz += h * (a1[2] + 2 * a2[2] + 2 * a3[2] + a4[2]) / 6
    return out


def fit_oracle(xyz: np.ndarray, fps: float, g: float, substeps: int, k3_max: float,
               spin_max: float) -> dict:
    """Fit drag, then drag plus Magnus, to a 3D path with NaN gaps; RMSE is 3D Euclidean, in m."""
    keep = ~np.isnan(xyz[:, 2])
    idx = np.where(keep)[0]
    n_frames = len(xyz)
    step = idx[min(3, len(idx) - 1)] - idx[0]
    v0 = (xyz[idx[min(3, len(idx) - 1)]] - xyz[idx[0]]) * fps / max(step, 1)
    start = xyz[idx[0]]
    lo = np.r_[start - 2, -45, -45, -30, 0.0]
    hi = np.r_[start + 2, 45, 45, 30, k3_max]
    guess = np.clip(np.r_[start, v0, 0.015], lo, hi)

    def residual(params: np.ndarray) -> np.ndarray:
        return (simulate(params, n_frames, fps, g, substeps)[keep] - xyz[keep]).ravel()

    drag = least_squares(residual, guess, bounds=(lo, hi), max_nfev=40)
    lo_spin, hi_spin = np.r_[lo, -spin_max, -spin_max], np.r_[hi, spin_max, spin_max]
    spin = least_squares(residual, np.r_[drag.x, 0.0, 0.0], bounds=(lo_spin, hi_spin), max_nfev=40)
    return {"params": spin.x,
            "rmse_spin_m": float(np.sqrt(np.mean(spin.fun ** 2) * 3.0)),
            "rmse_drag_m": float(np.sqrt(np.mean(drag.fun ** 2) * 3.0))}
