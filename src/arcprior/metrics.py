"""Metrics on per-flight tables (predict.py, run_grad.py, run_snv3d.py).

E-full: capped full-flight RMSE per view-flight, averaged over a physical flight's views, median
over physical flights. E-part: the same averaged over the open-end shares 0.3-0.9, on flights with
at least MIN_PART_FRAMES broadcast frames. Tests: paired bootstrap over physical flights.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import config as Config

PART_SHARES = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
SCOPES = ("broadcast", "static")


def load(paths: list[Path]) -> pd.DataFrame:
    return pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)


def in_scope(table: pd.DataFrame, scope: str) -> pd.DataFrame:
    if scope == "broadcast":
        return table[table["view"] == 0]
    if scope == "static":
        return table[table["view"] != 0]
    if scope.startswith("camera"):
        return table[table["view"] == int(scope[-2:])]
    raise ValueError(scope)


def part_population() -> set:
    """Physical flights long enough for the partial-flight curve (broadcast frames)."""
    flights = pd.read_csv(Config.FLIGHTS_DIR / "flights.csv")
    return set(flights.loc[flights["n"] >= Config.MIN_PART_FRAMES, "group"])


def per_flight(table: pd.DataFrame, method: str, scope: str, share: float = 1.0,
               value: str = "rmse_cap") -> pd.Series:
    """One value per physical flight: the mean over its views (and over seeds, if repeated)."""
    part = in_scope(table[(table["method"] == method) & np.isclose(table["share"], share)], scope)
    return part.groupby("group")[value].mean()


def e_full(table: pd.DataFrame, method: str, scope: str) -> pd.Series:
    return per_flight(table, method, scope, 1.0)


def e_part(table: pd.DataFrame, method: str, scope: str) -> pd.Series:
    part = in_scope(table[(table["method"] == method) &
                          table["share"].round(3).isin(PART_SHARES)], scope)
    part = part[part["group"].isin(part_population())]
    shares = part.groupby(["group", "view", "share"])["rmse_cap"].mean().reset_index()
    view_mean = shares.groupby(["group", "view"])["rmse_cap"].mean()
    counts = shares.groupby(["group", "view"])["share"].nunique()
    view_mean = view_mean[counts == len(PART_SHARES)]
    return view_mean.groupby("group").mean()


def paired(a: pd.Series, b: pd.Series) -> pd.DataFrame:
    both = pd.concat({"a": a, "b": b}, axis=1).dropna()
    return both


def bootstrap(a: pd.Series, b: pd.Series, draws: int = 10000, seed: int = 0) -> dict:
    """Median(a) - median(b) with a percentile interval, resampling physical flights."""
    both = paired(a, b)
    values = both.to_numpy()
    rng = np.random.default_rng(seed)
    picks = rng.integers(len(values), size=(draws, len(values)))
    gaps = np.median(values[picks, 0], axis=1) - np.median(values[picks, 1], axis=1)
    point = float(np.median(values[:, 0]) - np.median(values[:, 1]))
    p_two = 2.0 * min((gaps >= 0).mean(), (gaps <= 0).mean())
    return {"flights": int(len(values)), "a_median": float(np.median(values[:, 0])),
            "b_median": float(np.median(values[:, 1])), "gap": point,
            "low": float(np.quantile(gaps, 0.025)), "high": float(np.quantile(gaps, 0.975)),
            "p": float(min(1.0, p_two)),
            "share_a_better": float((values[:, 0] < values[:, 1]).mean())}


def holm(pvalues: dict) -> dict:
    """Holm-adjusted p-values for one family."""
    order = sorted(pvalues, key=pvalues.get)
    adjusted, running = {}, 0.0
    for rank, key in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * pvalues[key]))
        adjusted[key] = running
    return adjusted


def selection_score(table: pd.DataFrame, method: str, partial: bool = False) -> float:
    """Validation score for per-fold choices: mean of the broadcast and static medians."""
    metric = e_part if partial else e_full
    medians = [metric(table, method, scope).median() for scope in SCOPES]
    medians = [m for m in medians if np.isfinite(m)]
    return float(np.mean(medians)) if medians else float("inf")


def flight_metrics(path_dm: np.ndarray | None, xyz: np.ndarray) -> dict:
    """The same capped and answered metrics as predict.flight_table (numpy)."""
    cap = Config.CAP_M
    valid = np.isfinite(xyz[:, 2])
    n = len(xyz)
    if path_dm is None:
        path = np.full((n, 3), np.nan)
    else:
        path = np.asarray(path_dm, float)[:n] / Config.DM_PER_M
    answered = np.isfinite(path).all(axis=1)
    err = np.linalg.norm(path - xyz, axis=1)
    capped = np.where(answered, np.minimum(err, cap), cap)
    both = valid & answered

    def at(k: int) -> float:
        return float(err[k]) if both[k] else float("nan")

    apex = float("nan")
    if answered.any():
        apex = float(abs(np.nanmax(path[answered, 2]) - np.nanmax(xyz[valid, 2])))
    return {"rmse_cap": float(np.sqrt(np.mean(capped[valid] ** 2))),
            "rmse_answered": float(np.sqrt(np.mean(err[both] ** 2))) if both.any() else np.nan,
            "coverage": float(both.sum() / max(valid.sum(), 1)), "start_err": at(0),
            "end_err": at(n - 1), "apex_err": apex, "answered": bool(answered.any())}
