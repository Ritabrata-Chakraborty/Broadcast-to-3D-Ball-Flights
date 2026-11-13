"""Shared result loading for report.py and table_heights.py, and the flight mAP of Grad et al.

A prediction entry is "<test set>/<folder>:<method>"; several entries in a fold are seeds, whose
per-flight errors are averaged. The flight mAP uses Grad et al.'s own evaluator.
"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as Config
from ball_estimation.evaluate import _compute_new_metrics

logging.disable(logging.INFO)
PRED = Config.EVAL_ROOT / "pred"
KEYS = ["view", "half", "group", "share"]
NUMERIC = ["rmse_cap", "rmse_answered", "coverage", "start_err", "end_err", "apex_err", "fit_score",
           "omega", "rmse_seen_cap", "rmse_unseen_cap"]
VIEW_NAMES = ("broadcast", *[f"camera{c:02d}" for c in range(1, 7)])


def load_arm(entries: dict, split: str) -> pd.DataFrame:
    """One arm over all folds; several entries in a fold are seeds, averaged per flight.

    On validation only seed 0 is run, so later seeds are skipped there; on test every entry must
    exist.
    """
    parts = []
    for fold, items in entries.items():
        seeds = []
        for k, item in enumerate(items):
            folder, method = item.split(":", 1)
            path = PRED / folder / f"fold_{fold}_{split}_flights.csv.gz"
            if not path.exists():
                if split == "val" and k > 0:
                    print(f"skipped (not run on validation): {path}", file=sys.stderr)
                    continue
                raise FileNotFoundError(path)
            table = pd.read_csv(path)
            table = table[table["method"] == method]
            if table.empty:
                raise ValueError(f"{method} not in {path}")
            seeds.append(table)
        joined = pd.concat(seeds)
        joined["answered"] = joined["answered"].astype(float)
        columns = [c for c in NUMERIC + ["answered"] if c in joined]
        mean = joined.groupby(KEYS)[columns].mean().reset_index()
        first = joined.drop_duplicates(KEYS).set_index(KEYS)
        for column in ("n", "cut", "start_high", "end_high", "z_start", "z_end", "apex_true"):
            mean[column] = first.loc[pd.MultiIndex.from_frame(mean[KEYS]), column].to_numpy()
        mean["answered"] = mean["answered"] > 0.5
        mean["fold"] = fold
        parts.append(mean)
    return pd.concat(parts, ignore_index=True)


def flight_scope(pred: pd.DataFrame, gt: pd.DataFrame,
                 frames: np.ndarray) -> tuple[dict, pd.DataFrame]:
    """Grad et al.'s metrics on the flight frames only (no pause or arc labels)."""
    table = pred.drop_duplicates("file_name").merge(gt.drop_duplicates("file_name"), how="right",
                                                    on="file_name")
    table = table[table["file_name"].isin(frames)].copy()
    table["xyz_error"] = np.sqrt((table["x_predicted"] - table["x"]) ** 2 +
                                 (table["y_predicted"] - table["y"]) ** 2 +
                                 (table["z_predicted"] - table["z"]) ** 2) / Config.DM_PER_M
    table["z_error"] = (table["z_predicted"] - table["z"]).abs() / Config.DM_PER_M
    metrics = _compute_new_metrics(table, table["x"].notna())
    metrics["z_err"] = float(table["z_error"].mean())
    kept = table[["file_name", "x_predicted", "xyz_error", "mean_error"]].copy()
    kept["has_gt"] = table["x"].notna()
    return metrics, kept


def flight_map(entries: dict, split: str, rank: str) -> dict:
    """Grad et al.'s flight mAP (their evaluator), macro over units, first entry (seed) per fold."""
    frames = []
    for fold, items in entries.items():
        folder, method = items[0].split(":", 1)
        path = PRED / folder / f"fold_{fold}_{split}_frames.csv.gz"
        if not path.exists():
            return {}
        table = pd.read_csv(path)
        frames.append(table[table["method"] == method])
    table = pd.concat(frames)
    if rank not in table or table[rank].isna().all():
        return {}
    out = {"broadcast": [], "static": []}
    for (view, half), unit in table.groupby(["view", "half"]):
        root = Config.DATA_ROOT / Config.LP / VIEW_NAMES[view] / Config.HALVES[half]
        gt = pd.read_csv(root / "track/ball_3d-gt.csv")
        pred = unit.rename(columns={rank: "mean_error"})[
            ["file_name", "x_predicted", "y_predicted", "z_predicted", "mean_error"]]
        metrics, _ = flight_scope(pred, gt, unit["file_name"].to_numpy())
        out["broadcast" if view == 0 else "static"].append(metrics["mAP"])
    return {scope: float(np.mean(values)) for scope, values in out.items() if values}
