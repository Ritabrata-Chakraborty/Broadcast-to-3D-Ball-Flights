"""The paper's result tables from the frozen arms: full and partial flights, conditions, mAP.

Arms come from outputs/eval/freeze.json ("sets": <setting>_<kind> -> arm -> fold -> entries
"<test set>/<folder>:<method>", seeds averaged per flight). Full flights: E-full on the full-flight
sets; partial flights: E-part on the partial sets. Each cell is the median over physical flights
with a 95 % bootstrap interval, and each method has a paired gap to the ArcPrior with the lowest
median in its column (Holm over each table's broadcast gaps). Conditions: the median error per
level of each assigned condition (track, start height, end height) and its unpaired difference to
the reference level (clean / correct), with bootstrap intervals (descriptive). Also the uncapped
error over the answered frames and the flight mAP of Grad et al. --split train covers the networks
only (clean track, correct heights; in-sample).
Output: outputs/eval/report/<split>/ (CSV tables and REPORT.md).
Run in the ball3d env: scripts/py scripts/report.py --split test
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as Config
import report_common
from arcprior.metrics import SCOPES, bootstrap, e_full, e_part, holm, per_flight

LEVELS = ("correct", "wrong", "none")   # arcprior.contacts.LEVELS (that module needs torch)

DRAWS = 5000
# Two scenarios: GT = the dataset's track and camera; Real = video (YOLO11, tracker, PnLCalib MV).
# Each row: (input setting of its test set, arm name in freeze.json, row label). Real: ArcPrior
# Base and fine-tuned on the published heads' inputs, and fine-tuned on the fine-tuned heads'
# inputs (det,cal); the baselines on the published heads' inputs, LM-fitg also on (det,cal).
COLUMNS = {
    "GT": [("gt", "ArcPrior base", "ArcPrior Base"),
           ("gt", "ArcPrior real_ft", "ArcPrior ft (forgetting check)"),
           ("gt", "LM-fitg", "LM-fitg"), ("gt", "kNN-flights", "kNN-flights"),
           ("gt", "Grad et al.", "Grad et al."),
           ("gt", "SNv3D-size", "SNv3D-size (own boxes)"),
           ("gt", "SNv3D-size (track)", "SNv3D-size (track)")],
    "Real": [("real", "ArcPrior base", "ArcPrior Base"),
             ("real", "ArcPrior real_ft", "ArcPrior ft"),
             ("heads", "ArcPrior heads_ft", "ArcPrior + heads ft (det,cal)"),
             ("real", "LM-fitg", "LM-fitg"), ("heads", "LM-fitg", "LM-fitg, heads (det,cal)"),
             ("real", "kNN-flights", "kNN-flights"), ("real", "Grad et al.", "Grad et al."),
             ("real", "SNv3D-size", "SNv3D-size (own boxes)"),
             ("real", "SNv3D-size (track)", "SNv3D-size (track)")],
}
FORGETTING = "ArcPrior ft (forgetting check)"
CAMERAS = tuple(f"camera{c:02d}" for c in range(1, 6))


def median_interval(values: pd.Series, seed: int = 0) -> tuple[float, float, float]:
    data = values.dropna().to_numpy()
    if not len(data):
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    meds = np.median(data[rng.integers(len(data), size=(DRAWS, len(data)))], axis=1)
    return float(np.median(data)), float(np.quantile(meds, 0.025)), float(np.quantile(meds,
                                                                                         0.975))


def load_set(arms: dict, split: str) -> pd.DataFrame:
    tables = []
    for name, entries in arms.items():
        if split == "train" and not name.startswith("ArcPrior"):
            continue
        seed0 = entries if split != "train" else {f: e[:1] for f, e in entries.items()}
        try:
            tables.append(report_common.load_arm(seed0, split).assign(method=name))
        except FileNotFoundError as error:
            print(f"missing for {name}: {error}", file=sys.stderr)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def load_column(sets: dict, column: str, kind: str, split: str) -> pd.DataFrame:
    """Every row of one scenario column, method = its row label."""
    tables = []
    for setting, arm, label in COLUMNS[column]:
        arms = sets.get(f"{setting}_{kind}", {})
        if arm in arms:
            table = load_set({arm: arms[arm]}, split)
            if not table.empty:
                tables.append(table.assign(method=label))
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def metric_of(kind: str):
    return e_full if kind == "full" else e_part


def table_ab(sets: dict, kind: str, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cells (method x setting x scope) and gaps to the best ArcPrior of each column."""
    cells, gaps = [], []
    metric = metric_of(kind)
    for setting in COLUMNS:
        table = load_column(sets, setting, kind, split)
        if table.empty:
            continue
        for camera in CAMERAS:      # per static camera: cells only
            for method in table["method"].unique():
                series = metric(table, method, camera)
                med, low, high = median_interval(series)
                cells.append({"setting": setting, "scope": camera, "method": method,
                              "median": med, "low": low, "high": high, "flights": len(series)})
        for scope in SCOPES:
            values = {m: metric(table, m, scope) for m in table["method"].unique()}
            for method, series in values.items():
                med, low, high = median_interval(series)
                cells.append({"setting": setting, "scope": scope, "method": method,
                              "median": med, "low": low, "high": high, "flights": len(series)})
            ours = {m: s.median() for m, s in values.items()
                    if m.startswith("ArcPrior") and m != FORGETTING}
            if not ours or split == "train":
                continue
            best = min(ours, key=ours.get)
            for method, series in values.items():
                if method == best:
                    continue
                test = bootstrap(values[best], series, draws=DRAWS)
                gaps.append({"setting": setting, "scope": scope, "best": best, "method": method,
                             **test})
    gaps = pd.DataFrame(gaps)
    if not gaps.empty:
        family = gaps[gaps["scope"] == "broadcast"]
        adjusted = holm({i: p for i, p in zip(family.index, family["p"])})
        gaps["p_holm"] = [adjusted.get(i, float("nan")) for i in gaps.index]
    return pd.DataFrame(cells), gaps


def conditions(dataset: str, split: str) -> pd.DataFrame:
    """Assigned condition per physical flight (the same in every view of it)."""
    rows = []
    for fold in Config.FOLDS:
        with np.load(Config.EVAL_DATA / dataset / f"fold_{fold}" / f"{split}.npz") as a:
            rows.append(pd.DataFrame({"group": a["group"], "track_masked": a["track_masked"],
                                      "start_level": a["start_level"],
                                      "end_level": a["end_level"]}))
    table = pd.concat(rows).drop_duplicates("group").set_index("group")
    table["track"] = np.where(table["track_masked"], "masked", "clean")
    for end in ("start", "end"):
        table[f"{end}_level"] = [LEVELS[k] for k in table[f"{end}_level"]]
    return table


def table_c(sets: dict, kind: str, split: str) -> pd.DataFrame:
    rows = []
    factors = (("track", "clean"), ("start_level", "correct")) + \
        ((("end_level", "correct"),) if kind == "full" else ())
    for setting in COLUMNS:
        table = load_column(sets, setting, kind, split)
        cond = conditions(f"gt_{kind}", split)   # one assignment for every input setting
        for method in table["method"].unique():
            for scope in SCOPES:
                values = metric_of(kind)(table, method, scope)
                joined = cond.reindex(values.index)
                for factor, reference in factors:
                    base = values[joined[factor] == reference]
                    for level in joined[factor].dropna().unique():
                        part = values[joined[factor] == level]
                        med, low, high = median_interval(part)
                        row = {"setting": setting, "method": method, "scope": scope,
                               "factor": factor, "level": level, "flights": len(part),
                               "median": med, "low": low, "high": high}
                        if level != reference:
                            rng = np.random.default_rng(1)
                            a, b = part.to_numpy(), base.to_numpy()
                            diffs = [np.median(a[rng.integers(len(a), size=len(a))]) -
                                     np.median(b[rng.integers(len(b), size=len(b))])
                                     for _ in range(2000)]
                            row.update(diff=float(np.median(a) - np.median(b)),
                                       diff_low=float(np.quantile(diffs, 0.025)),
                                       diff_high=float(np.quantile(diffs, 0.975)))
                        rows.append(row)
    return pd.DataFrame(rows)


def uncapped(sets: dict, split: str) -> pd.DataFrame:
    """Full flights without the 10 m cap: RMSE over the frames a method answers, per flight
    (views averaged), its median over the flights it answers, and how much it answers."""
    rows = []
    for column in COLUMNS:
        table = load_column(sets, column, "full", split)
        for method in table["method"].unique():
            for scope in SCOPES:
                answered = per_flight(table, method, scope, value="rmse_answered")
                capped = e_full(table, method, scope)
                part = table[(table["method"] == method) & np.isclose(table["share"], 1.0)]
                part = part[(part["view"] == 0) if scope == "broadcast" else (part["view"] != 0)]
                med, low, high = median_interval(answered)
                rows.append({"setting": column, "scope": scope, "method": method,
                             "uncapped_median": med, "low": low, "high": high,
                             "uncapped_p90": float(answered.quantile(0.9)),
                             "flights_answered": int(answered.notna().sum()),
                             "flights": len(capped),
                             "frame_coverage": float(part["coverage"].mean()),
                             "capped_median": float(capped.median())})
    return pd.DataFrame(rows)


def table_map(sets: dict, split: str) -> pd.DataFrame:
    """The benchmark's flight mAP (its evaluator; frames ranked by the fit score, first seed)."""
    rows = []
    for column, members in COLUMNS.items():
        for setting, arm, label in members:
            entries = sets.get(f"{setting}_full", {}).get(arm)
            if entries is None:
                continue
            values = report_common.flight_map(entries, split, "fit_score")
            if values:
                rows.append({"setting": column, "method": label, **values})
    return pd.DataFrame(rows)


def fmt(frame: pd.DataFrame) -> str:
    return frame.round(3).to_markdown(index=False) if not frame.empty else "(none)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True, choices=("train", "val", "test"))
    args = parser.parse_args()
    freeze = json.loads((Config.EVAL_ROOT / "freeze.json").read_text())
    out = Config.EVAL_ROOT / "report" / args.split
    out.mkdir(parents=True, exist_ok=True)
    lines = [f"# Report, {args.split} split", "",
             f"Freeze written {freeze['written']}. Capped RMSE (m), medians over physical flights "
             "with 95 % bootstrap intervals; gaps are the best ArcPrior of the column minus the "
             "method (negative favours ArcPrior), Holm over each table's broadcast gaps.", ""]
    for kind, label in (("full", "Table A, full flights (E-full)"),
                        ("part", "Table B, partial flights (E-part)")):
        cells, gaps = table_ab(freeze["sets"], kind, args.split)
        cells.to_csv(out / f"table_{kind}.csv", index=False)
        gaps.to_csv(out / f"gaps_{kind}.csv", index=False)
        if not cells.empty:
            cells["cell"] = [f"{m:.3f} [{lo:.2f}, {hi:.2f}]" for m, lo, hi in
                             zip(cells["median"], cells["low"], cells["high"])]
            wide = cells.pivot_table(index="method", columns=["setting", "scope"], values="cell",
                                     aggfunc="first")
            lines += [f"## {label}", "", wide.to_markdown(), ""]
        if not gaps.empty:
            lines += ["### Gaps to the best ArcPrior", "",
                      fmt(gaps[["setting", "scope", "best", "method", "a_median", "b_median",
                                "gap", "low", "high", "p", "p_holm", "share_a_better"]]), ""]
        if args.split != "train":
            cond = table_c(freeze["sets"], kind, args.split)
            cond.to_csv(out / f"conditions_{kind}.csv", index=False)
            lines += [f"## Table C, conditions ({kind} flights)", "",
                      fmt(cond[cond["scope"] == "broadcast"]), ""]
    if args.split != "train":
        maps = table_map(freeze["sets"], args.split)
        maps.to_csv(out / "map_full.csv", index=False)
        lines += ["## Flight mAP (full flights, first seed)", "", fmt(maps), ""]
        free = uncapped(freeze["sets"], args.split)
        free.to_csv(out / "uncapped_full.csv", index=False)
        lines += ["## Full flights without the 10 m cap (frames each method answers)", "",
                  fmt(free), ""]
    (out / "REPORT.md").write_text("\n".join(lines))
    print("\n".join(lines[:60]))


if __name__ == "__main__":
    main()
