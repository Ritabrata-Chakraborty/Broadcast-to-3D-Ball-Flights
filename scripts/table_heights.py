"""Paper Table 1: contact heights on the same clean-track test flights (GT inputs, fixed levels).

ArcPrior Base (3 seeds, the prior frozen per fold for gt_full), LM-fitg and Grad et al. (no heights,
camera01/05 released points turned back) on outputs/eval/data/gt_fix{correct,wrong,none}; median
capped RMSE per physical flight (static views averaged, seeds averaged) and paired bootstrap gaps.
Output: outputs/eval/report/test/table1.csv and table1_gaps.csv.
Run in the ball3d env: scripts/py scripts/table_heights.py
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as Config  # noqa: E402
import report_common  # noqa: E402
from arcprior.metrics import bootstrap, e_full  # noqa: E402

LEVELS = ("correct", "wrong", "none")
FOLDS = ("1", "2", "3", "4")
DRAWS = 5000
OUT = Config.EVAL_ROOT / "report" / "test"


def arms_for(level: str, choices: dict) -> dict:
    name = f"gt_fix{level}"
    arms = {"ArcPrior Base": {f: [f"{name}/base_s{k}:{choices[f]}" for k in range(3)]
                              for f in FOLDS},
            "LM-fitg": {f: [f"{name}/physics:lm_fitg"] for f in FOLDS}}
    if level == "none":
        arms["Grad et al."] = {f: [f"{name}/grad:b1"] for f in FOLDS}
    return arms


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    choices = json.loads((Config.EVAL_ROOT / "freeze.json").read_text())["choices"]
    choices = choices["base_gt_full"]
    cells, per_level, gaps = [], {}, []
    for level in LEVELS:
        tables = {name: report_common.load_arm(entries, "test").assign(method=name)
                  for name, entries in arms_for(level, choices).items()}
        for name, table in tables.items():
            for scope in ("broadcast", "static"):
                values = e_full(table, name, scope)
                per_level[(level, name, scope)] = values
                cells.append({"level": level, "method": name, "scope": scope,
                              "median": float(values.median()), "flights": len(values)})
    for scope in ("broadcast", "static"):
        for name in ("ArcPrior Base", "LM-fitg"):
            for level in ("wrong", "none"):
                result = bootstrap(per_level[(level, name, scope)],
                                   per_level[("correct", name, scope)], DRAWS)
                gaps.append({"scope": scope, "a": f"{name} {level}", "b": f"{name} correct",
                             **result})
        for level in LEVELS:
            result = bootstrap(per_level[(level, "ArcPrior Base", scope)],
                               per_level[(level, "LM-fitg", scope)], DRAWS)
            gaps.append({"scope": scope, "a": f"ArcPrior Base {level}",
                         "b": f"LM-fitg {level}", **result})
        for name in ("ArcPrior Base", "LM-fitg"):
            result = bootstrap(per_level[("none", name, scope)],
                               per_level[("none", "Grad et al.", scope)], DRAWS)
            gaps.append({"scope": scope, "a": f"{name} none", "b": "Grad et al. none", **result})
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cells).to_csv(OUT / "table1.csv", index=False)
    pd.DataFrame(gaps).to_csv(OUT / "table1_gaps.csv", index=False)
    print(pd.DataFrame(cells).pivot_table(index=["level", "method"], columns="scope",
                                          values="median").round(3).to_string())
    print(pd.DataFrame(gaps).round(3).to_string())


if __name__ == "__main__":
    main()
