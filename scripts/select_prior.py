"""Per-fold choice among candidates on validation: each fold on its own validation block.

Each candidate is a (prediction folder, method) pair; the score is arcprior.metrics.selection_score
(mean of the broadcast and static median capped RMSE, E-full or E-part) over that fold's
validation flights, averaged over the seeds given for that candidate.
Output: outputs/eval/select/<name>.json (per fold: every candidate's score and the choice).
Run in the ball3d env:
scripts/py scripts/select_prior.py <name> --candidates gt_full/base_s0:ap:gauss:1:obj ...
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as Config
from arcprior.metrics import load, selection_score



def parse(candidate: str) -> tuple[list[str], str]:
    """'gt_full/base_s0+gt_full/base_s1:ap:gauss:1:obj' -> (folders, method)."""
    folders, method = candidate.split(":", 1)
    return folders.split("+"), method


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--partial", action="store_true", help="score by E-part, not E-full")
    parser.add_argument("--folds", nargs="+", default=sorted(Config.FOLDS))
    parser.add_argument("--root", type=Path, default=Config.EVAL_ROOT,
                        help="folder with pred/ and select/")
    args = parser.parse_args()
    out = args.root / "select"
    out.mkdir(parents=True, exist_ok=True)
    report = {"candidates": args.candidates, "partial": args.partial, "folds": {}}
    for fold in args.folds:
        scores = {}
        for candidate in args.candidates:
            folders, method = parse(candidate)
            values = []
            for folder in folders:
                path = args.root / "pred" / folder / f"fold_{fold}_val_flights.csv.gz"
                if path.exists():
                    values.append(selection_score(load([path]), method, args.partial))
            scores[candidate] = float(np.mean(values)) if values else float("inf")
        chosen = min(scores, key=scores.get)
        report["folds"][fold] = {"scores": scores, "chosen": chosen}
        print(f"fold {fold}: {chosen}  " + "  ".join(f"{k.split(':', 1)[1]}={v:.3f}"
                                                   for k, v in scores.items()), flush=True)
    target = out / f"{args.name}.json"
    if target.exists():
        raise FileExistsError(f"{target} exists")
    target.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
