"""Input sources per setting: one dataset per input setting, each fold from its chosen track.

gt: the dataset's track and cameras. real (published heads) and heads (fine-tuned heads, det,cal):
each fold takes the contact-free track chosen on that fold's validation (Config.TRACK_SOURCE:
Track-Phys in folds 1-3, Track-CV in fold 4). The GT contact heights z_start / z_end are kept for
training labels and the test draws only.
Output: outputs/eval/data/<setting>_src/fold_<k>/ (copies) and flights.csv.
Run in the ball3d env: scripts/py scripts/input_sources.py
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as Config

SOURCES = {
    "gt": {f: "gt" for f in Config.FOLDS},
    "real": {f: f"real_{track}" for f, track in Config.TRACK_SOURCE.items()},
    "heads": {f: f"heads_{track}" for f, track in Config.TRACK_SOURCE.items()},
}


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    for name, folds in SOURCES.items():
        out = Config.EVAL_DATA / f"{name}_src"
        if (out / "build.json").exists():
            print(f"{out} exists")
            continue
        for fold, source in folds.items():
            shutil.copytree(Config.FOLDS_DIR / source / f"fold_{fold}", out / f"fold_{fold}")
        shutil.copy(Config.FOLDS_DIR / folds["1"] / "flights.csv", out / "flights.csv")
        (out / "build.json").write_text(json.dumps({"folds": folds}, indent=1))
        print(name, folds)


if __name__ == "__main__":
    main()
