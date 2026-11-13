"""Contact-free copy of a video input (real, heads): every input taken from contacts is removed.

The flight window stays (full flights know their first and last frame, partial flights only the
first), but no anchor, no contact type and no contact height reaches a method: pitch_xy is NaN,
start_high / end_high are False. The GT heights z_start / z_end stay for the report's breakdowns.
Output: outputs/folds/<out>/ (same arrays, flights.csv, build.json).
Run in the ball3d env: scripts/py scripts/drop_contacts.py real --out real_cv
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as Config


def strip(arrays: dict) -> dict:
    out = dict(arrays)
    out["pitch_xy"] = np.full_like(arrays["pitch_xy"], np.nan)
    out["start_high"] = np.zeros_like(arrays["start_high"])
    out["end_high"] = np.zeros_like(arrays["end_high"])
    out["start_anchor_shift"] = np.full_like(arrays["start_anchor_shift"], -1)
    out["end_anchor_shift"] = np.full_like(arrays["end_anchor_shift"], -1)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    source = Config.FOLDS_DIR / args.scenario
    out = Config.FOLDS_DIR / args.out
    if (out / "build.json").exists():
        raise FileExistsError(f"{out} exists")
    counts = {}
    for path in sorted(source.glob("fold_*/*.npz")):
        with np.load(path) as archive:
            arrays = strip({k: archive[k] for k in archive.files})
        target = out / path.parent.name / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, **arrays)
        counts[f"{path.parent.name}/{path.stem}"] = int(len(arrays["n"]))
    shutil.copy(source / "flights.csv", out / "flights.csv")
    built = json.loads((source / "build.json").read_text())
    report = {"source": args.scenario, "no_contacts": True, "view_flights": counts,
              "source_build": built}
    (out / "build.json").write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
