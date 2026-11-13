"""Choose each model's keypoint and line thresholds by Final Score on the SN23-valid subset.

Output: outputs/calibration/tuning.csv, chosen_thresholds.json. Test sets are not touched.
"""
import argparse
import json
import sys
from itertools import product
from multiprocessing import get_context
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calib_eval as Eval  # noqa: E402
import calib_solve as Solve  # noqa: E402
from calib_common import Config  # noqa: E402

TUNE_DATASET = "SN23-valid"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(Config.CALIB_MODELS))
    parser.add_argument("--workers", type=int, default=14)
    args = parser.parse_args()
    rows = []
    with get_context("fork").Pool(args.workers, Solve.init_worker) as pool:
        for model, (kp_th, line_th) in product(args.models,
                                               product(Config.CALIB_KP_GRID,
                                                       Config.CALIB_LINE_GRID)):
            tag = Solve.pair_tag(kp_th, line_th)
            Solve.solve_dataset(pool, model, TUNE_DATASET, kp_th, line_th, tag)
            path = Eval.eval_dataset(pool, model, tag, TUNE_DATASET)
            frames = pd.read_csv(path)
            cr = frames["ok"].mean()
            jac5 = frames.loc[frames["ok"], "jac5"].mean()
            rows.append({"model": model, "kp_th": kp_th, "line_th": line_th, "tag": tag, "cr": cr,
                         "jac5": jac5, "fs": cr * jac5})
            print(rows[-1], flush=True)
    table = pd.DataFrame(rows)
    tuning = Config.CALIB_ROOT / "tuning.csv"
    if tuning.exists():
        old = pd.read_csv(tuning)
        table = pd.concat([old.loc[~old["model"].isin(args.models)], table], ignore_index=True)
    table.to_csv(tuning, index=False)
    best = table.loc[table.groupby("model")["fs"].idxmax()]
    chosen = {r.model: {"kp_th": r.kp_th, "line_th": r.line_th, "tag": r.tag, "fs": r.fs}
              for r in best.itertuples()}
    (Config.CALIB_ROOT / "chosen_thresholds.json").write_text(json.dumps(chosen, indent=1))
    print(chosen)


if __name__ == "__main__":
    main()
