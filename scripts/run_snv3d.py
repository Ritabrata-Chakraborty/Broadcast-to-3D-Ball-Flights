"""SNv3D-size: SoccerNet-v3D's monocular localisation (its Eq. 7) on our test sets, frame by frame.

Depth along the box-centre ray is f * 0.22 m / box diameter, with the test set's cameras.
--boxes opt: its own detector, YOLO11-L with the size-optimised weights yolo-sn-ball-opt.pt
  (outputs/extract/snv3d_opt), the highest-scoring box of each frame above a minimum score chosen
  per fold on the training flights (median error over the answered frames); tag snv3d_opt.
--boxes track: the boxes of the test set's own track (centre = bottom - diameter / 2); tag snv3d.
It needs no contacts and cannot predict an unseen frame; mAP ranks frames by 1 - box score.
Output: outputs/eval/pred/<test set>/<tag>/fold_<k>_<split>_{flights,frames}.csv.gz
Run in the ball3d env: scripts/py scripts/run_snv3d.py gt_full --fold 1 --splits test --boxes opt
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
from arcprior.metrics import flight_metrics

EXTRACT = Config.EXTRACT_DIR / "snv3d_opt"
VIEW_NAMES = ("broadcast", *[f"camera{c:02d}" for c in range(1, 7)])
SCORE_GRID = (0.05, 0.1, 0.25, 0.5)


def best_boxes(view: int, half: int) -> pd.DataFrame:
    """Highest-scoring box per frame: centre and diameter at 1920 wide, and its score."""
    name = f"{VIEW_NAMES[view]}_{Config.HALVES[half]}"
    boxes = pd.read_csv(EXTRACT / f"{name}.det.csv.gz")
    width = json.loads((EXTRACT / f"{name}.json").read_text())["width"]
    boxes = boxes.sort_values("score", ascending=False).drop_duplicates("file_name")
    scale = Config.IMAGE_WIDTH_PX / width
    return pd.DataFrame({"u": (boxes["x0"] + boxes["x1"]) / 2 * scale,
                         "v": (boxes["y0"] + boxes["y1"]) / 2 * scale,
                         "d": ((boxes["x1"] - boxes["x0"]) + (boxes["y1"] - boxes["y0"])) / 2
                         * scale, "score": boxes["score"]}).set_index(boxes["file_name"])


def localise(uvd: np.ndarray, rot: np.ndarray, trans: np.ndarray, intr: np.ndarray) -> np.ndarray:
    """Eq. 7 per frame: world metres (height up) of the ball whose box spans d pixels."""
    p_c = np.stack([(uvd[:, 0] - intr[:, 2]) / intr[:, 0], (uvd[:, 1] - intr[:, 3]) / intr[:, 1],
                    np.ones(len(uvd))], axis=1)
    depth = Config.BALL_DIAMETER_M * intr[:, 0] / uvd[:, 2]
    x_cam = p_c * depth[:, None]
    world = np.einsum("nji,nj->ni", rot, x_cam - trans)
    world[:, 2] *= -1.0
    return world


def track_boxes(arrays: dict, i: int) -> tuple[np.ndarray, np.ndarray]:
    """The scenario track's own boxes: centre and diameter (1920 wide), score 1."""
    n = int(arrays["n"][i])
    d = arrays["size_det"][i, :n].astype(float)
    uvd = np.c_[arrays["uv_det"][i, :n, 0], arrays["uv_det"][i, :n, 1] - d / 2.0, d]
    return uvd, np.where(np.isfinite(uvd).all(axis=1), 1.0, np.nan)


def predict_split(arrays: dict, boxes: dict | None, min_score: float) -> tuple[list, list]:
    rows, frames = [], []
    for i in range(len(arrays["n"])):
        n = int(arrays["n"][i])
        a = int(arrays["a"][i])
        if boxes is None:
            uvd, score = track_boxes(arrays, i)
        else:
            table = boxes[(int(arrays["view"][i]), int(arrays["half"][i]))]
            picked = table.reindex(a + np.arange(n))
            uvd = picked[["u", "v", "d"]].to_numpy(float)
            score = picked["score"].to_numpy(float)
        keep = np.isfinite(uvd).all(axis=1) & (score >= min_score) & (uvd[:, 2] > 0)
        world = np.full((n, 3), np.nan)
        if keep.any():
            world[keep] = localise(uvd[keep], arrays["rot"][i, :n][keep].astype(float),
                                   arrays["trans"][i, :n][keep].astype(float),
                                   arrays["intr"][i, :n][keep].astype(float))
        far = np.linalg.norm(world[:, :2], axis=1) > 1000.0
        world[far] = np.nan
        metrics = flight_metrics(world * Config.DM_PER_M, arrays["xyz"][i, :n].astype(float))
        rows.append({"method": "snv3d", "share": 1.0, "view": int(arrays["view"][i]),
                     "half": int(arrays["half"][i]), "group": int(arrays["group"][i]), "n": n,
                     "cut": n, "start_high": bool(arrays["start_high"][i]),
                     "end_high": bool(arrays["end_high"][i]),
                     "z_start": float(arrays["z_start"][i]), "z_end": float(arrays["z_end"][i]),
                     "apex_true": float(np.nanmax(arrays["xyz"][i, :n, 2])),
                     "fit_score": float(1.0 - np.nanmean(score[keep])) if keep.any() else np.nan,
                     "omega": np.nan, "min_score": min_score, **metrics})
        values = world * Config.DM_PER_M
        frames.append(pd.DataFrame({
            "method": "snv3d", "view": int(arrays["view"][i]), "half": int(arrays["half"][i]),
            "group": int(arrays["group"][i]), "file_name": a + np.arange(n),
            "x_predicted": values[:, 0], "y_predicted": values[:, 1], "z_predicted": values[:, 2],
            "fit_score": np.where(keep, 1.0 - score, np.nan), "omega": np.nan}))
    return rows, frames


def load(fold: str, split: str, dataset: str, root: Path = Config.EVAL_DATA) -> dict:
    with np.load(root / dataset / f"fold_{fold}" / f"{split}.npz") as archive:
        return {k: archive[k] for k in archive.files}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="test set name, e.g. gt_full or real_full")
    parser.add_argument("--fold", required=True, choices=sorted(Config.FOLDS))
    parser.add_argument("--splits", nargs="+", default=["val"])
    parser.add_argument("--boxes", default="opt", choices=("opt", "track"))
    parser.add_argument("--data-root", type=Path, default=Config.EVAL_DATA)
    parser.add_argument("--out-root", type=Path, default=Config.EVAL_ROOT / "pred")
    args = parser.parse_args()
    tag = "snv3d_opt" if args.boxes == "opt" else "snv3d"
    name = args.dataset
    out = args.out_root / name / tag
    out.mkdir(parents=True, exist_ok=True)
    boxes = None
    if args.boxes == "opt":
        boxes = {(v, h): best_boxes(v, h) for v in Config.VIEWS for h in (0, 1)}
    train = load(args.fold, "train", name, args.data_root)
    grid = {}
    for min_score in SCORE_GRID if boxes is not None else (0.0,):
        rows, _ = predict_split(train, boxes, min_score)
        table = pd.DataFrame(rows)
        # The tables report error over the answered frames, and the capped median is 10 m at
        # every threshold, so the threshold is chosen on that (train split only).
        key = "rmse_answered" if boxes is not None else "rmse_cap"
        grid[min_score] = float(table[key].median())
    chosen = min(grid, key=grid.get)
    print(f"fold {args.fold}: train median RMSE by min score {grid}, chosen {chosen}")
    for split in args.splits:
        stem = out / f"fold_{args.fold}_{split}"
        if Path(f"{stem}_flights.csv.gz").exists():
            raise FileExistsError(f"{stem}_flights.csv.gz exists")
        rows, frames = predict_split(load(args.fold, split, name, args.data_root), boxes, chosen)
        pd.DataFrame(rows).to_csv(f"{stem}_flights.csv.gz", index=False)
        pd.concat(frames).to_csv(f"{stem}_frames.csv.gz", index=False)
        Path(f"{stem}_meta.json").write_text(json.dumps(
            {"fold": args.fold, "split": split, "min_score_grid_train": grid,
             "min_score": chosen, "dataset": args.dataset, "boxes": args.boxes,
             "extract": str(EXTRACT) if boxes is not None else None}, indent=1))
        table = pd.DataFrame(rows)
        print(split, table.groupby(table["view"] == 0)[["rmse_cap", "coverage"]].median())


if __name__ == "__main__":
    main()
