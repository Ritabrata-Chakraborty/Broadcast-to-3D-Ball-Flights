"""Head-training data for the static cameras 01-05, one LP half at a time.

Frames: each static camera of the half, outside every flight window (+-10 frames) and pause,
sampled as pairs 6 frames apart at seek points about 260 frames apart (4K decoding is the cost).
Labels follow head_data_broadcast.py: the dataset's detector box within 8 px of its tracked ball
bottom and 40 px of the projected GT (both at 1920 wide). YOLO images are the 2 x 2 inference tile
holding the ball, plus a ball-free tile on a quarter of the frames; PnLCalib images are the whole
frame at 1920 x 1080 with its 57 keypoints from the dataset camera. The last 10 % of each camera's
frames by time are validation.
Output: outputs/heads/static/data/<half>/ (yolo/{images,labels}/{train,val}, data.yaml,
images/{train,val}, keypoints_{train,val}.npz). Video frames: never commit.
Run in the snv3d env: scripts/py --env snv3d scripts/head_data_static.py --half half_1
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as Config
from yolo_frames import TILE_OVERLAP_PX, iter_frames, tile_origins
from head_data_broadcast import CENTRE_M, dataset_cameras, project, template_points

OUT = Config.HEADS_DIR / "static" / "data"
VIEW_NAMES = ("broadcast", *[f"camera{c:02d}" for c in range(1, 7)])
TILE = (1984, 1112)
SEEK_STEP = 260
PAIR_GAP = 6
WINDOW_MARGIN = 10
TRACK_MATCH_PX = 8.0         # at 1920 wide
GT_MATCH_PX = 40.0           # at 1920 wide
NEGATIVE_TILE_SHARE = 0.25
VAL_SHARE = 0.1
PNL_WIDTH = 1920
SEED = 0


def flight_windows(view: int, half: int) -> list[tuple[int, int]]:
    """(first, last) frame of every view-flight of one static unit."""
    windows = []
    for fold in Config.FOLDS:
        for split in ("train", "val", "test"):
            with np.load(Config.FOLDS_DIR / "gt" / f"fold_{fold}" / f"{split}.npz") as z:
                keep = (z["view"] == view) & (z["half"] == half)
                windows += list(zip(z["a"][keep], z["a"][keep] + z["n"][keep] - 1))
    return sorted(set(windows))


def choose_frames(view: int, half: int, n_frames: int, pauses: pd.DataFrame) -> np.ndarray:
    blocked = np.zeros(n_frames + PAIR_GAP + 1, bool)
    for a, b in flight_windows(view, half):
        blocked[max(0, a - WINDOW_MARGIN):b + WINDOW_MARGIN + 1] = True
    for a, b in zip(pauses["start_pause"], pauses["end_pause"]):
        blocked[int(a):int(b) + 1] = True
    starts = np.arange(SEEK_STEP // 2, n_frames - PAIR_GAP, SEEK_STEP)
    frames = np.sort(np.r_[starts, starts + PAIR_GAP])
    return frames[~blocked[frames]]


def ball_label(frame: int, boxes: dict, tracked: pd.DataFrame, gt_uv: np.ndarray,
               scale: float) -> np.ndarray | None:
    """ball3d's box that agrees with its tracked ball bottom and the projected GT, or None."""
    if frame not in boxes or frame not in tracked.index:
        return None
    track_uv = tracked.loc[frame, ["xk", "bottom"]].to_numpy(float)
    if not np.isfinite(track_uv).all():
        return None
    box = boxes[frame]
    bottom = np.c_[(box[:, 0] + box[:, 2]) / 2, box[:, 3]]
    centre = np.c_[(box[:, 0] + box[:, 2]) / 2, (box[:, 1] + box[:, 3]) / 2]
    to_track = np.hypot(*(bottom - track_uv).T) * scale
    best = int(np.argmin(to_track))
    if to_track[best] > TRACK_MATCH_PX or np.hypot(*(centre[best] - gt_uv)) * scale > GT_MATCH_PX:
        return None
    return box[best]


def tiles_of(width: int, height: int) -> list[tuple[int, int]]:
    return [(x, y) for y in tile_origins(height, TILE[1], TILE_OVERLAP_PX)
            for x in tile_origins(width, TILE[0], TILE_OVERLAP_PX)]


def yolo_line(box: np.ndarray, x0: int, y0: int) -> str:
    cx = ((box[0] + box[2]) / 2 - x0) / TILE[0]
    cy = ((box[1] + box[3]) / 2 - y0) / TILE[1]
    return f"0 {cx:.6f} {cy:.6f} {(box[2] - box[0]) / TILE[0]:.6f} " \
           f"{(box[3] - box[1]) / TILE[1]:.6f}\n"


def build_camera(view: int, half: int, out: Path, rng: np.random.Generator) -> dict:
    base = Config.DATA_ROOT / Config.LP / VIEW_NAMES[view] / Config.HALVES[half]
    meta = json.loads((base / "sequence_metadata.json").read_text())
    width, height = int(meta["width"]), int(meta["height"])
    scale = Config.IMAGE_WIDTH_PX / width
    gt = pd.read_csv(base / "track/ball_3d-gt.csv").drop_duplicates("file_name")
    gt = gt.set_index("file_name")[["x", "y", "z"]] / Config.DM_PER_M
    cams = dataset_cameras(base)
    tracked = pd.read_csv(base / "track/ball_3d.basic_fitg.csv", low_memory=False,
                          usecols=["file_name", "xk", "yk_original", "b"])
    tracked = tracked.drop_duplicates("file_name").set_index("file_name")
    tracked["bottom"] = tracked["yk_original"] + tracked["b"]
    det = pd.read_csv(base / "detection/ball_detection.csv",
                      usecols=["file_name", "x0", "y0", "x1", "y1"])
    corners = ["x0", "y0", "x1", "y1"]
    boxes = {int(f): t[corners].to_numpy(float) for f, t in det.groupby("file_name")}
    # PnLCalib names keypoints in its own pitch orientation; on a camera whose solves come out
    # turned by 180 degrees against the dataset frame (real build), keypoint k sits at flip(p_k).
    built = json.loads((Config.FOLDS_DIR / "real" / "build.json").read_text())
    flips = built["folds"]["1"]["flips"]
    is_flipped = flips[f"{VIEW_NAMES[view]}_{Config.HALVES[half]}"][0]
    points = template_points()
    if is_flipped:
        points = points * np.array([-1.0, -1.0, 1.0])
    template = points + CENTRE_M
    frames = choose_frames(view, half, int(gt.index.max()), pd.read_csv(base / "dev/pauses.csv"))
    frames = frames[np.isin(frames, cams.index)]
    cut = frames[int(len(frames) * (1 - VAL_SHARE))] if len(frames) else 0
    counts = {"frames": int(len(frames)), "positive_tiles": 0, "negative_tiles": 0,
              "pnl_images": 0, "flipped": bool(is_flipped)}
    keypoints = {"train": ([], []), "val": ([], [])}
    tiles = tiles_of(width, height)
    cap = cv2.VideoCapture(str(base / "input.mkv"))
    for frame, image in iter_frames(cap, frames):
        split = "val" if frame >= cut else "train"
        row = cams.loc[frame].to_numpy(float)
        name = f"{VIEW_NAMES[view]}_{frame:06d}"
        uv, inside = project(template, row, width, height)
        small = cv2.resize(image, (PNL_WIDTH, PNL_WIDTH * height // width),
                           interpolation=cv2.INTER_AREA)
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), small,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        keypoints[split][0].append(name)
        keypoints[split][1].append(np.where(inside[:, None], uv * PNL_WIDTH / width, np.nan))
        counts["pnl_images"] += 1
        if frame not in gt.index or not np.isfinite(gt.loc[frame].to_numpy(float)).all():
            continue
        ball = gt.loc[frame].to_numpy(float) * np.array([1.0, 1.0, -1.0])
        ball_uv, ball_in = project(ball[None], row, width, height)
        box = ball_label(int(frame), boxes, tracked, ball_uv[0], scale) if ball_in[0] else None
        if box is None:
            continue
        (out / "yolo" / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "yolo" / "labels" / split).mkdir(parents=True, exist_ok=True)
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        holding = [(x, y) for x, y in tiles if x + 8 <= cx <= x + TILE[0] - 8
                   and y + 8 <= cy <= y + TILE[1] - 8]
        if holding:
            x, y = min(holding, key=lambda t: abs(t[0] + TILE[0] / 2 - cx) +
                       abs(t[1] + TILE[1] / 2 - cy))
            tag = f"{name}_{x}_{y}"
            cv2.imwrite(str(out / "yolo" / "images" / split / f"{tag}.jpg"),
                        image[y:y + TILE[1], x:x + TILE[0]], [cv2.IMWRITE_JPEG_QUALITY, 95])
            (out / "yolo" / "labels" / split / f"{tag}.txt").write_text(yolo_line(box, x, y))
            counts["positive_tiles"] += 1
        empty = [(x, y) for x, y in tiles if not (x - 40 <= cx <= x + TILE[0] + 40 and
                                                    y - 40 <= cy <= y + TILE[1] + 40)]
        if empty and rng.random() < NEGATIVE_TILE_SHARE:
            x, y = empty[rng.integers(len(empty))]
            tag = f"{name}_{x}_{y}"
            cv2.imwrite(str(out / "yolo" / "images" / split / f"{tag}.jpg"),
                        image[y:y + TILE[1], x:x + TILE[0]], [cv2.IMWRITE_JPEG_QUALITY, 95])
            (out / "yolo" / "labels" / split / f"{tag}.txt").write_text("")
            counts["negative_tiles"] += 1
    cap.release()
    counts["keypoints"] = keypoints
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--half", required=True, choices=Config.HALVES)
    parser.add_argument("--views", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    args = parser.parse_args()
    half = Config.HALVES.index(args.half)
    out = OUT / args.half
    if (out / "data.yaml").exists():
        raise FileExistsError(f"{out} exists; move it away first")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    summary, names, uvs = {}, {"train": [], "val": []}, {"train": [], "val": []}
    for view in args.views:
        counts = build_camera(view, half, out, rng)
        for split in ("train", "val"):
            names[split] += counts["keypoints"][split][0]
            uvs[split] += counts["keypoints"][split][1]
        del counts["keypoints"]
        summary[VIEW_NAMES[view]] = counts
        print(VIEW_NAMES[view], counts, flush=True)
    for split in ("train", "val"):
        np.savez_compressed(out / f"keypoints_{split}.npz", frames=np.array(names[split]),
                            uv=np.asarray(uvs[split], np.float32), width=float(PNL_WIDTH),
                            height=float(PNL_WIDTH * 9 // 16))
    (out / "data.yaml").write_text(f"path: {out / 'yolo'}\ntrain: images/train\n"
                                   f"val: images/val\nnames:\n  0: ball\n")
    (out / "build.json").write_text(json.dumps(
        {"half": args.half, "views": args.views, "cameras": summary, "seek_step": SEEK_STEP,
         "pair_gap": PAIR_GAP, "window_margin": WINDOW_MARGIN, "track_match_px": TRACK_MATCH_PX,
         "gt_match_px": GT_MATCH_PX, "negative_tile_share": NEGATIVE_TILE_SHARE,
         "val_share": VAL_SHARE, "seed": SEED}, indent=1))


if __name__ == "__main__":
    main()
