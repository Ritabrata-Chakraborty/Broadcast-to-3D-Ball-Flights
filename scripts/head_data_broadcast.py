"""Head-training data for the YOLO and PnLCalib heads on LP broadcast, one half at a time.

Train images are frames of that half outside every flight window (and outside pauses), so the
heads never see a flight that the trajectory stage validates or tests on. Validation images are the
frames of that half's own validation flights. Ball labels are the dataset's detector boxes that
agree with both its track and the projected 3D GT; frames whose ball cannot be labelled that way
are skipped, and frames with the GT ball outside the image are kept as negatives. Keypoint targets
are PnLCalib's 57 pitch points projected by the dataset camera.
Output: outputs/heads/broadcast/data/<half>/ (images, labels, keypoints_<split>.npz, data.yaml).
Video frames are non-commercial research data: never commit them.
Run in the snv3d env: scripts/py --env snv3d scripts/head_data_broadcast.py
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as Config
from yolo_frames import iter_frames

OUT = Config.HEADS_DIR / "broadcast" / "data"
PNL_ROOT = Config.THIRD_PARTY_ROOT / "PnLCalib"
CENTRE_M = np.array([*Config.LP_PITCH_CENTRE_DM, 0.0]) / Config.DM_PER_M
CAM_COLS = ["rot_x", "rot_y", "rot_z", "tx", "ty", "tz", "fx", "fy", "princ_x", "princ_y"]
REF_W_PX, REF_H_PX = 640.0, 360.0
TRAIN_STRIDE = 4
VAL_STRIDE = 2
WINDOW_MARGIN = 10        # frames kept clear around each flight window
TRACK_MATCH_PX = 8.0      # ball3d box bottom centre to ball3d's tracked point
GT_MATCH_PX = 40.0        # ball3d box centre to the projected GT, which is biased by about 10 px
NEGATIVE_SHARE = 0.1      # cap on frames with the GT ball outside the image
GOAL_TOP_KEYPOINTS = (12, 15, 16, 19)
GOAL_HEIGHT_M = 2.44
SEED = 0


def template_points() -> np.ndarray:
    """PnLCalib's 57 keypoints in its world frame: metres, pitch centre origin, z down."""
    sys.path.insert(0, str(PNL_ROOT))
    from utils.utils_calib import keypoint_world_coords_2D
    points = np.c_[np.asarray(keypoint_world_coords_2D[:57], float), np.zeros(57)]
    points[[k - 1 for k in GOAL_TOP_KEYPOINTS], 2] = -GOAL_HEIGHT_M
    return points


def dataset_cameras(base: Path) -> pd.DataFrame:
    """Dataset camera per frame: Rodrigues rotation, translation (dm), intrinsics at 640x360."""
    cams = pd.read_csv(base / "camera_smooth.csv").drop_duplicates("frame_index")
    return cams.set_index("frame_index")[CAM_COLS]


def project(points: np.ndarray, row: np.ndarray, width: float, height: float) -> tuple:
    """Pixels of points given in the camera's input frame (x, y, -height), and in-image flags."""
    rot = Rotation.from_rotvec(row[:3].copy()).as_matrix()
    x_cam = points @ rot.T + row[3:6] / Config.DM_PER_M
    fx, fy = row[6] * width / REF_W_PX, row[7] * height / REF_H_PX
    cx, cy = row[8] * width / REF_W_PX, row[9] * height / REF_H_PX
    with np.errstate(invalid="ignore", divide="ignore"):
        uv = np.c_[fx * x_cam[:, 0] / x_cam[:, 2] + cx, fy * x_cam[:, 1] / x_cam[:, 2] + cy]
    inside = (x_cam[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < width)
    inside &= (uv[:, 1] >= 0) & (uv[:, 1] < height)
    return uv, inside


def flight_windows(half: str) -> pd.DataFrame:
    """Broadcast view-flights of one half from both folds, with their split in this half's fold."""
    rows = []
    for fold in Config.HALF_FOLDS:
        for split in ("train", "val", "test"):
            with np.load(Config.FLIGHTS_DIR / f"fold_{fold}" / f"{split}.npz") as z:
                keep = (z["view"] == 0) & (z["half"] == Config.HALVES.index(half))
                rows.append(pd.DataFrame({"a": z["a"][keep], "n": z["n"][keep],
                                          "fold": fold, "split": split}))
    return pd.concat(rows, ignore_index=True)


def choose_frames(half: str, gt: pd.DataFrame, pauses: pd.DataFrame) -> dict[str, np.ndarray]:
    """Train frames outside flights and pauses; validation frames inside this fold's val flights."""
    fold = next(f for f, h in Config.HALF_FOLDS.items() if h == half)
    windows = flight_windows(half)
    last = int(gt.index.max()) + 1
    blocked = np.zeros(last + WINDOW_MARGIN + Config.MAX_FRAMES, bool)
    for a, b in zip(pauses["start_pause"], pauses["end_pause"]):
        blocked[a:b + 1] = True
    in_flight = np.zeros_like(blocked)
    for a, n in zip(windows["a"], windows["n"]):
        in_flight[max(a - WINDOW_MARGIN, 0):a + n + WINDOW_MARGIN] = True
    frames = gt.index.to_numpy()
    train = frames[~blocked[frames] & ~in_flight[frames]][::TRAIN_STRIDE]
    val_rows = windows[(windows["fold"] == fold) & (windows["split"] == "val")]
    val = np.unique(np.concatenate([np.arange(a, a + n) for a, n in zip(val_rows["a"],
                                                                          val_rows["n"])]))
    val = val[np.isin(val, frames) & ~blocked[val]][::VAL_STRIDE]
    return {"train": train, "val": val}


def ball_label(frame: int, boxes: dict, tracked: pd.DataFrame, gt_uv: np.ndarray) -> np.ndarray:
    """The ball3d box that agrees with ball3d's track and the projected GT, or None."""
    if frame not in boxes or frame not in tracked.index:
        return None
    track_uv = tracked.loc[frame, ["xk", "yk"]].to_numpy(float)
    if not np.isfinite(track_uv).all():
        return None
    box = boxes[frame]
    bottom = np.c_[(box[:, 0] + box[:, 2]) / 2, box[:, 3]]
    centre = np.c_[(box[:, 0] + box[:, 2]) / 2, (box[:, 1] + box[:, 3]) / 2]
    to_track = np.hypot(*(bottom - track_uv).T)
    best = int(np.argmin(to_track))
    if to_track[best] > TRACK_MATCH_PX or np.hypot(*(centre[best] - gt_uv)) > GT_MATCH_PX:
        return None
    return box[best]


def build_half(half: str) -> dict:
    base = Config.DATA_ROOT / Config.LP / "broadcast" / half
    meta = json.loads((base / "sequence_metadata.json").read_text())
    width, height = float(meta["width"]), float(meta["height"])
    gt = pd.read_csv(base / "track/ball_3d-gt.csv").drop_duplicates("file_name")
    gt = gt.set_index("file_name")[["x", "y", "z"]] / Config.DM_PER_M
    cams = dataset_cameras(base)
    gt = gt[gt.index.isin(cams.index)]
    pauses = pd.read_csv(base / "dev/pauses.csv")
    tracked = pd.read_csv(base / "track/ball_3d.basic_fitg.csv", usecols=["file_name", "xk", "yk"],
                          low_memory=False).drop_duplicates("file_name").set_index("file_name")
    det = pd.read_csv(base / "detection/ball_detection.csv", usecols=["file_name", "x0", "y0",
                                                                        "x1", "y1"])
    corners = ["x0", "y0", "x1", "y1"]
    boxes = {int(f): t[corners].to_numpy(float) for f, t in det.groupby("file_name")}
    template = template_points() + CENTRE_M   # broadcast is not flipped against ball3d's frame
    rng = np.random.default_rng(SEED)

    plan, counts = {}, {}
    for split, frames in choose_frames(half, gt, pauses).items():
        kept, labels, kps = [], [], []
        n_negative = 0
        for frame in frames:
            ball = gt.loc[frame].to_numpy(float) * np.array([1.0, 1.0, -1.0])
            row = cams.loc[frame].to_numpy(float)
            ball_uv, ball_in = project(ball[None], row, width, height)
            if ball_in[0]:
                box = ball_label(int(frame), boxes, tracked, ball_uv[0])
                if box is None:
                    continue
            else:
                if n_negative >= NEGATIVE_SHARE * len(frames) or rng.random() > 0.5:
                    continue
                box = None
                n_negative += 1
            uv, inside = project(template, row, width, height)
            kept.append(int(frame))
            labels.append(box)
            kps.append(np.where(inside[:, None], uv, np.nan))
        plan[split] = (np.asarray(kept), labels, np.asarray(kps))
        counts[split] = {"candidates": int(len(frames)), "kept": len(kept),
                         "negatives": n_negative}

    out = OUT / half
    cap = cv2.VideoCapture(str(base / "input.mkv"))
    for split, (frames, labels, kps) in plan.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        lookup = dict(zip(frames.tolist(), labels))
        order = np.sort(frames)
        written = 0
        for frame, image in iter_frames(cap, order):
            cv2.imwrite(str(out / "images" / split / f"{frame:06d}.jpg"), image,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            box = lookup[frame]
            text = ""
            if box is not None:
                cx, cy = (box[0] + box[2]) / 2 / width, (box[1] + box[3]) / 2 / height
                text = f"0 {cx:.6f} {cy:.6f} {(box[2] - box[0]) / width:.6f} " \
                       f"{(box[3] - box[1]) / height:.6f}\n"
            (out / "labels" / split / f"{frame:06d}.txt").write_text(text)
            written += 1
        if written != len(order):
            raise RuntimeError(f"{half} {split}: wrote {written} of {len(order)} frames")
        np.savez_compressed(out / f"keypoints_{split}.npz", frames=frames,
                            uv=kps.astype(np.float32), width=width, height=height)
    cap.release()
    (out / "data.yaml").write_text(f"path: {out}\ntrain: images/train\nval: images/val\n"
                                   f"names:\n  0: ball\n")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--halves", nargs="+", default=list(Config.HALVES))
    args = parser.parse_args()
    summary = {}
    for half in args.halves:
        if (OUT / half / "data.yaml").exists():
            raise FileExistsError(f"{OUT / half} exists; move it away first")
        summary[half] = build_half(half)
        print(half, summary[half], flush=True)
    (OUT / "build.json").write_text(json.dumps(
        {"halves": summary, "train_stride": TRAIN_STRIDE, "val_stride": VAL_STRIDE,
         "window_margin": WINDOW_MARGIN, "track_match_px": TRACK_MATCH_PX,
         "gt_match_px": GT_MATCH_PX, "negative_share": NEGATIVE_SHARE, "seed": SEED}, indent=1))


if __name__ == "__main__":
    main()
