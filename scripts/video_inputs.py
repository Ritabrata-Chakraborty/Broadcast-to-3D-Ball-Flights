"""Video inputs of one view and half: YOLO candidate boxes, the Track-CV tracker, PnLCalib cameras.

Library for build_folds.py, track_phys.py and input_quality.py. PnLCalib cameras (SoccerNet frame,
z down) are converted to the dataset frame with the pitch-centre constant and one 180-degree turn
per static unit; the tracker is a constant-velocity dynamic programme over the candidate boxes.
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as Config

EXTRACT = Config.EXTRACT_DIR / "published"
VIEWS = ("broadcast", *[f"camera{c:02d}" for c in range(1, 7)])
CENTRE_M = np.array([*Config.LP_PITCH_CENTRE_DM, 0.0]) / Config.DM_PER_M
FLIP = np.diag([-1.0, -1.0, 1.0])
MAX_REP_ERR_PX = Config.CALIB_MAX_REP_ERR_PX   # at 960 wide, as in the calibration study
MAX_CANDIDATES = 10
FOCAL_OUTLIER = 0.15      # broadcast frames whose focal is this far from the flight median
MIN_CAMERA_SHARE = 0.25   # a broadcast flight needs a camera on this share of its frames
SMOOTH_FRAMES = 5
# Correct picks sit 10-20 px off the projected GT (a fixed offset), so tuning scores at 40 px.
TUNE_HIT_PX = 40.0
RIVAL_PX = 40.0           # a box this far from the tracked point competes with it
MAX_GAP = 5               # frames a track may bridge without a box
MAX_FILL = 10             # interior gaps up to this many frames are interpolated
MAX_EXTEND = 5            # frames a track is extended past its first and last box
EXTEND_FIT = 4            # boxes used for that linear extension
TRACK_CANDIDATES = 6      # highest-scoring boxes per frame the tracker considers
CLUTTER_CELL_PX = 10.0
CLUTTER_SHARE = 0.02      # a cell holding boxes on more of a unit's frames is fixed clutter
# Tracker grid, searched on fold a train flights; the first entry is the fallback.
TRACKER_GRID = {"vmax_px": (80.0, 150.0), "accel_px": (10.0, 20.0, 40.0),
                "min_score": (0.2, 0.3, 0.4, 0.5, 0.6)}
DEFAULT_TRACKER = {"vmax_px": 80.0, "accel_px": 20.0, "min_score": 0.4}


# ---------------------------------------------------------------- Cameras


def to_dataset_camera(cam: dict, solve_wh: tuple, is_flipped: bool) -> tuple:
    """PnLCalib camera (SoccerNet frame, z down) as rot, trans (m) and intrinsics at 1920 wide."""
    rot_sn = np.asarray(cam["rotation_matrix"], float)
    centre = np.asarray(cam["position_meters"], float)
    flip = FLIP if is_flipped else np.eye(3)
    rot = rot_sn @ flip
    trans = -rot_sn @ (flip @ CENTRE_M + centre)
    scale = Config.IMAGE_WIDTH_PX / solve_wh[0]
    intr = np.array([cam["x_focal_length"], cam["y_focal_length"], *cam["principal_point"]]) * scale
    return rot, trans, intr


def camera_centre(rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    return -rot.T @ trans


def load_cameras(view: int, half: int, extract: Path = EXTRACT) -> pd.DataFrame:
    """Accepted PnLCalib solves of one unit, one row per frame, in both pitch orientations."""
    name = f"{VIEWS[view]}_{Config.HALVES[half]}"
    rows = []
    with gzip.open(extract / f"{name}.cam.jsonl.gz", "rt") as handle:
        for line in handle:
            row = json.loads(line)
            if not row["ok"] or row["rep_err"] > MAX_REP_ERR_PX * row["wh"][0] / 960.0:
                continue
            if not 0 < row["cam"]["x_focal_length"] < 50 * row["wh"][0]:
                continue
            for is_flipped in (False, True):
                rot, trans, intr = to_dataset_camera(row["cam"], row["wh"], is_flipped)
                rows.append({"frame": int(row["id"]), "flipped": is_flipped, "rot": rot,
                             "trans": trans, "intr": intr,
                             "centre": camera_centre(rot, trans),
                             "rep_err": float(row["rep_err"]) * 960.0 / row["wh"][0],
                             "n_kp": int(row["n_kp"])})
    return pd.DataFrame(rows)


def dataset_centres(view: int, half: int) -> pd.Series:
    """Dataset camera centres (m, height down as in ball3d) by frame, for the flip bit only."""
    base = Config.DATA_ROOT / Config.LP / VIEWS[view] / Config.HALVES[half]
    cams = pd.read_csv(base / "camera_smooth.csv").drop_duplicates("frame_index")
    rot = Rotation.from_rotvec(cams[["rot_x", "rot_y", "rot_z"]].to_numpy(float).copy())
    trans = cams[["tx", "ty", "tz"]].to_numpy(float) / Config.DM_PER_M
    centres = -np.einsum("nji,nj->ni", rot.as_matrix(), trans)
    return pd.Series(list(centres), index=cams["frame_index"].to_numpy())


def choose_flip(cams: pd.DataFrame, view: int, half: int) -> tuple[bool, float]:
    """One orientation per unit: the one whose camera sits nearer the dataset camera, by vote."""
    reference = dataset_centres(view, half)
    votes = []
    for frame, both in cams.groupby("frame"):
        if frame not in reference.index:
            continue
        gaps = {r.flipped: np.linalg.norm(r.centre - reference[frame]) for r in both.itertuples()}
        votes.append(gaps[True] < gaps[False])
    share = float(np.mean(votes)) if votes else 0.0
    return share > 0.5, share


def pack_camera(rot: np.ndarray, trans: np.ndarray, intr: np.ndarray) -> np.ndarray:
    """rotvec, centre and intrinsics as one interpolable vector."""
    rotvec = Rotation.from_matrix(rot).as_rotvec()
    return np.r_[rotvec, camera_centre(rot, trans), intr]


def unpack_camera(vector: np.ndarray) -> tuple:
    rot = Rotation.from_rotvec(vector[:3]).as_matrix()
    return rot, -rot @ vector[3:6], vector[6:10]


def static_camera(cams: pd.DataFrame) -> np.ndarray:
    """Median camera of a static unit: it does not move, so frames only average out noise."""
    vectors = np.stack([pack_camera(r.rot, r.trans, r.intr) for r in cams.itertuples()])
    return np.median(vectors, axis=0)


def broadcast_track(cams: pd.DataFrame, frames: np.ndarray) -> tuple[np.ndarray, float, dict]:
    """Per-frame cameras for one flight: focal outliers dropped, gaps filled, then smoothed.

    Also returns the calibration confidence per frame: whether the frame had its own accepted
    solve, its reprojection error (px at 960 wide) and keypoint count, and the focal spread.
    """
    chosen = cams[cams["frame"].isin(frames)].sort_values("frame")
    share = len(chosen) / len(frames)
    quality = {"cam_solved": np.zeros(len(frames), bool),
               "cam_rep_err": np.full(len(frames), np.nan),
               "cam_n_kp": np.zeros(len(frames)), "cam_focal_spread": np.nan}
    if len(chosen) == 0:
        return np.full((len(frames), 10), np.nan), 0.0, quality
    vectors = np.stack([pack_camera(r.rot, r.trans, r.intr) for r in chosen.itertuples()])
    focal = vectors[:, 6]
    keep = np.abs(focal / np.median(focal) - 1.0) <= FOCAL_OUTLIER
    vectors, known = vectors[keep], chosen["frame"].to_numpy()[keep]
    share = len(known) / len(frames)
    position = np.searchsorted(frames, known)
    quality["cam_solved"][position] = True
    quality["cam_rep_err"][position] = chosen["rep_err"].to_numpy()[keep]
    quality["cam_n_kp"][position] = chosen["n_kp"].to_numpy()[keep]
    quality["cam_focal_spread"] = float(np.std(vectors[:, 6] / np.median(vectors[:, 6])))
    # Rotation vectors are continuous here: one broadcast camera turns little within a flight.
    filled = np.stack([np.interp(frames, known, vectors[:, k]) for k in range(10)], axis=1)
    smooth = pd.DataFrame(filled).rolling(SMOOTH_FRAMES, center=True, min_periods=1).mean()
    return smooth.to_numpy(), share, quality


# ---------------------------------------------------------------- Tracking


def drop_clutter(boxes: pd.DataFrame, n_frames: int, scale: float) -> pd.DataFrame:
    """Remove boxes in image cells that hold a box on many frames: lights, logos, the scoreboard.

    An airborne ball does not stay in one cell, so this only removes fixed false positives.
    """
    u = ((boxes["x0"] + boxes["x1"]) / 2.0 * scale // CLUTTER_CELL_PX).astype(int)
    v = ((boxes["y0"] + boxes["y1"]) / 2.0 * scale // CLUTTER_CELL_PX).astype(int)
    cell = u * 100000 + v
    frames_per_cell = pd.Series(boxes["file_name"].to_numpy()).groupby(cell.to_numpy()).nunique()
    crowded = frames_per_cell[frames_per_cell > CLUTTER_SHARE * n_frames].index
    return boxes[~cell.isin(crowded).to_numpy()]


def candidates(view: int, half: int, extract: Path = EXTRACT) -> tuple[dict, float]:
    """Boxes per frame at 1920 px width: centre x, bottom y, centre y, diameter and score."""
    name = f"{VIEWS[view]}_{Config.HALVES[half]}"
    boxes = pd.read_csv(extract / f"{name}.det.csv.gz")
    meta = json.loads((extract / f"{name}.json").read_text())
    scale = Config.IMAGE_WIDTH_PX / meta["width"]
    boxes = drop_clutter(boxes, meta["frames"], scale)
    boxes = boxes.sort_values(["file_name", "score"], ascending=[True, False])
    boxes = boxes.groupby("file_name").head(MAX_CANDIDATES)
    table = np.c_[(boxes["x0"] + boxes["x1"]) / 2.0, boxes["y1"], (boxes["y0"] + boxes["y1"]) / 2.0,
                  ((boxes["x1"] - boxes["x0"]) + (boxes["y1"] - boxes["y0"])) / 2.0]
    table = np.c_[table * scale, boxes["score"].to_numpy()]
    out = {}
    for frame, rows in zip(boxes["file_name"].to_numpy(), table):
        out.setdefault(int(frame), []).append(rows)
    return {k: np.asarray(v) for k, v in out.items()}, scale


def track(frames: np.ndarray, boxes: dict, vmax_px: float, accel_px: float,
          min_score: float) -> np.ndarray:
    """Best constant-velocity chain of boxes through a flight, bridging up to MAX_GAP misses.

    A state is a box and the box before it, so a step pays for the change in image velocity,
    ((v_new - v_old) / accel_px)^2, not for the motion itself: a fast ball costs no more than a
    still player. Speeds above vmax_px per frame are forbidden. Each chosen box earns its score
    above min_score; frames without one cost nothing, so an empty track scores 0. Returns (n, 5):
    centre x, bottom y, centre y, diameter, score; NaN where missed.
    """
    options = [boxes.get(int(f), np.zeros((0, 5)))[:TRACK_CANDIDATES] for f in frames]
    size = len(frames)
    # Per node (t, j): incoming edges as cost, velocity (NaN at a track start) and back-pointer.
    cost = [[np.array([min_score - box[4]]) for box in opt]
            for t, opt in enumerate(options)]
    speed = [[np.full((1, 2), np.nan) for _ in opt] for opt in options]
    back = [[[(-1, -1, -1)] for _ in opt] for opt in options]
    for t in range(1, size):
        for j, box in enumerate(options[t]):
            for gap in range(1, min(MAX_GAP, t) + 1):
                prev = options[t - gap]
                for i in range(len(prev)):
                    velocity = (box[:2] - prev[i, :2]) / gap
                    if np.linalg.norm(velocity) > vmax_px:
                        continue
                    change = velocity - speed[t - gap][i]
                    accel = np.nan_to_num((change ** 2).sum(axis=1) / accel_px ** 2)
                    total = cost[t - gap][i] + accel
                    k = int(total.argmin())
                    value = total[k] + min_score - box[4]
                    cost[t][j] = np.r_[cost[t][j], value]
                    speed[t][j] = np.r_[speed[t][j], velocity[None]]
                    back[t][j].append((t - gap, i, k))
    out = np.full((size, 5), np.nan)
    best = (0.0, None)
    for t in range(size):
        for j in range(len(options[t])):
            k = int(cost[t][j].argmin())
            value = cost[t][j][k]
            if value < best[0]:
                best = (value, (t, j, k))
    node = best[1]
    while node is not None and node[0] >= 0:
        t, j, k = node
        out[t] = options[t][j]
        node = back[t][j][k]
    return out


def fill_track(path: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fill short gaps by interpolation and extend the ends a few frames by a linear fit.

    ball3d's tracked point is Kalman-smoothed and exists on every frame, which is what gives it
    an anchor on the contact frame; this is the closest simple equivalent. Returns the filled
    rows (score NaN where filled) and which frames were filled.
    """
    out = path.copy()
    seen = np.where(np.isfinite(path[:, 0]))[0]
    filled = np.zeros(len(path), bool)
    if len(seen) == 0:
        return out, filled
    frame = np.arange(len(path))
    for a, b in zip(seen[:-1], seen[1:]):
        if 1 < b - a <= MAX_FILL + 1:
            gap = frame[a + 1:b]
            for k in range(4):
                out[gap, k] = np.interp(gap, [a, b], path[[a, b], k])
            filled[gap] = True
    for ends, outside in ((seen[:EXTEND_FIT], frame[max(0, seen[0] - MAX_EXTEND):seen[0]]),
                          (seen[-EXTEND_FIT:], frame[seen[-1] + 1:seen[-1] + 1 + MAX_EXTEND])):
        if len(outside) == 0:
            continue
        for k in range(4):
            if len(ends) >= 2:
                slope, offset = np.polyfit(ends, path[ends, k], 1)
                out[outside, k] = slope * outside + offset
            else:
                out[outside, k] = path[ends[0], k]
        filled[outside] = True
    return out, filled


# ---------------------------------------------------------------- Dataset


def ground_hit(uv: np.ndarray, rot: np.ndarray, trans: np.ndarray, intr: np.ndarray) -> np.ndarray:
    """Pitch xy (m) where each pixel's ray meets the ground, as ball3d's x_pitch2D does."""
    x = (uv[:, 0] - intr[:, 2]) / intr[:, 0]
    y = (uv[:, 1] - intr[:, 3]) / intr[:, 1]
    direction = np.einsum("nji,nj->ni", rot, np.c_[x, y, np.ones_like(x)]) * [1.0, 1.0, -1.0]
    centre = -np.einsum("nji,nj->ni", rot, trans) * [1.0, 1.0, -1.0]
    with np.errstate(divide="ignore", invalid="ignore"):
        along = centre[:, 2] / -direction[:, 2]
        hit = centre + direction * along[:, None]
    hit[(along <= 0) | (along > Config.MAX_RAY_M)] = np.nan
    return hit[:, :2]


def build_flight(item, boxes: dict, cams: dict, tracker: dict, size: int, original: dict,
                 replace: tuple[str, ...] = ("track", "camera")) -> dict:
    """Tracked point, cameras and anchors of one view-flight, padded to the stored length.

    replace picks what R3 estimates; the rest stays as in the dataset (ball3d's track, the
    dataset cameras), which splits the loss between detection and calibration.
    """
    frames = np.arange(item.a, item.a + item.n)
    if "track" in replace:
        path, filled = fill_track(track(frames, boxes[(item.view, item.half)], **tracker))
    else:
        path = np.full((item.n, 5), np.nan)
        path[:, :2] = original["uv_det"][:item.n]
        filled = np.zeros(item.n, bool)
    unit = cams[(item.view, item.half)]
    quality = {"cam_solved": np.ones(item.n, bool), "cam_rep_err": np.full(item.n, np.nan),
               "cam_n_kp": np.full(item.n, np.nan), "cam_focal_spread": np.nan}
    if "camera" not in replace:
        rot = original["rot"][:item.n].astype(float)
        trans = original["trans"][:item.n].astype(float)
        intr = original["intr"][:item.n].astype(float)
        share, has_camera = 1.0, True
    elif item.view == 0:
        vectors, share, quality = broadcast_track(unit["frames"], frames)
    else:
        vectors = np.repeat(unit["static"][None], len(frames), axis=0)
        share = 1.0
    if "camera" in replace:
        has_camera = share >= MIN_CAMERA_SHARE and np.isfinite(vectors).all()
        if not has_camera:
            vectors = np.repeat(unit["fallback"][None], len(frames), axis=0)
        parts = [unpack_camera(v) for v in vectors]
        rot = np.stack([p[0] for p in parts])
        trans = np.stack([p[1] for p in parts])
        intr = np.stack([p[2] for p in parts])
    pitch = ground_hit(path[:, :2], rot, trans, intr)
    if not has_camera:
        pitch[:] = np.nan      # no anchor, so the flight is not answered, as ball3d would
    out = {"uv_det": np.full((size, 2), np.nan), "pitch_xy": np.full((size, 2), np.nan),
           "filled": np.zeros(size, bool), "rival_score_det": np.full(size, np.nan),
           "cam_solved": np.zeros(size, bool), "cam_rep_err": np.full(size, np.nan),
           "cam_n_kp": np.full(size, np.nan),
           "size_det": np.full(size, np.nan), "score_det": np.full(size, np.nan),
           "rot": np.repeat(rot[-1:], size, axis=0), "trans": np.repeat(trans[-1:], size, axis=0),
           "intr": np.repeat(intr[-1:], size, axis=0)}
    out["uv_det"][:item.n] = path[:, :2]
    out["pitch_xy"][:item.n] = pitch
    out["size_det"][:item.n] = path[:, 3]
    out["score_det"][:item.n] = path[:, 4]
    out["filled"][:item.n] = filled
    out["rival_score_det"][:item.n] = rival_scores(frames, path, boxes[(item.view, item.half)])
    for key in ("cam_solved", "cam_rep_err", "cam_n_kp"):
        out[key][:item.n] = quality[key]
    out["cam_focal_spread"] = quality["cam_focal_spread"]
    out["rot"][:item.n], out["trans"][:item.n], out["intr"][:item.n] = rot, trans, intr
    out.update(camera_share=share, has_camera=has_camera)
    return out


def rival_scores(frames: np.ndarray, path: np.ndarray, boxes: dict) -> np.ndarray:
    """Best score of a box at least RIVAL_PX from the tracked point, per frame (0 if none)."""
    out = np.zeros(len(frames))
    for k, frame in enumerate(frames):
        rows = boxes.get(int(frame))
        if rows is None or not np.isfinite(path[k, 0]):
            continue
        far = np.hypot(rows[:, 0] - path[k, 0], rows[:, 1] - path[k, 1]) >= RIVAL_PX
        out[k] = rows[far, 4].max() if far.any() else 0.0
    return out
