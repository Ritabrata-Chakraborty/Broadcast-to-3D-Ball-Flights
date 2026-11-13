"""Shared parts of the calibration comparison: dataset listing, references and raw storage."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as Config  # noqa: E402

N_KEYPOINTS = 57
N_LINES = 23
MAIN_CAMERA = "Main camera center"
SN_SPLITS = {"SN23-valid": ("calibration-2023", "valid"), "SN23-test": ("calibration-2023", "test"),
             "SN22-test-center": ("calibration", "test")}
IMAGE_DATASETS = (*SN_SPLITS, "WC14-test", "TSWC-test")
VIDEO_DATASETS = ("ISSIA", "LP-broadcast", "LP-static")
HOMOGRAPHY_DATASETS = ("WC14-test", "TSWC-test")
HOMOGRAPHY_WH = (1280, 720)
GOAL_KEYPOINTS = (12, 15, 16, 19)      # off the ground plane, per PnLCalib's coords_to_dict
GOAL_LINES = (7, 8, 9, 10, 11, 12)
TSWC_IMAGES = "Dataset/80_95"
TSWC_HOMOGRAPHIES = "Annotations/80_95"


# ---------------------------------------------------------------- Dataset plan


def dataset_images(dataset: str) -> list[Path]:
    """Sorted images of an image dataset, with the SN22 and SN23-valid subsets applied."""
    if dataset == "WC14-test":
        return sorted(Config.WC14_ROOT.glob("*.jpg"), key=lambda p: int(p.stem))
    if dataset == "TSWC-test":
        seqs = (Config.TSWC_ROOT / "test.txt").read_text().split()
        root = Config.TSWC_ROOT / TSWC_IMAGES
        return [p for s in seqs for p in sorted((root / s).glob("*.jpg"))]
    task, split = SN_SPLITS[dataset]
    root = Config.SN_DATA_ROOT / task / split
    images = sorted(root.glob("*.jpg"))
    if dataset == "SN22-test-center":
        info = json.loads((root / "match_info_cam_gt.json").read_text())
        images = [p for p in images if info.get(p.name, {}).get("camera") == MAIN_CAMERA]
    if dataset == "SN23-valid":
        rng = np.random.default_rng(Config.CALIB_SEED)
        keep = np.sort(rng.choice(len(images), Config.CALIB_TUNE_IMAGES, replace=False))
        images = [images[i] for i in keep]
    return images


def image_id(dataset: str, path: Path) -> str:
    """Identifier of one image; TSWC names repeat across sequences, so it adds the sequence."""
    if dataset == "TSWC-test":
        return f"{path.parent.parent.name}__{path.parent.name}__{path.stem}"
    return path.stem


def annotation_root(dataset: str) -> Path | None:
    """Directory of line annotations in SoccerNet format; None for video and for TSWC."""
    if dataset in SN_SPLITS:
        task, split = SN_SPLITS[dataset]
        return Config.SN_DATA_ROOT / task / split
    return Config.WC14_ROOT if dataset == "WC14-test" else None


def annotated_homography(dataset: str, frame_id: str) -> np.ndarray:
    """Annotated image-to-template homography of one WC14 or TSWC image, with H[2, 2] = 1."""
    if dataset == "WC14-test":
        gt = np.loadtxt(Config.WC14_ROOT / f"{frame_id}.homographyMatrix").reshape(3, 3)
    else:
        side, seq, name = frame_id.split("__")
        gt = np.load(Config.TSWC_ROOT / TSWC_HOMOGRAPHIES / side / seq / f"{name}_homography.npy")
    return gt / gt[2, 2]


def play_clip_starts(unit: str, n_clips: int, rng: np.random.Generator) -> list[int]:
    """Evenly spread starts of clips whose frames all have a dataset camera and are in play."""
    cams = pd.read_csv(Config.DATA_ROOT / unit / "camera_smooth.csv", usecols=["frame_index"])
    is_ok = np.zeros(int(cams["frame_index"].max()) + 1, bool)
    is_ok[cams["frame_index"].to_numpy()] = True
    pauses = Config.DATA_ROOT / unit / "dev" / "pauses.csv"
    if pauses.exists():
        for row in pd.read_csv(pauses).itertuples():
            is_ok[row.start_pause:row.end_pause] = False
    length = Config.CALIB_CLIP_FRAMES
    windows = np.convolve(is_ok, np.ones(length, int), "valid") == length
    starts = np.flatnonzero(windows)
    if len(starts) == 0:
        return []
    return sorted(int(rng.choice(s)) for s in np.array_split(starts, n_clips) if len(s))


def video_plan() -> pd.DataFrame:
    """Frames per video unit: dataset, unit, clip, frame. Deterministic for a given seed."""
    rng = np.random.default_rng(Config.CALIB_SEED)
    rows = []
    for camera in Config.ISSIA_PNL_CAMERAS:
        unit = f"ISSIA-3D/camera{camera:02d}/half_1"
        cams = pd.read_csv(Config.DATA_ROOT / unit / "camera_smooth.csv", usecols=["frame_index"])
        frames = rng.choice(cams["frame_index"].unique(), Config.CALIB_ISSIA_FRAMES, replace=False)
        # Each ISSIA frame is its own clip: the bootstrap resamples frames within a camera.
        rows += [("ISSIA", unit, int(f), int(f)) for f in np.sort(frames)]
    units = [("LP-broadcast", f"{Config.LP}/broadcast/{h}", Config.CALIB_LP_BROADCAST_CLIPS)
             for h in Config.HALVES]
    units += [("LP-static", f"{Config.LP}/camera{c:02d}/{h}", Config.CALIB_LP_STATIC_CLIPS)
              for c in range(1, 7) for h in Config.HALVES]
    for group, unit, n_clips in units:
        for clip, start in enumerate(play_clip_starts(unit, n_clips, rng)):
            rows += [(group, unit, clip, start + i) for i in range(Config.CALIB_CLIP_FRAMES)]
    return pd.DataFrame(rows, columns=["dataset", "unit", "clip", "frame"])


# ---------------------------------------------------------------- Focal length reference


def focal_from_plane(forward: np.ndarray, principal: tuple[float, float]) -> float:
    """Focal length (px) from a plane-to-image homography with square pixels; NaN if unreliable.

    With K = diag(f, f, 1) after centring, the first two columns of K^-1 G are orthogonal and of
    equal length. That gives two estimates of f^2, and both must exist and agree.
    """
    centred = np.array([[1.0, 0.0, -principal[0]], [0.0, 1.0, -principal[1]], [0.0, 0.0, 1.0]])
    g = centred @ forward
    g1, g2 = g[:, 0] / np.linalg.norm(g[:, 0]), g[:, 1] / np.linalg.norm(g[:, 0])
    denominators = (g1[2] * g2[2], g2[2] ** 2 - g1[2] ** 2)
    if min(abs(denominators[0]), abs(denominators[1])) < 1e-14:
        return np.nan
    orthogonal = -(g1[0] * g2[0] + g1[1] * g2[1]) / denominators[0]
    equal_norm = (g1[0] ** 2 + g1[1] ** 2 - g2[0] ** 2 - g2[1] ** 2) / denominators[1]
    if orthogonal <= 0 or equal_norm <= 0:
        return np.nan
    f1, f2 = np.sqrt(orthogonal), np.sqrt(equal_norm)
    if abs(f1 - f2) / min(f1, f2) > Config.CALIB_FOCAL_MAX_DISAGREE:
        return np.nan
    return float((f1 + f2) / 2.0)


# ---------------------------------------------------------------- Raw network outputs


def raw_dir(model: str, dataset: str) -> Path:
    return Config.CALIB_ROOT / "raw" / model / dataset


def save_part(path: Path, ids: list, size_wh: list, kp: np.ndarray, ln: np.ndarray) -> None:
    """Write one part atomically, so a killed run resumes from complete parts only."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, ids=np.array(ids), size_wh=np.array(size_wh, np.int32),
                        kp=kp.astype(np.float32), ln=ln.astype(np.float32))
    tmp.replace(path)


def threshold_points(kp: np.ndarray, ln: np.ndarray, kp_th: float, line_th: float,
                     is_ground_only: bool = False) -> tuple[dict, dict]:
    """Same selection as PnLCalib's coords_to_dict, on arrays (57, 3) and (23, 2, 3)."""
    skip_kp = GOAL_KEYPOINTS if is_ground_only else ()
    skip_ln = GOAL_LINES if is_ground_only else ()
    keypoints = {i + 1: {"x": float(p[0]), "y": float(p[1]), "p": float(p[2])}
                 for i, p in enumerate(kp) if p[2] > kp_th and i + 1 not in skip_kp}
    lines = {i + 1: {"x_1": float(a[0]), "y_1": float(a[1]), "p_1": float(a[2]),
                     "x_2": float(b[0]), "y_2": float(b[1]), "p_2": float(b[2])}
             for i, (a, b) in enumerate(ln)
             if a[2] > line_th and b[2] > line_th and i + 1 not in skip_ln}
    return keypoints, lines


def main() -> None:
    """Write the fixed video frame plan once; an existing plan is kept."""
    out = Config.CALIB_ROOT / "video_plan.csv"
    if out.exists():
        print(f"[keep] {out}")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    plan = video_plan()
    plan.to_csv(out, index=False)
    print(plan.groupby(["dataset", "unit"])["frame"].agg(["count", "min", "max"]).to_string())


if __name__ == "__main__":
    main()


def video_source(unit: str) -> tuple[Path, bool]:
    """Video path of a unit and whether it must be mirrored to match the dataset camera."""
    if not unit.startswith("ISSIA-3D/"):
        return Config.DATA_ROOT / unit / "input.mkv", False
    camera = int(unit.split("/")[1].removeprefix("camera"))
    path = Config.DATA_ROOT / Config.ISSIA_VIDEO.format(camera=camera)
    return path, camera in Config.ISSIA_MIRRORED_CAMERAS
