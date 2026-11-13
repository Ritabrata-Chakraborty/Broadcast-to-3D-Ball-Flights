"""Input quality on the test flights: PnLCalib cameras and YOLO ball tracks, published vs
fine-tuned heads (det,cal), against the dataset camera and the projected 3D ground truth.

Each fold's test flights are scored with that fold's inputs; the four folds cover every view-flight
once. Output: outputs/eval/report/inputs/{calibration,tracking}.csv and INPUTS.md (supplement).
Run: scripts/py --env snv3d scripts/input_quality.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import config as Config  # noqa: E402
from video_inputs import candidates  # noqa: E402

OUT = Config.EVAL_DATA.parent / "report" / "inputs"
VIEW_NAMES = ("Broadcast", "Cam01", "Cam02", "Cam03", "Cam04", "Cam05")
NEAR_PX = 30.0
SETTINGS = {"published": "real_src", "fine-tuned": "heads_src"}
EXTRACT_SOURCE = {"published": "real_cv", "fine-tuned": "heads_cv"}


# ---------------------------------------------------------------- Inputs


def load(name: str, fold: str) -> dict:
    with np.load(Config.EVAL_DATA / name / f"fold_{fold}" / "test.npz") as z:
        return {k: z[k] for k in z.files}


def extract_dir(setting: str, fold: str) -> Path:
    build = json.loads((Config.FOLDS_DIR / EXTRACT_SOURCE[setting] / "build.json").read_text())
    path = Path(build["source_build"]["folds"][fold]["extract"])
    return path if path.is_absolute() else ROOT / path


def top_detection(data: dict, extract: Path) -> np.ndarray:
    """Highest-score YOLO box per frame (centre x, bottom y at 1920 px), the tracker's own input."""
    out = np.full(data["uv_gt"].shape, np.nan)
    cache = {}
    for i, (view, half, a, n) in enumerate(zip(data["view"], data["half"], data["a"], data["n"])):
        key = (int(view), int(half))
        if key not in cache:
            cache[key] = candidates(*key, extract=extract)[0]
        for t in range(n):
            boxes = cache[key].get(int(a + t))
            if boxes is not None and len(boxes):
                out[i, t] = boxes[boxes[:, 4].argmax(), :2]
    return out


def gt_pixels(ref: dict, cam: dict) -> np.ndarray:
    """3D ground truth projected with the cameras of `cam`."""
    import torch     # local: the figure script imports this module in the CPU env
    from arcprior.rays import project
    tensor = {k: torch.tensor(cam[k], dtype=torch.float64) for k in ("rot", "trans", "intr")}
    xyz = torch.tensor(ref["xyz"], dtype=torch.float64)
    return project(xyz, tensor["rot"], tensor["trans"], tensor["intr"])[0].numpy()


# ---------------------------------------------------------------- Scores


def frames_mask(ref: dict) -> np.ndarray:
    steps = np.arange(ref["uv_gt"].shape[1])[None]
    return ref["visible"] & (steps < ref["n"][:, None]) & np.isfinite(ref["uv_gt"]).all(-1)


def track_rows(ref: dict, uv: np.ndarray, label: str, mask: np.ndarray) -> list[dict]:
    err = np.linalg.norm(uv - ref["uv_gt"], axis=-1)
    rows = []
    for view in range(len(VIEW_NAMES)):
        m = mask & (ref["view"] == view)[:, None]
        e = err[m]
        present = np.isfinite(e)
        rows.append({"view": VIEW_NAMES[view], "input": label, "frames": int(m.sum()),
                     "near": float((e[present] <= NEAR_PX).sum()),
                     "wrong": float((e[present] > NEAR_PX).sum()),
                     "missing": float((~present).sum()),
                     "err_near_px": list(e[present & (e <= NEAR_PX)])})
    return rows


def calib_rows(ref: dict, data: dict, label: str, mask: np.ndarray) -> list[dict]:
    gap = np.linalg.norm(gt_pixels(ref, data) - ref["uv_gt"], axis=-1)
    focal = np.abs(data["intr"][..., 0] / ref["intr"][..., 0] - 1.0) * 100.0
    has_cam = data["has_camera"][:, None] & np.ones_like(mask)
    rows = []
    for view in range(len(VIEW_NAMES)):
        m = mask & (ref["view"] == view)[:, None]
        rows.append({"view": VIEW_NAMES[view], "heads": label, "frames": int(m.sum()),
                     "no_camera": float((m & ~has_cam).sum()),
                     "focal": list(focal[m & has_cam]), "gap": list(gap[m & has_cam])})
    return rows


def summarise_tracks(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["err"] = df.pop("err_near_px")
    keep = []
    for view_input, part in df.groupby(["view", "input"], sort=False):
        frames = part["frames"].sum()
        errs = np.concatenate([np.asarray(e, float) for e in part["err"]])
        keep.append({"view": view_input[0], "input": view_input[1], "frames": int(frames),
                     "near_pct": 100 * part["near"].sum() / frames,
                     "wrong_pct": 100 * part["wrong"].sum() / frames,
                     "missing_pct": 100 * part["missing"].sum() / frames,
                     "median_near_px": float(np.median(errs)) if len(errs) else np.nan})
    return pd.DataFrame(keep)


def summarise_calib(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    keep = []
    for (view, heads), part in df.groupby(["view", "heads"], sort=False):
        focal = np.concatenate([np.asarray(v, float) for v in part["focal"]])
        gap = np.concatenate([np.asarray(v, float) for v in part["gap"]])
        frames = part["frames"].sum()
        keep.append({"view": view, "heads": heads, "frames": int(frames),
                     "no_camera_pct": 100 * part["no_camera"].sum() / frames,
                     "focal_err_med_pct": float(np.median(focal)),
                     "gap_med_px": float(np.median(gap)),
                     "gap_p90_px": float(np.quantile(gap, .9))})
    return pd.DataFrame(keep)


# ---------------------------------------------------------------- Main


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tracks, calibs = [], []
    for fold in Config.FOLDS:
        ref = load("gt_src", fold)
        mask = frames_mask(ref)
        for setting, name in SETTINGS.items():
            data = load(name, fold)
            for key in ("group", "view", "a", "n"):
                assert (data[key] == ref[key]).all(), (name, fold, key)
            det = top_detection(ref, extract_dir(setting, fold))
            tracks += track_rows(ref, det, f"YOLO ({setting})", mask)
            tracks += track_rows(ref, data["uv_det"], f"YOLO + tracker ({setting})", mask)
            calibs += calib_rows(ref, data, setting, mask)
        tracks += track_rows(ref, ref["uv_det"], "Benchmark track", mask)
        print("fold", fold, "done", flush=True)
    track_table = summarise_tracks(tracks)
    calib_table = summarise_calib(calibs)
    track_table.to_csv(OUT / "tracking.csv", index=False, float_format="%.2f")
    calib_table.to_csv(OUT / "calibration.csv", index=False, float_format="%.2f")
    text = ["# Inputs on the test flights", "",
            f"Tracking: share of in-flight frames within {NEAR_PX:.0f} px of the projected GT ball "
            "(1920 px wide), wrong (farther), missing; median error of the near frames.", "",
            track_table.round(2).to_string(index=False), "",
            "Calibration vs the dataset camera: median |f/f_ref - 1| (%), and the 3D ground truth "
            "projected with the PnLCalib camera vs with the dataset camera (px).", "",
            calib_table.round(2).to_string(index=False), ""]
    (OUT / "INPUTS.md").write_text("\n".join(text))
    print("\n".join(text))


if __name__ == "__main__":
    main()
