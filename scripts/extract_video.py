"""Video inputs on LP flight frames: SoccerNet-v3D YOLO11 ball boxes and PnLCalib MV cameras.

One decode pass per view and half. YOLO runs on every frame of every flight (2 x 2 tiles on the 4K
static views); PnLCalib MV runs on every broadcast flight frame and on STATIC_CALIB_FRAMES frames
per static camera, which does not move. The calibration thresholds were chosen on SN23-valid by the
calibration study. Output: outputs/extract/<name>/<view>_<half>.{det.csv.gz,cam.jsonl.gz,json};
existing units are skipped. --yolo and --kp-head run fine-tuned heads; --no-calib runs YOLO only
(the size-optimised weights for SNv3D-size).
Run in the snv3d env (GPU): scripts/py --env snv3d scripts/extract_video.py
"""
import argparse
import gzip
import json
import platform
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as Config
from calib_infer import PnLCalibNet, peaks
from calib_solve import init_worker, solve_frame
from yolo_frames import detect_frame, iter_frames

WEIGHTS = "yolo-sn-ball.pt"
IMGSZ = 1280
CONF = 0.05
TILE_4K = (1984, 1112)   # 2 x 2 tiles at 4K, about the 1920-wide scale the detector was trained at
TILE_HD = (1920, 1080)
PNL_MODEL = "pnlcalib_mv"
STATIC_CALIB_FRAMES = 60
VIEWS = ("broadcast", *[f"camera{c:02d}" for c in range(1, 7)])
OUT = Config.EXTRACT_DIR / "published"


def flight_windows() -> pd.DataFrame:
    """Every frozen view-flight of both folds: view, half, first frame and length."""
    rows = []
    for fold in Config.HALF_FOLDS:
        for split in ("train", "val", "test"):
            with np.load(Config.FLIGHTS_DIR / f"fold_{fold}" / f"{split}.npz") as archive:
                rows.append(pd.DataFrame({"view": archive["view"], "half": archive["half"],
                                          "a": archive["a"], "n": archive["n"]}))
    return pd.concat(rows).drop_duplicates().reset_index(drop=True)


def unit_frames(windows: pd.DataFrame, view: int, half: int) -> np.ndarray:
    """Sorted unique video frames covered by one unit's flights."""
    chosen = windows[(windows["view"] == view) & (windows["half"] == half)]
    spans = [np.arange(a, a + n) for a, n in zip(chosen["a"], chosen["n"])]
    return np.unique(np.concatenate(spans))


def thresholds() -> tuple[float, float]:
    """PnLCalib MV keypoint and line thresholds selected by the calibration study."""
    chosen = json.loads((Config.CALIB_ROOT / "chosen_thresholds.json").read_text())[PNL_MODEL]
    return float(chosen["kp_th"]), float(chosen["line_th"])


def run_unit(view: int, half: int, frames: np.ndarray, yolo, pnl: PnLCalibNet, pool,
             kp_th: float, line_th: float, out: Path = OUT) -> None:
    """Detect and calibrate one unit's flight frames."""
    name = f"{VIEWS[view]}_{Config.HALVES[half]}"
    det_path, cam_path = out / f"{name}.det.csv.gz", out / f"{name}.cam.jsonl.gz"
    if det_path.exists() and cam_path.exists():
        print(f"[skip] {name}", flush=True)
        return
    unit = f"{Config.LP}/{VIEWS[view]}/{Config.HALVES[half]}"
    cap = cv2.VideoCapture(str(Config.DATA_ROOT / unit / "input.mkv"))
    if not cap.isOpened():
        raise FileNotFoundError(unit)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    tile = TILE_4K if width > 2000 else TILE_HD
    is_static = view != 0
    calib = frames
    if is_static:
        pick = np.linspace(0, len(frames) - 1, STATIC_CALIB_FRAMES).round().astype(int)
        calib = frames[np.unique(pick)]
    if pnl is None:
        calib = frames[:0]
    calib_set = set(calib.tolist())

    boxes, pending = [], []
    began = time.time()
    n_done = 0
    for index, image in iter_frames(cap, frames):
        for k, (x0, y0, x1, y1, score) in enumerate(detect_frame(yolo, image, IMGSZ, CONF,
                                                                 tile=tile, half=True)):
            boxes.append((index, x0, y0, x1, y1, k, score))
        if index in calib_set:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            kp, ln = peaks(pnl, [rgb])[PNL_MODEL]
            job = (index, unit, kp[0], ln[0], Config.CALIB_NET_WH[PNL_MODEL], kp_th, line_th)
            pending.append(pool.apply_async(solve_frame, (job,)))
        n_done += 1
        if n_done % 1000 == 0:
            print(f"  {name}: {n_done} / {len(frames)} frames, "
                  f"{n_done / (time.time() - began):.1f} fps", flush=True)
    cap.release()
    if n_done != len(frames):
        raise RuntimeError(f"{name}: decoded {n_done} of {len(frames)} frames")

    columns = ["file_name", "x0", "y0", "x1", "y1", "detection_id", "score"]
    pd.DataFrame(boxes, columns=columns).to_csv(det_path, index=False)
    with gzip.open(cam_path, "wt") as handle:
        for result in pending:
            handle.write(json.dumps(result.get(), default=float, separators=(",", ":")) + "\n")
    meta = {"unit": unit, "frames": int(len(frames)), "calibrated_frames": len(calib),
            "boxes": len(boxes), "width": width, "tile": tile,
            "seconds": round(time.time() - began)}
    (out / f"{name}.json").write_text(json.dumps(meta, indent=1))
    print(f"[ok] {meta}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--views", nargs="+", type=int, default=list(range(len(VIEWS))))
    parser.add_argument("--workers", type=int, default=6, help="CPU processes for the solver")
    parser.add_argument("--halves", nargs="+", type=int, default=list(range(len(Config.HALVES))))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--yolo", type=Path, default=Config.WEIGHTS_ROOT / "snv3d" / WEIGHTS)
    parser.add_argument("--kp-head", type=Path, default=None,
                        help="state dict of a head-tuned PnLCalib MV keypoint network")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--no-calib", action="store_true", help="YOLO boxes only, no PnLCalib")
    parser.add_argument("--windows", type=Path, default=None,
                        help="CSV of view, half, a, n: only these flight windows")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    kp_th, line_th = thresholds()
    windows = flight_windows()
    if args.windows is not None:
        windows = pd.read_csv(args.windows)[["view", "half", "a", "n"]]
    tag = "_".join(str(v) for v in args.views)
    (args.out / f"config_views_{tag}.json").write_text(json.dumps(
        {"yolo": str(args.yolo), "kp_head": str(args.kp_head) if args.kp_head else None,
         "no_calib": args.no_calib, "views": args.views, "halves": args.halves,
         "windows": str(args.windows) if args.windows else None,
         "imgsz": IMGSZ, "conf": CONF, "tile_4k": TILE_4K, "pnlcalib": PNL_MODEL,
         "kp_th": kp_th, "line_th": line_th, "static_calib_frames": STATIC_CALIB_FRAMES,
         "solve_wh": Config.CALIB_EVAL_WH, "view_flights": len(windows),
         "python": platform.python_version(), "torch": str(torch.__version__),
         "opencv": cv2.__version__}, indent=1))
    # The solver pool forks before CUDA starts, which forked workers cannot share.
    with get_context("fork").Pool(args.workers, init_worker) as pool:
        from ultralytics import YOLO
        yolo = YOLO(str(args.yolo))
        torch.manual_seed(Config.CALIB_SEED)
        pnl = None
        if not args.no_calib:
            pnl = PnLCalibNet(PNL_MODEL, args.device, *Config.CALIB_PNL_WEIGHTS[PNL_MODEL])
        if args.kp_head is not None and pnl is not None:
            pnl.kp_net.load_state_dict(torch.load(args.kp_head, map_location=args.device))
            pnl.kp_net.eval()
        for view in args.views:
            for half in args.halves:
                frames = unit_frames(windows, view, half)
                run_unit(view, half, frames, yolo, pnl, pool, kp_th, line_th, args.out)


if __name__ == "__main__":
    main()
