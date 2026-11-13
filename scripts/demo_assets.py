"""Demo clips for the project page: video, inputs and every method's 3D path for the demo flights.

Flights: the ten six-view test flights of build_demo_sets.py (demo_flights.json). Each view of each
flight gives one clip (the flight plus PAD frames either side):
  <flight>_<view>.mp4    960x540 H.264 video (LP footage: never commit; the page encrypts it)
  <flight>_<view>.json   dataset camera per frame, ground truth, and per input setting (gt, real,
                         heads) the tracked ball, YOLO's boxes and PnLCalib's pitch lines, plus
                         each method's path for the nine (start, end) contact-height pairs
and manifest.json. Methods: LM-fitg, Grad et al., ArcPrior Base and (video inputs) ArcPrior ft,
each ArcPrior with every prior; "frozen" marks the fold's prior chosen on validation.
Output: site/assets/demo/. Run in the ball3d env: scripts/py scripts/demo_assets.py
"""
import argparse
import gzip
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as Config  # noqa: E402

VIEW_NAMES = ("broadcast", "camera01", "camera02", "camera03", "camera04", "camera05")
WIDTH, HEIGHT = 960, 540
PAD = 12
LEVELS = ("true", "off", "none")
SETTINGS = ("gt", "real", "heads")
PRIORS = {"ap:none:0:obj": "none", "ap:gauss:1:obj": "gauss",
          "ap:subspace:1:obj": "subspace1", "ap:subspace:3:obj": "subspace3"}
# Freeze keys of the prior chosen on validation, per setting and network.
FROZEN = {("gt", "base"): "base_gt_full", ("real", "base"): "base_real_full",
          ("heads", "base"): "base_heads_full", ("real", "ft"): "real_ft_real_full",
          ("heads", "ft"): "heads_ft_heads_full"}
PITCH_LENGTH_M, PITCH_WIDTH_M = 105.0, 68.0


def camera_matrix(rot: np.ndarray, trans: np.ndarray, intr: np.ndarray, image_h: float) -> list:
    """3x4 matrix from world metres (z up) to 960x540 pixels, in the dataset's convention."""
    scale = HEIGHT / image_h
    k = np.array([[intr[0] * scale, 0, intr[2] * scale], [0, intr[1] * scale, intr[3] * scale],
                  [0, 0, 1]])
    p = k @ np.hstack([rot @ np.diag([1.0, 1.0, -1.0]), trans[:, None]])
    return [round(float(v), 6) for v in p.ravel()]


def pnl_polylines(cam: dict) -> list:
    """PnLCalib's pitch lines for one of its cameras, in 960x540 pixels."""
    sys.path.insert(0, str(Config.THIRD_PARTY_ROOT / "PnLCalib" / "sn_calibration" / "src"))
    from evaluate_camera import get_polylines

    lines = []
    for points in get_polylines(cam, WIDTH, HEIGHT, sampling_factor=0.9).values():
        pts = [[round(p["x"], 1), round(p["y"], 1)] for p in points
               if -50 < p["x"] < WIDTH + 50 and -50 < p["y"] < HEIGHT + 50]
        if len(pts) > 1:
            lines.append(pts)
    return lines


def load_extract(source: Path, view: str, half: str) -> tuple[pd.DataFrame, dict, float]:
    unit = f"{view}_{half}"
    boxes = pd.read_csv(source / f"{unit}.det.csv.gz")
    cams = {}
    with gzip.open(source / f"{unit}.cam.jsonl.gz", "rt") as handle:
        for row in map(json.loads, handle):
            if row.get("ok"):
                cams[int(row["id"])] = row["cam"]
    width = json.loads((source / f"{unit}.json").read_text()).get("width", 1920)
    return boxes, cams, float(width)


def video_inputs(source: Path, view: str, half: str, frames: np.ndarray) -> dict:
    """YOLO's six best boxes per clip frame and PnLCalib's lines (per frame on broadcast)."""
    boxes, cams, width = load_extract(source, view, half)
    scale = WIDTH / width
    boxes = boxes[boxes["file_name"].isin(frames)]
    yolo = []
    for frame in frames:
        rows = boxes[boxes["file_name"] == frame].sort_values("score", ascending=False).head(6)
        yolo.append([[round(v * scale, 1) for v in (r.x0, r.y0, r.x1, r.y1)] + [round(r.score, 2)]
                     for r in rows.itertuples()])
    known = np.array(sorted(cams))
    if not len(known):
        return {"boxes": yolo, "lines": {"static": True, "lines": []}}
    if view == "broadcast":
        lines = [pnl_polylines(cams[int(known[np.argmin(np.abs(known - f))])]) for f in frames]
        return {"boxes": yolo, "lines": {"static": False, "lines": lines}}
    middle = frames[len(frames) // 2]
    cam = cams[int(known[np.argmin(np.abs(known - middle))])]
    return {"boxes": yolo, "lines": {"static": True, "lines": pnl_polylines(cam)}}


def write_video(view: str, half: str, start: int, count: int, target: Path) -> int:
    """H.264 clip (CRF 26, yuv420p, fast start) through the ffmpeg binary of imageio-ffmpeg."""
    import imageio_ffmpeg

    capture = cv2.VideoCapture(str(Config.DATA_ROOT / Config.LP / view / half / "input.mkv"))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start)
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo",
               "-pix_fmt", "bgr24", "-s", f"{WIDTH}x{HEIGHT}", "-r", str(Config.FPS), "-i", "-",
               "-c:v", "libx264", "-preset", "slow", "-crf", "26", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(target)]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    written = 0
    for _ in range(count):
        is_read, image = capture.read()
        if not is_read:
            break
        encoder.stdin.write(cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
                            .tobytes())
        written += 1
    capture.release()
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError(f"ffmpeg failed on {target}")
    return written


def rounded(path: np.ndarray, digits: int = 2) -> list:
    return [None if not np.isfinite(p).all() else [round(float(v), digits) for v in p]
            for p in path]


def load_predictions(pred_root: Path, fold: str) -> dict:
    """Frames and flight rows per (setting, network) for one fold; method names as in predict.py."""
    out = {}
    for setting in SETTINGS:
        for tag in ("base", "ft", "grad"):
            folder = pred_root / f"demo_{setting}" / tag
            if not (folder / f"fold_{fold}_test_flights.csv.gz").exists():
                continue
            flights = pd.read_csv(folder / f"fold_{fold}_test_flights.csv.gz")
            frames = pd.read_csv(folder / f"fold_{fold}_test_frames.csv.gz")
            out[(setting, tag)] = (flights[flights["share"] == 1.0], frames)
    return out


def method_paths(preds: dict, setting: str, view: int, flight: int,
                 frames: np.ndarray) -> dict:
    """{method key: [9 entries of {path, rmse, coverage} or None]} for one view-flight."""
    out = {}
    for (name, tag), (flights, table) in preds.items():
        if name != setting:
            continue
        for method in flights["method"].unique():
            if tag == "grad":
                key = "grad"
            elif method == "lm_fitg":
                key = "lm_fitg"
            else:
                key = f"{tag}:{PRIORS[method]}"
            entries = []
            for pair in range(len(LEVELS) ** 2):
                group = flight * 10 + pair
                row = flights[(flights["method"] == method) & (flights["view"] == view)
                              & (flights["group"] == group)]
                if row.empty:
                    entries.append(None)
                    continue
                rows = table[(table["method"] == method) & (table["view"] == view)
                             & (table["group"] == group)].set_index("file_name")
                path = rows.reindex(frames)[["x_predicted", "y_predicted", "z_predicted"]]
                path = path.to_numpy(float) / Config.DM_PER_M
                entries.append({"path": rounded(path),
                                "rmse": round(float(row["rmse_cap"].iloc[0]), 2),
                                "coverage": round(float(row["coverage"].iloc[0]), 2)})
            out[key] = entries
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Config.EVAL_DATA)
    parser.add_argument("--pred-root", type=Path, default=Config.EVAL_ROOT / "pred")
    parser.add_argument("--freeze", type=Path, default=Config.EVAL_ROOT / "freeze.json")
    parser.add_argument("--freeze-keys", nargs="*", default=[],
                        help="renames of FROZEN keys, as new=old (older freeze files)")
    parser.add_argument("--published", type=Path, default=Config.EXTRACT_DIR / "published")
    parser.add_argument("--heads-root", type=Path, default=Config.EXTRACT_DIR,
                        help="holds heads_half_1 and heads_half_2 (link_heads.py)")
    parser.add_argument("--out", type=Path, default=Config.REPO_ROOT / "site" / "assets" / "demo")
    parser.add_argument("--skip-video", action="store_true", help="keep existing mp4 files")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    renames = dict(pair.split("=") for pair in args.freeze_keys)
    choices = json.loads(args.freeze.read_text())["choices"]
    flights = json.loads((args.data_root / "demo_flights.json").read_text())

    manifest = []
    for chosen in flights:
        flight, fold, half = chosen["flight"], chosen["fold"], Config.HALVES[chosen["half"]]
        heads_half = Config.HALVES[1 - chosen["half"]]     # heads trained on the other half
        data = {s: np.load(args.data_root / f"demo_{s}" / f"fold_{fold}" / "test.npz")
                for s in SETTINGS}
        preds = load_predictions(args.pred_root, fold)
        frozen = {f"{s}:{tag}": PRIORS[choices[renames.get(key, key)][fold]]
                  for (s, tag), key in FROZEN.items()}
        for view, view_name in enumerate(VIEW_NAMES):
            z = data["gt"]
            rows = np.where((z["flight"] == flight) & (z["view"] == view) & (z["pair"] == 0))[0]
            if not len(rows):
                continue
            i = rows[0]
            a, n = int(z["a"][i]), int(z["n"][i])
            start, count = max(0, a - PAD), n + 2 * PAD
            name = f"{flight}_{view_name}"
            video = args.out / f"{name}.mp4"
            if not (args.skip_video and video.exists()):
                count = write_video(view_name, half, start, count, video)
            clip_frames = np.arange(start, start + count)
            flight_frames = np.arange(a, a + n)
            gt = z["xyz"][i, :n]
            nearest = np.clip(clip_frames - a, 0, n - 1)
            cams = [camera_matrix(z["rot"][i, k], z["trans"][i, k], z["intr"][i, k],
                                  float(z["image_h"][i])) for k in range(n)]
            inputs, methods = {}, {}
            for setting in SETTINGS:
                zs = data[setting]
                j = np.where((zs["flight"] == flight) & (zs["view"] == view)
                             & (zs["pair"] == 0))[0][0]
                # Tracks are stored at 1920 px wide; the clip is 960 wide.
                track = zs["uv_det"][j, :n] * WIDTH / Config.IMAGE_WIDTH_PX
                inputs[setting] = {"track": rounded(track, 1)}
                if setting != "gt":
                    source = (args.published if setting == "real"
                              else args.heads_root / f"heads_{heads_half}")
                    inputs[setting].update(video_inputs(source, view_name, half, clip_frames))
                inputs[setting]["heights"] = {
                    end: [round(float(zs[f"h_{end}"][k]), 2) if zs[f"{end}_given"][k] else None
                          for k in np.where((zs["flight"] == flight) & (zs["view"] == view))[0]]
                    for end in ("start", "end")}
                methods[setting] = method_paths(preds, setting, view, flight, flight_frames)
            apex = float(np.nanmax(gt[:, 2]))
            payload = {
                "flight": flight, "percentile": chosen["percentile"], "view": view_name,
                "half": chosen["half"] + 1, "fps": Config.FPS, "width": WIDTH, "height": HEIGHT,
                "start": int(clip_frames[0]), "count": int(count), "flight_start": a,
                "flight_n": n, "apex": round(apex, 2),
                "z_contact": [round(float(z["z_start"][i]), 2), round(float(z["z_end"][i]), 2)],
                "levels": LEVELS, "camera": [cams[k] for k in nearest], "gt": rounded(gt),
                "inputs": inputs, "methods": methods, "frozen": frozen,
                "pitch": {"centre": [c / Config.DM_PER_M for c in Config.LP_PITCH_CENTRE_DM],
                          "length": PITCH_LENGTH_M, "width": PITCH_WIDTH_M},
            }
            (args.out / f"{name}.json").write_text(json.dumps(payload, separators=(",", ":")))
            manifest.append({"name": name, "flight": flight, "percentile": chosen["percentile"],
                             "view": view_name, "half": chosen["half"] + 1,
                             "apex": round(apex, 1), "frames": int(count)})
            print(name, count, "frames")
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
