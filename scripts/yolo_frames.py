"""YOLO on video frames: tiled detection for 4K views and sequential frame decoding.

Library for extract_video.py and the head-data scripts. Frames larger than the tile are split into
overlapping tiles (2 x 2 on the 4K static views) and the boxes merged back to full-frame pixels.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TILE_W = 1920
TILE_H = 1080
TILE_OVERLAP_PX = 64
NMS_DIST_PX = 8.0
SEEK_GAP_FRAMES = 250  # seek when the next wanted frame is further ahead than this


def tile_origins(length: int, tile: int, overlap: int) -> list[int]:
    """Start offsets covering [0, length) with tiles of size `tile` overlapping by `overlap`."""
    if length <= tile:
        return [0]
    n = int(np.ceil((length - overlap) / (tile - overlap)))
    return np.linspace(0, length - tile, n).round().astype(int).tolist()


def detect_frame(model: YOLO, frame: np.ndarray, imgsz: int, conf: float,
                 tile: tuple[int, int] = (TILE_W, TILE_H), half: bool = False) -> np.ndarray:
    """Return (K, 5) boxes [x0, y0, x1, y1, score] in full-frame pixels, tile duplicates merged."""
    h, w = frame.shape[:2]
    tile_w, tile_h = tile
    tiles = [(x, y) for y in tile_origins(h, tile_h, TILE_OVERLAP_PX)
             for x in tile_origins(w, tile_w, TILE_OVERLAP_PX)]
    crops = [frame[y:y + tile_h, x:x + tile_w] for x, y in tiles]
    results = model.predict(crops, imgsz=imgsz, conf=conf, verbose=False, half=half)

    boxes = []
    for (x, y), res in zip(tiles, results):
        b = res.boxes
        if len(b) == 0:
            continue
        xyxy = b.xyxy.cpu().numpy() + np.array([x, y, x, y])
        boxes.append(np.c_[xyxy, b.conf.cpu().numpy()])
    if not boxes:
        return np.zeros((0, 5))

    # Merge duplicates from overlapping tiles: greedy by score on centre distance.
    boxes = np.concatenate(boxes)
    boxes = boxes[np.argsort(-boxes[:, 4])]
    centres = 0.5 * (boxes[:, :2] + boxes[:, 2:4])
    keep = []
    for i in range(len(boxes)):
        if all(np.linalg.norm(centres[i] - centres[j]) > NMS_DIST_PX for j in keep):
            keep.append(i)
    return boxes[keep]


def iter_frames(cap: cv2.VideoCapture, frames: np.ndarray):
    """Yield (index, image) for the wanted frames, grabbing short gaps and seeking long ones."""
    pos = 0
    for idx in frames:
        if idx - pos > SEEK_GAP_FRAMES:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            pos = int(idx)
        while pos < idx:
            if not cap.grab():
                return
            pos += 1
        ok, frame = cap.read()
        if not ok:
            return
        pos += 1
        yield int(idx), frame
