"""Inference speed of one model on a fixed set of decoded images, with no disk or video I/O.

The network stage covers preprocessing, forward pass and peak extraction, with CUDA synchronised.
The solver stage covers keypoint completion and the PnL solve on one CPU thread.
Output: outputs/calibration/speed/<model>.json
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calib_common as Common  # noqa: E402
import calib_infer as Infer  # noqa: E402
import calib_solve as Solve  # noqa: E402
from calib_common import Config  # noqa: E402

SPEED_DATASET = "SN23-test"


def gpu_state() -> str:
    """Other GPU load at measurement time, recorded because it affects the numbers."""
    query = ["nvidia-smi", "--query-gpu=name,memory.used,utilization.gpu",
             "--format=csv,noheader"]
    return subprocess.run(query, capture_output=True, text=True, check=False).stdout.strip()


def time_network(net, images: list[np.ndarray]) -> list[float]:
    """Mean milliseconds per image, for each repeat after warm-up."""
    for image in images[:Config.CALIB_SPEED_WARMUP]:
        Infer.peaks(net, [image])
    repeats = []
    for _ in range(Config.CALIB_SPEED_REPEATS):
        torch.cuda.synchronize()
        begin = time.perf_counter()
        for image in images:
            Infer.peaks(net, [image])
        torch.cuda.synchronize()
        repeats.append((time.perf_counter() - begin) * 1000.0 / len(images))
    return repeats


def time_solver(net, images: list[np.ndarray], name: str, kp_th: float, line_th: float) -> float:
    """Mean milliseconds per image of the solver at this model's chosen thresholds."""
    cv2.setNumThreads(1)
    outputs = [Infer.peaks(net, [image])[name] for image in images]
    wh = Config.CALIB_NET_WH[name]
    begin = time.perf_counter()
    for i, (kp, ln) in enumerate(outputs):
        Solve.solve_frame((i, SPEED_DATASET, kp[0], ln[0], wh, kp_th, line_th))
    return (time.perf_counter() - begin) * 1000.0 / len(images)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=Config.CALIB_MODELS)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    chosen = json.loads((Config.CALIB_ROOT / "chosen_thresholds.json").read_text())[args.model]
    family = args.model if args.model in Config.CALIB_PNL_WEIGHTS else "soccermaster"
    paths = Common.dataset_images(SPEED_DATASET)[:Config.CALIB_SPEED_IMAGES]
    images = [np.array(Image.open(p).convert("RGB")) for p in paths]
    net = Infer.build_net(family, args.device, 1)
    torch.cuda.reset_peak_memory_stats()
    before = gpu_state()
    repeats = time_network(net, images)
    peak = torch.cuda.max_memory_allocated() / 2**20
    solver_ms = time_solver(net, images, args.model, chosen["kp_th"], chosen["line_th"])
    network_ms = float(np.mean(repeats))
    stats = {"model": args.model, "images": len(images), "network_ms": network_ms,
             "network_ms_repeats": repeats, "solver_ms": solver_ms,
             "total_ms": network_ms + solver_ms, "fps": 1000.0 / (network_ms + solver_ms),
             "network_fps": 1000.0 / network_ms, "peak_gpu_mb": peak, "gpu_before": before,
             "torch": torch.__version__, "precision": "float32", "batch_size": 1}
    out = Config.CALIB_ROOT / "speed" / f"{args.model}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1), flush=True)


if __name__ == "__main__":
    main()
