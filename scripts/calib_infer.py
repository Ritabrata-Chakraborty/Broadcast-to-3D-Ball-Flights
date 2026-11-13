"""Run PnLCalib or SoccerMaster over the comparison datasets and store raw heatmap peaks.

Peaks are stored with their scores, so thresholds and the solver are applied offline (calib_solve).
Output: outputs/calibration/raw/<model>/<dataset>/*.npz (resumable), plus <model>_run.json.
"""
import argparse
import json
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import yaml
from PIL import Image
from torchvision.transforms import v2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calib_common as Common  # noqa: E402
from calib_common import Config  # noqa: E402

PNL_ROOT = Config.THIRD_PARTY_ROOT / "PnLCalib"
sys.path.insert(0, str(PNL_ROOT))
from utils.utils_heatmap import (  # noqa: E402
    get_keypoints_from_heatmap_batch_maxpool,
    get_keypoints_from_heatmap_batch_maxpool_l,
)

PART_IMAGES = 200
SN_MEAN = 0.5
TEMPORAL_START_LAYER = 16
BACKBONE_HIDDEN_DIM = 1024
SM_HEAD_CHANNELS = (58, 24)


# ---------------------------------------------------------------- Networks


class PnLCalibNet:
    """HRNet pair at 960x540, one frame at a time: batching and AMP change PnLCalib's outputs."""

    batch_size = 1

    def __init__(self, model: str, device: str, weights_kp: str, weights_ln: str) -> None:
        from model.cls_hrnet import get_cls_net
        from model.cls_hrnet_l import get_cls_net as get_cls_net_l
        cfg = yaml.safe_load((PNL_ROOT / "config" / "hrnetv2_w48.yaml").read_text())
        cfg_l = yaml.safe_load((PNL_ROOT / "config" / "hrnetv2_w48_l.yaml").read_text())
        weights = Config.WEIGHTS_ROOT / "pnlcalib"
        self.kp_net = get_cls_net(cfg)
        self.kp_net.load_state_dict(torch.load(weights / weights_kp, map_location=device))
        self.ln_net = get_cls_net_l(cfg_l)
        self.ln_net.load_state_dict(torch.load(weights / weights_ln, map_location=device))
        self.kp_net.to(device).eval()
        self.ln_net.to(device).eval()
        self.device = device
        self.peak_sets = {model: (1, 1)}

    def heatmaps(self, images: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
        tensor = F.to_tensor(images[0]).float().unsqueeze(0)
        tensor = T.Resize((540, 960))(tensor).to(self.device)
        return self.kp_net(tensor), self.ln_net(tensor)


class SoccerMasterNet:
    """Spatial layers plus keypoint and line heads, per frame.

    The heads read the features from before the temporal layers (`local_features_early` in
    models/soccer_master.py), so those layers, and the 30-frame clip they need, cannot affect them.
    """

    def __init__(self, device: str, batch_size: int) -> None:
        sys.path.insert(0, str(Config.SOCCERMASTER_ROOT))
        from models.keypoints_detection import KeypointsHead
        from models.lines_detection import LinesHead
        from models.soccer_master import VisionBackbone
        self.batch_size = batch_size
        self.device = device
        self.peak_sets = Config.CALIB_SM_PEAKS
        backbone = VisionBackbone(str(Config.SIGLIP2_WEIGHTS), Config.CALIB_CLIP_FRAMES,
                                  TEMPORAL_START_LAYER)
        state = torch.load(Config.SOCCERMASTER_WEIGHTS / "backbone.pt", map_location="cpu")
        result = backbone.load_state_dict(state, strict=False)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(f"backbone mismatch: {result}")
        self.backbone = backbone.to(device).eval()
        self.kp_head = KeypointsHead(BACKBONE_HIDDEN_DIM, SM_HEAD_CHANNELS[0])
        self.kp_head.load_state_dict(self.head_state("KeypointsDetection.pt", "keypoints_head."))
        self.ln_head = LinesHead(BACKBONE_HIDDEN_DIM, SM_HEAD_CHANNELS[1])
        self.ln_head.load_state_dict(self.head_state("LinesDetection.pt", "lines_head."))
        self.kp_head.to(device).eval()
        self.ln_head.to(device).eval()

    @staticmethod
    def head_state(name: str, prefix: str) -> dict:
        state = torch.load(Config.SOCCERMASTER_WEIGHTS / name, map_location="cpu",
                           weights_only=True)
        return {k.removeprefix(prefix): v for k, v in state.items()}

    def preprocess(self, images: list[np.ndarray]) -> torch.Tensor:
        """Same steps as the repo's test transforms: to float, bicubic 512x512, normalize."""
        out = []
        for image in images:
            tensor = v2.functional.to_dtype(v2.functional.to_image(image), torch.float32,
                                            scale=True)
            tensor = v2.functional.resize(tensor, [512, 512],
                                          interpolation=v2.InterpolationMode.BICUBIC)
            out.append(F.normalize(tensor, [SN_MEAN] * 3, [SN_MEAN] * 3))
        return torch.stack(out).to(self.device)

    def features(self, batch: torch.Tensor) -> torch.Tensor:
        """Features after the spatial layers, as (N, D, 32, 32)."""
        x = self.backbone.vision_model_embedding(batch)
        for idx in range(self.backbone.temporal_start_layer):
            x = self.backbone.encoder_blocks[idx](x, batch.shape[0], 1)
        side = int(np.sqrt(x.shape[1]))
        return x.permute(0, 2, 1).reshape(x.shape[0], x.shape[2], side, side)

    def heatmaps(self, images: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.features(self.preprocess(images))
        return self.kp_head(features), self.ln_head(features)


def build_net(model: str, device: str, batch_size: int):
    if model in Config.CALIB_PNL_WEIGHTS:
        return PnLCalibNet(model, device, *Config.CALIB_PNL_WEIGHTS[model])
    return SoccerMasterNet(device, batch_size)


def peaks(net, images: list[np.ndarray]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per peak set, top keypoint per channel and two line ends in network pixels."""
    with torch.no_grad():
        kp_map, ln_map = net.heatmaps(images)
    out = {}
    for name, (kp_dist, ln_dist) in net.peak_sets.items():
        kp = get_keypoints_from_heatmap_batch_maxpool(kp_map[:, :-1],
                                                      min_keypoint_pixel_distance=kp_dist)
        ln = get_keypoints_from_heatmap_batch_maxpool_l(ln_map[:, :-1],
                                                        min_keypoint_pixel_distance=ln_dist)
        out[name] = (kp.squeeze(2).cpu().numpy(), ln.cpu().numpy())
    return out


# ---------------------------------------------------------------- Data sources


def read_frames(cap: cv2.VideoCapture, frames: np.ndarray, is_mirrored: bool) -> list[np.ndarray]:
    """RGB frames, seeking only on a jump; raises at a video end, so nothing is silently skipped."""
    images = []
    position = None
    for frame in frames:
        if position != frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame))
        ok, image = cap.read()
        if not ok:
            raise RuntimeError(f"decode failed at frame {frame}")
        position = frame + 1
        image = image[:, ::-1] if is_mirrored else image
        images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return images


def collect(net, images: list[np.ndarray], store: dict) -> None:
    """Run the network in batches and append peaks per peak set."""
    for start in range(0, len(images), net.batch_size):
        for name, (kp, ln) in peaks(net, images[start:start + net.batch_size]).items():
            store.setdefault(name, ([], []))
            store[name][0].append(kp)
            store[name][1].append(ln)


def save_store(net, dataset: str, stem: str, ids: list, sizes: list, store: dict) -> None:
    for name in net.peak_sets:
        Common.save_part(Common.raw_dir(name, dataset) / f"{stem}.npz", ids, sizes,
                         np.concatenate(store[name][0]), np.concatenate(store[name][1]))


def is_stored(net, dataset: str, stem: str) -> bool:
    return all((Common.raw_dir(n, dataset) / f"{stem}.npz").exists() for n in net.peak_sets)


def run_images(net, dataset: str, limit: int) -> int:
    """All images of one image dataset, in parts of PART_IMAGES."""
    images = Common.dataset_images(dataset)
    images = images[:limit] if limit else images
    done = 0
    for part, first in enumerate(range(0, len(images), PART_IMAGES)):
        stem = f"part_{part:05d}"
        chunk = images[first:first + PART_IMAGES]
        if is_stored(net, dataset, stem):
            continue
        store, sizes = {}, []
        for start in range(0, len(chunk), net.batch_size):
            rgb = [np.asarray(Image.open(p).convert("RGB"))
                   for p in chunk[start:start + net.batch_size]]
            collect(net, rgb, store)
            sizes += [(im.shape[1], im.shape[0]) for im in rgb]
        ids = [Common.image_id(dataset, p) for p in chunk]
        save_store(net, dataset, stem, ids, sizes, store)
        done += len(chunk)
        print(f"  {dataset}: {first + len(chunk)}/{len(images)}", flush=True)
    return done


def run_video_unit(net, dataset: str, unit: str, table: pd.DataFrame) -> int:
    """All planned frames of one video unit, one file per unit."""
    stem = unit.replace("/", "__")
    if is_stored(net, dataset, stem):
        return 0
    source, is_mirrored = Common.video_source(unit)
    cap = cv2.VideoCapture(str(source))
    ids, sizes, store = [], [], {}
    for _, clip in table.groupby("clip", sort=True):
        frames = clip["frame"].to_numpy()
        rgb = read_frames(cap, frames, is_mirrored)
        collect(net, rgb, store)
        sizes += [(im.shape[1], im.shape[0]) for im in rgb]
        ids += frames.tolist()
    cap.release()
    save_store(net, dataset, stem, ids, sizes, store)
    print(f"  {unit}: {len(ids)} frames", flush=True)
    return len(ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                        choices=[*Config.CALIB_PNL_WEIGHTS, "soccermaster"])
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--limit", type=int, default=0, help="first N images, for a smoke test")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--kp-head", type=Path, default=None,
                        help="fine-tuned PnLCalib MV keypoint state dict (train_pnlcalib_head.py)")
    parser.add_argument("--name", default=None, help="stored model name for --kp-head")
    args = parser.parse_args()

    torch.manual_seed(Config.CALIB_SEED)
    net = build_net(args.model, args.device, 1)
    if args.kp_head is not None:
        if args.model != "pnlcalib_mv" or not (args.name or "").startswith("pnlcalib_mv_"):
            parser.error("--kp-head needs --model pnlcalib_mv and --name pnlcalib_mv_<tag>")
        net.kp_net.load_state_dict(torch.load(args.kp_head, map_location=args.device))
        net.kp_net.eval()
        net.peak_sets = {args.name: (1, 1)}
        args.model = args.name
    plan = pd.read_csv(Config.CALIB_ROOT / "video_plan.csv")
    for dataset in args.datasets:
        begin, count = time.time(), 0
        if dataset in Common.IMAGE_DATASETS:
            count = run_images(net, dataset, args.limit)
        else:
            table = plan.loc[plan["dataset"] == dataset]
            for unit, group in table.groupby("unit", sort=True):
                count += run_video_unit(net, dataset, unit, group)
        stats = {"model": args.model, "dataset": dataset, "frames": count,
                 "seconds": time.time() - begin, "seed": Config.CALIB_SEED, "batch_size": 1,
                 "torch": torch.__version__, "python": platform.python_version(),
                 "peak_gpu_mb": torch.cuda.max_memory_allocated() / 2**20}
        out = Common.raw_dir(args.model, dataset) / f"run_{int(begin)}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(stats, indent=1))
        print(f"[ok] {args.model} {dataset}: {count} frames", flush=True)


if __name__ == "__main__":
    main()
