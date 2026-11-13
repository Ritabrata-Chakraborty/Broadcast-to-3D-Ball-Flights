"""Fine-tune only the final head of PnLCalib MV's keypoint network on one LP half.

The HRNet backbone stays frozen in eval mode; `head` (1x1 conv, batch norm, final conv, softmax)
trains with PnLCalib's own target and loss: a Gaussian (sigma 2) per keypoint at half the 960x540
input, a background channel, per-pixel MSE. Targets are the 57 pitch points projected by the dataset
camera (head_data_broadcast.py). Preprocessing matches inference: float32, CPU resize, batch-free
statistics. Validation reports keypoint pixel error on the fold's validation-flight frames.
Output: outputs/heads/broadcast/pnlcalib/<fold>/ (head-tuned MV_kp state, log.csv, config.json).
Run in the snv3d env: scripts/py --env snv3d scripts/train_pnlcalib_head.py --fold a
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
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as Config

PNL_ROOT = Config.THIRD_PARTY_ROOT / "PnLCalib"
sys.path.insert(0, str(PNL_ROOT))
from model.cls_hrnet import get_cls_net  # noqa: E402
from utils.utils_heatmap import get_keypoints_from_heatmap_batch_maxpool  # noqa: E402

HEADS = Config.HEADS_DIR / "broadcast" / "data"
OUT = Config.HEADS_DIR / "broadcast" / "pnlcalib"
NET_W, NET_H = 960, 540
DOWN = 2
SIGMA = 2.0
N_KP = 57
SEED = 0


def heatmap_target(uv: np.ndarray, width: float) -> np.ndarray:
    """PnLCalib's target: Gaussian per in-frame keypoint at half network size, then background."""
    h, w = NET_H // DOWN, NET_W // DOWN
    scale = NET_W / width / DOWN
    ys, xs = np.mgrid[0:h, 0:w]
    target = np.zeros((N_KP + 1, h, w), np.float32)
    for k, (u, v) in enumerate(uv):
        if np.isfinite(u):
            # PnLCalib truncates the resized keypoint to an integer pixel.
            px, py = int(u * scale), int(v * scale)
            target[k] = np.exp(-((xs - px) ** 2 + (ys - py) ** 2) / (2 * SIGMA ** 2))
    target[N_KP] = 1.0 - target[:N_KP].sum(axis=0)
    return np.clip(target, 0.0, 1.0)


class HeadData(Dataset):
    def __init__(self, folder: Path, split: str, limit: int | None) -> None:
        with np.load(folder / f"keypoints_{split}.npz") as z:
            self.frames, self.uv, self.width = z["frames"], z["uv"], float(z["width"])
        if limit is not None and len(self.frames) > limit:
            pick = np.random.default_rng(SEED).choice(len(self.frames), limit, replace=False)
            pick.sort()
            self.frames, self.uv = self.frames[pick], self.uv[pick]
        self.images = folder / "images" / split

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, i: int):
        name = self.frames[i]
        name = name if isinstance(name, str) else f"{int(name):06d}"
        image = cv2.imread(str(self.images / f"{name}.jpg"))
        tensor = F.to_tensor(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).float()
        tensor = T.Resize((NET_H, NET_W))(tensor)
        return tensor, torch.from_numpy(heatmap_target(self.uv[i], self.width)), \
            torch.from_numpy(self.uv[i] * NET_W / self.width)


def load_net(device: str) -> torch.nn.Module:
    cfg = yaml.safe_load((PNL_ROOT / "config" / "hrnetv2_w48.yaml").read_text())
    net = get_cls_net(cfg)
    net.load_state_dict(torch.load(Config.WEIGHTS_ROOT / "pnlcalib" / "MV_kp", map_location=device))
    return net.to(device)


def set_trainable(net: torch.nn.Module) -> list:
    """Freeze all but the head; the backbone keeps its running batch-norm statistics."""
    for p in net.parameters():
        p.requires_grad = False
    for p in net.head.parameters():
        p.requires_grad = True
    net.eval()
    net.head.train()
    return [p for p in net.head.parameters()]


@torch.no_grad()
def evaluate(net, loader, device: str, kp_th: float) -> dict:
    """Keypoint error in 960-wide pixels of peaks above the threshold, and hit rates."""
    net.eval()
    errors, n_true, n_found, n_false = [], 0, 0, 0
    loss_sum, count = 0.0, 0
    for image, target, uv in loader:
        out = net(image.to(device))
        loss_sum += float(((out - target.to(device)) ** 2).mean()) * len(image)
        count += len(image)
        peaks = get_keypoints_from_heatmap_batch_maxpool(out[:, :-1], min_keypoint_pixel_distance=1)
        peaks = peaks.squeeze(2).cpu().numpy()   # (batch, 57, 3): x, y at 960 wide, score
        for b in range(len(image)):
            for k in range(N_KP):
                is_true = bool(np.isfinite(uv[b, k, 0]))
                is_found = peaks[b, k, 2] >= kp_th
                n_true += is_true
                if is_found and is_true:
                    n_found += 1
                    errors.append(np.hypot(*(peaks[b, k, :2] - uv[b, k].numpy())))
                elif is_found:
                    n_false += 1
    errors = np.asarray(errors)
    return {"val_loss": loss_sum / max(count, 1), "kp_err_median_px": float(np.median(errors)),
            "kp_err_p90_px": float(np.percentile(errors, 90)), "recall": n_found / max(n_true, 1),
            "false_per_image": n_false / max(count, 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", required=True,
                        help="a / b (heads folder of that half), or any name with --data")
    parser.add_argument("--data", type=Path, default=None,
                        help="heads folder (keypoints_<split>.npz, images/<split>/)")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--accum", type=int, default=2, help="images per optimiser step")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--limit", type=int, default=4000, help="training images drawn per fold")
    parser.add_argument("--val-limit", type=int, default=None, help="for smoke tests")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    half = Config.HALF_FOLDS.get(args.fold, args.fold)
    folder = args.data or HEADS / half
    out = args.out / args.fold
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    kp_th = float(json.loads((Config.CALIB_ROOT / "chosen_thresholds.json").read_text())
                  ["pnlcalib_mv"]["kp_th"])
    train = DataLoader(HeadData(folder, "train", args.limit), batch_size=args.batch, shuffle=True,
                       num_workers=args.workers, generator=torch.Generator().manual_seed(SEED))
    val = DataLoader(HeadData(folder, "val", args.val_limit), batch_size=args.batch,
                     num_workers=args.workers)
    net = load_net(args.device)
    params = set_trainable(net)
    optimizer = torch.optim.Adam(params, lr=args.lr)
    (out / "config.json").write_text(json.dumps(
        {**vars(args), "out": str(args.out), "data": str(folder), "half": half, "kp_th": kp_th,
         "train_images": len(train.dataset),
         "val_images": len(val.dataset), "seed": SEED, "torch": str(torch.__version__),
         "python": platform.python_version()}, indent=1))

    log = [{"epoch": 0, **evaluate(net, val, args.device, kp_th)}]
    print(log[-1], flush=True)
    best = log[-1]["val_loss"]
    torch.save(net.state_dict(), out / "MV_kp_head")
    began = time.time()
    for epoch in range(1, args.epochs + 1):
        set_trainable(net)
        running = 0.0
        optimizer.zero_grad()
        for step, (image, target, _) in enumerate(train, 1):
            loss = ((net(image.to(args.device)) - target.to(args.device)) ** 2).mean()
            (loss / args.accum).backward()
            # Gradient accumulation: batch 1 fits the 6 GB GPU, the update sees `accum` images.
            if step % args.accum == 0 or step == len(train):
                optimizer.step()
                optimizer.zero_grad()
            running += float(loss.detach())
            if step % 500 == 0:
                print(f"epoch {epoch} step {step}: loss {running / step:.3e}, "
                      f"{time.time() - began:.0f} s", flush=True)
        log.append({"epoch": epoch, "train_loss": running / len(train),
                    **evaluate(net, val, args.device, kp_th)})
        print(log[-1], flush=True)
        if log[-1]["val_loss"] < best:
            best = log[-1]["val_loss"]
            torch.save(net.state_dict(), out / "MV_kp_head")
    pd.DataFrame(log).to_csv(out / "log.csv", index=False)


if __name__ == "__main__":
    main()
