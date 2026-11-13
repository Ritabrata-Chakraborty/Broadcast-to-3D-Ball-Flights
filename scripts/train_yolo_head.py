"""Fine-tune only the Detect head of SoccerNet-v3D's YOLO11-L ball detector on one LP half.

The backbone and neck (layers 0-22) stay frozen. Broadcast: data from head_data_broadcast.py, fold a
trains on half 1 and fold b on half 2. Static: --data and --out point at head_data_static.py's data.
Output: outputs/heads/broadcast/yolo/<fold>/ (ultralytics run, weights/best.pt).
Run in the snv3d env: scripts/py --env snv3d scripts/train_yolo_head.py --fold a
"""
import argparse
import json
import platform
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as Config

WEIGHTS = Config.WEIGHTS_ROOT / "snv3d" / "yolo-sn-ball.pt"
HEADS = Config.HEADS_DIR / "broadcast" / "data"
OUT = Config.HEADS_DIR / "broadcast" / "yolo"
FROZEN_LAYERS = 23   # everything before the Detect head (layer 23 of 24)
IMGSZ = 1280
SEED = 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", required=True,
                        help="a / b (heads folder of that half), or any name with --data")
    parser.add_argument("--data", type=Path, default=None, help="ultralytics data.yaml")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr0", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    from ultralytics import YOLO
    import ultralytics

    half = Config.HALF_FOLDS.get(args.fold, args.fold)
    data = args.data or HEADS / half / "data.yaml"
    # ultralytics nests a relative project folder under runs/<task>/: always pass an absolute path.
    out = args.out.resolve()
    if (out / args.fold).exists():
        raise FileExistsError(f"{out / args.fold} exists; move it away first")
    model = YOLO(str(WEIGHTS))
    model.train(data=str(data), imgsz=IMGSZ, epochs=args.epochs,
                batch=args.batch, freeze=FROZEN_LAYERS, lr0=args.lr0, optimizer="AdamW",
                cos_lr=True, seed=SEED, deterministic=True, workers=args.workers,
                project=str(out), name=args.fold, exist_ok=False, plots=False, verbose=False)
    (out / args.fold / "joint_config.json").write_text(json.dumps(
        {"fold": args.fold, "half": half, "data": str(data), "weights": str(WEIGHTS),
         "frozen_layers": FROZEN_LAYERS,
         "imgsz": IMGSZ, "epochs": args.epochs, "batch": args.batch, "lr0": args.lr0,
         "seed": SEED, "ultralytics": ultralytics.__version__, "torch": str(torch.__version__),
         "python": platform.python_version()}, indent=1))


if __name__ == "__main__":
    main()
