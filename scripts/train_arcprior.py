"""Train or fine-tune ArcPrior on one fold: contact heights as input, corrupted inputs.

One model for full, cut, masked and masked cut flights: each step draws a cut (whole flight with
probability --full-share), runs of missing frames and a run of wrong points
(arcprior.contacts.corrupt), fills short gaps as the video tracker does, and draws the user's
contact heights (arcprior.contacts.sample_contacts; the end is never given on a cut flight).
Checkpoint choice: the network alone on the mixed full-flight validation set (clean / masked track,
correct / wrong / missing heights), mean of the broadcast and static median RMSE over shares 0.3,
0.5, 0.7 and 1.0 (the end contact is dropped below 1.0).
Fine-tuning: --init <run_dir> starts from that run's best.pt; --replay-scenario mixes a share of
another input setting (the same flights) into every batch.
Output: outputs/eval/runs/<run_id>/ (config, log, best.pt, last.pt, summary).
Run in the snv3d env:
  scripts/py --env snv3d scripts/train_arcprior.py --fold 1 --name base --net-device cuda
"""
import argparse
import json
import math
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as Config
from arcprior.contacts import build_test_set, corrupt, fill_short_gaps, gap_table, sample_contacts
from arcprior.network import (N_FEATURES_V5, ArcPrior, build_batch3, compute_losses3, cut_at,
                              decode3, sample_cuts)
from arcprior.rays import FlightSet

VAL_SHARES = (0.3, 0.5, 0.7, 1.0)


def seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    torch.use_deterministic_algorithms(True, warn_only=True)
    # The fused attention kernels' backward is not deterministic on CUDA; the math kernel is.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def forward(model: ArcPrior, batch) -> torch.Tensor:
    """Raw network outputs on the CPU, the transformer on its own device."""
    device = next(model.parameters()).device
    return model(batch.tokens.to(device), batch.padding.to(device)).cpu()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--fold", required=True, choices=sorted(Config.FOLDS))
    parser.add_argument("--scenario", default="gt",
                        help="dataset under outputs/eval/data (e.g. real_src) or outputs/folds")
    parser.add_argument("--full-share", type=float, default=0.5)
    parser.add_argument("--no-corrupt", action="store_true", help="ablation: no masking")
    parser.add_argument("--p-clean", type=float, default=0.3)
    parser.add_argument("--max-missing", type=float, default=0.4)
    parser.add_argument("--p-wrong", type=float, default=0.3)
    parser.add_argument("--init", type=Path, default=None, help="fine-tune from this run")
    parser.add_argument("--replay-scenario", default=None)
    parser.add_argument("--replay-share", type=float, default=0.3)
    parser.add_argument("--hypotheses", type=int, default=4)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--w-path", type=float, default=1.0)
    parser.add_argument("--w-reproj", type=float, default=0.1)
    parser.add_argument("--w-end", type=float, default=0.02)
    parser.add_argument("--w-height", type=float, default=0.1)
    parser.add_argument("--w-params", type=float, default=0.1)
    parser.add_argument("--w-nll", type=float, default=0.05)
    parser.add_argument("--w-select", type=float, default=0.1)
    parser.add_argument("--real-label-weight", type=float, default=0.3)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--net-device", default="cpu")
    return parser.parse_args()


def root_of(scenario: str) -> Path:
    """A dataset under outputs/eval/data (e.g. real_src), else under outputs/folds."""
    own = Config.EVAL_DATA / scenario
    return own if own.exists() else Config.FOLDS_DIR / scenario


def load(fold: str, split: str, scenario: str) -> FlightSet:
    return FlightSet(fold, split, "cpu", root=root_of(scenario))


def attach_test_inputs(flights: FlightSet, fold: str, scenario: str, gaps: dict) -> None:
    """Replace a validation FlightSet's inputs by the mixed full-flight set (build_batch3 reads
    the contact heights and the filled flags from it)."""
    with np.load(root_of(scenario) / f"fold_{fold}" / "val.npz") as archive:
        built = build_test_set({k: archive[k] for k in archive.files}, "full", "val", gaps)
    for key in ("uv_det", "size_det", "filled", "h_start", "h_end", "start_given",
                "end_given"):
        setattr(flights, key, torch.as_tensor(built[key]))


@torch.no_grad()
def network_rmse(model: ArcPrior, flights: FlightSet, share: float, batch_size: int = 32
                 ) -> np.ndarray:
    """Whole-flight 3D RMSE (m) of the network alone (highest-logit hypothesis) per view-flight."""
    was_training = model.training
    model.eval()
    out = []
    for begin in range(0, flights.size, batch_size):
        index = torch.arange(begin, min(begin + batch_size, flights.size))
        batch = build_batch3(flights, index, cut_at(flights.n[index], share))
        decoded = decode3(forward(model, batch), batch)
        pick = decoded["logit"].argmax(dim=1)
        path = decoded["path"][torch.arange(len(index)), pick]
        gap = ((path - batch.xyz) ** 2).sum(-1)
        has_gt = batch.has_gt.float()
        out.append(torch.sqrt((gap * has_gt).sum(1) / has_gt.sum(1).clamp(min=1)).numpy())
    model.train(was_training)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.concatenate(out)


def score(model: ArcPrior, val: FlightSet) -> dict:
    row = {}
    is_broadcast = (val.view == 0).numpy()
    for share in VAL_SHARES:
        rmse = network_rmse(model, val, share)
        row[f"b_{share}"] = float(np.median(rmse[is_broadcast]))
        row[f"s_{share}"] = float(np.median(rmse[~is_broadcast]))
    row["select"] = float(np.mean([(row[f"b_{s}"] + row[f"s_{s}"]) / 2 for s in VAL_SHARES]))
    return row


class Sampler:
    """Corrupted training batches of one scenario's train flights."""

    def __init__(self, flights: FlightSet, gaps: dict, args: argparse.Namespace) -> None:
        self.flights, self.gaps, self.args = flights, gaps, args

    def batch(self, index: torch.Tensor, generator: torch.Generator):
        f, args = self.flights, self.args
        mirrored = torch.rand(len(index), generator=generator) < 0.5
        n = f.n[index]
        cut = sample_cuts(n, generator, args.full_share)
        uv, size = f.uv_det[index], f.size_det[index]
        if not args.no_corrupt:
            uv, size = corrupt(uv, size, n, f.view[index], self.gaps, f.uv_det, f.size_det,
                               f.view, f.n, generator, args.p_clean, args.max_missing,
                               args.p_wrong)
        uv, filled = fill_short_gaps(uv, n)
        contact = sample_contacts(f.z_start[index].float(), f.z_end[index].float(), cut == n,
                                  generator)
        return build_batch3(f, index, cut, mirrored, args.real_label_weight, uv=uv,
                            size_override=size, filled=filled, contact=contact)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    torch.set_num_threads(args.threads)
    run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_f{args.fold}_{args.name}_s{args.seed}"
    out = Config.EVAL_ROOT / "runs" / run_id
    out.mkdir(parents=True)
    train = load(args.fold, "train", args.scenario)
    val = load(args.fold, "val", args.scenario)
    # Gap lengths of the video track on this fold's training flights.
    with np.load(Config.FOLDS_DIR / f"real_{Config.TRACK_SOURCE[args.fold]}" /
                 f"fold_{args.fold}" / "train.npz") as video:
        gaps = gap_table(video["uv_det"], video["n"], video["view"])
    attach_test_inputs(val, args.fold, args.scenario, gaps)
    samplers = [(1.0 - (args.replay_share if args.replay_scenario else 0.0),
                 Sampler(train, gaps, args))]
    if args.replay_scenario:
        replay = load(args.fold, "train", args.replay_scenario)
        if not torch.equal(replay.group, train.group):
            raise ValueError("replay scenario holds different flights")
        samplers.append((args.replay_share, Sampler(replay, gaps, args)))

    model = ArcPrior(args.hypotheses, dropout=args.dropout, n_features=N_FEATURES_V5)
    if args.init is not None:
        state = torch.load(args.init / "best.pt", map_location="cpu")
        if state["config"]["fold"] != args.fold:
            raise ValueError("init run is another fold")
        model.load_state_dict(state["model"])
    model.to(args.net_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup = max(1, min(200, args.steps // 20))

    def schedule(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, args.steps - warmup)
        return 0.02 + 0.98 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    weights = {"path": args.w_path, "reproj": args.w_reproj, "end": args.w_end,
               "height": args.w_height, "params": args.w_params, "nll": args.w_nll,
               "select": args.w_select}
    info = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "run_id": run_id, "n_features": N_FEATURES_V5, "no_size": False,
            "gap_source": f"real_{Config.TRACK_SOURCE[args.fold]}",
            "gap_runs": {str(k): int(len(v)) for k, v in gaps.items()},
            "train_view_flights": train.size, "val_view_flights": val.size,
            "val_set": "mixed full", "torch": str(torch.__version__),
            "numpy": np.__version__, "python": platform.python_version(),
            "parameters": sum(p.numel() for p in model.parameters())}
    (out / "config.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info), flush=True)

    generator = torch.Generator().manual_seed(args.seed)
    log, best = [], (float("inf"), -1)
    n_skipped = 0
    began = time.time()
    for step in range(args.steps + 1):
        if step % args.eval_every == 0 or step == args.steps:
            row = {"step": step, "seconds": time.time() - began, "skipped": n_skipped,
                   **score(model, val)}
            log.append(row)
            pd.DataFrame(log).to_csv(out / "log.csv", index=False)
            print(json.dumps({k: round(v, 3) if isinstance(v, float) else v
                              for k, v in row.items()}), flush=True)
            if row["select"] < best[0]:
                best = (row["select"], step)
                torch.save({"model": model.state_dict(), "config": info, "step": step},
                           out / "best.pt")
        if step == args.steps:
            break
        total = 0.0
        for share, sampler in samplers:
            count = max(1, int(round(share * args.batch)))
            index = torch.randperm(sampler.flights.size, generator=generator)[:count]
            batch = sampler.batch(index, generator)
            decoded = decode3(forward(model, batch), batch)
            loss, _ = compute_losses3(decoded, batch, weights)
            total = total + loss * count / args.batch
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        if not torch.isfinite(grad_norm):
            n_skipped += 1
            if n_skipped > Config.MAX_SKIPPED_STEPS:
                raise RuntimeError(f"{n_skipped} non-finite gradients: stopping")
            optimizer.zero_grad(set_to_none=True)
            continue
        optimizer.step()
        scheduler.step()

    torch.save({"model": model.state_dict(), "config": info, "step": args.steps}, out / "last.pt")
    summary = {"run_id": run_id, "best_select": best[0], "best_step": best[1],
               "seconds": time.time() - began, **log[-1]}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
