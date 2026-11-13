"""Predictions of the paper's methods on one test set, fold and split.

Methods (output names):
  lm_fitg   LM-fitg: Grad et al.'s drag + free-g model in our solver, three starts, no network
  m5        kNN-flights: mean GT path of the 5 nearest training flights by start anchor, first
            ray and time seen
  net       the network alone, highest-logit hypothesis (validation ablation)
  ap:<prior>:<weight>:obj  ArcPrior: every hypothesis refined with drag, lift and sidespin and a
            prior (none / gauss / subspace = weak-direction), the lowest objective kept
One row per view-flight, method and share seen: the capped whole-flight RMSE (E-full at share 1,
E-part below), coverage, contact and apex errors, fit score and Omega. With --frames, share 1 also
writes per-frame predictions in the dataset's decimetres (for the flight mAP of Grad et al.).
Output: outputs/eval/pred/<test set>/<tag>/fold_<k>_<split>_{flights,frames}.csv.gz
Run in the snv3d env: scripts/py --env snv3d scripts/predict.py gt_full --fold 1 --split val ...
"""
import argparse
import json
import platform
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as Config
from arcprior.flight import integrate
from arcprior.network import (N_FEATURES, ArcPrior, build_batch3, cut_at, decode3, repeat_batch,
                              to_world3)
from arcprior.observability import objective, observability, weak_projection
from arcprior.rays import FlightSet
from arcprior.solver import (LIFT_CFG, LM_FITG_CFG, RefineConfig, ball3d_guess, pack, pack_prior,
                             refine)

START_TIMES = (1.0, 2.0, 4.0)
NEIGHBOURS = 5


# ---------------------------------------------------------------- Starts and solvers


def physics_starts(batch) -> list[torch.Tensor]:
    """ball3d's guess and two longer flight times, so a rising cut ball can keep rising."""
    base = ball3d_guess(batch)
    seen = ((batch.seen.sum(dim=1) - 1).clamp(min=2).double() / Config.FPS)
    starts = []
    for multiple in START_TIMES:
        theta = base.clone()
        theta[:, 5] = Config.G_MPS2 * seen * multiple / 2.0
        if multiple == 1.0:
            theta[:, 5] = base[:, 5]
        starts.append(theta)
    return starts


def best_of(starts: list[torch.Tensor], batch, cfg: RefineConfig) -> tuple:
    best_theta, best_score = None, None
    for start in starts:
        theta, score = refine(start, batch, cfg)
        if best_theta is None:
            best_theta, best_score = theta, score
            continue
        is_better = score < best_score
        best_theta = torch.where(is_better[:, None], theta, best_theta)
        best_score = torch.where(is_better, score, best_score)
    return best_theta, best_score


def with_g(theta9: torch.Tensor) -> torch.Tensor:
    g = torch.full_like(theta9[..., :1], Config.G_MPS2)
    return torch.cat([theta9, g], dim=-1).double()


def parse_ap(name: str) -> tuple[str, float, str]:
    _, kind, weight, select = name.split(":")
    return kind, float(weight), select


def arcprior_fit(decoded: dict, batch, name: str, box: float) -> dict:
    """Refine every hypothesis with one prior setting and keep one per flight."""
    kind, weight, select = parse_ap(name)
    count, k = decoded["theta"].shape[:2]
    theta0 = with_g(decoded["theta"].reshape(count * k, -1))
    std = decoded["std"].reshape(count * k, -1).double()
    wide = repeat_batch(batch, k)
    cfg = replace(LIFT_CFG, start_shift_m=box, prior_weight=weight if kind != "none" else 0.0,
                  prior_kind="old" if kind in ("none", "old") else kind)
    prior = theta0
    if kind in ("gauss", "subspace"):
        proj = weak_projection(theta0, std, wide, cfg, weight) if kind == "subspace" else None
        prior = pack_prior(theta0, std, proj)
    theta, score = refine(theta0, wide, cfg, prior=prior)
    cost = objective(theta, wide, cfg, prior) / pack(wide)["n_present"]
    if select != "obj":
        raise ValueError(f"unknown selection {select}")
    pick = (-cost).view(count, k).argmax(dim=1)
    rows = torch.arange(count) * k + pick
    return {"theta": theta[rows], "score": score[rows], "cfg": cfg,
            "prior": prior[rows] if prior is not None else None, "hypothesis": pick}


# ---------------------------------------------------------------- kNN-flights


def flight_features(batch) -> torch.Tensor:
    """Start anchor (world), first ray direction (world) and seen time."""
    rows = torch.arange(len(batch.index))
    first = batch.present.float().argmax(dim=1)
    ray = batch.ray[rows, first]
    ray = torch.einsum("bji,bj->bi", batch.to_flight, ray)
    seen = (batch.seen.sum(dim=1) - 1).float() / Config.FPS
    return torch.cat([batch.origin[:, :2] / 10.0, ray * 3.0, seen[:, None]], dim=1)


class NearestFlights:
    """GT paths of the train flights, relative to their start, with their kNN features."""

    def __init__(self, train: FlightSet, share: float) -> None:
        index = torch.arange(train.size)
        batch = build_batch3(train, index, cut_at(train.n, share))
        self.features = flight_features(batch)
        # Without contacts (video inputs) no train flight is anchored; the origin is then the
        # first ray's ground hit for train and query flights alike.
        self.keep = batch.has_pitch_start if bool(batch.has_pitch_start.any()) else batch.has_obs
        xyz = train.xyz.clone()
        start = xyz[:, 0].clone()
        start[:, 2] = 0.0
        self.relative = xyz - start[:, None]
        self.n = train.n

    def predict(self, batch) -> torch.Tensor:
        """Mean neighbour path in world metres, shifted to the flight's start anchor."""
        query = flight_features(batch)
        pool = self.features[self.keep]
        spread = pool.std(dim=0).clamp(min=1e-3)
        dist = torch.cdist(query / spread, pool / spread)
        nearest = dist.topk(NEIGHBOURS, largest=False).indices
        candidates = torch.where(self.keep)[0][nearest]
        paths = self.relative[candidates][:, :, :batch.n_frames]
        mean = torch.nanmean(paths, dim=1)
        mean = torch.where(torch.isnan(mean), torch.zeros_like(mean), mean)
        origin = torch.cat([batch.origin[:, :2], torch.zeros_like(batch.origin[:, :1])], dim=1)
        return mean + origin[:, None]


# ---------------------------------------------------------------- Scoring


def path_of(theta: torch.Tensor, horizon: int) -> torch.Tensor:
    return integrate(theta[:, :3], theta[:, 3:6], theta[:, 6], theta[:, 7], theta[:, 8], horizon,
                     theta[:, 9]).float()


def answer_mask(batch, answerable: torch.Tensor) -> torch.Tensor:
    """Frames a method answers: the whole flight, or up to the last detection on a full flight
    without an end anchor (ball3d's rule); nothing for an unanswerable flight."""
    frame = torch.arange(batch.n_frames)[None]
    last_seen = batch.present.shape[1] - 1 - batch.present.flip(1).float().argmax(dim=1)
    stop = torch.where(batch.end_known & ~batch.has_pitch_end, last_seen, batch.n_full - 1)
    return (frame <= stop[:, None]) & batch.in_flight & answerable[:, None]


def flight_table(path_flight: torch.Tensor, batch, answered: torch.Tensor) -> dict:
    """Per-flight metrics against the GT, both in the flight frame (distances are unchanged)."""
    err = (path_flight - batch.xyz).norm(dim=-1)
    valid = batch.has_gt.float()
    ans = answered.float()
    cap = Config.CAP_M
    capped = torch.where(answered, err.clamp(max=cap), torch.full_like(err, cap))
    count = valid.sum(dim=1).clamp(min=1.0)
    rows = torch.arange(len(err))
    seen = batch.seen.float() * valid
    future = (1.0 - batch.seen.float()) * valid

    def rms(weight: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        total = weight.sum(dim=1)
        out = torch.sqrt((values ** 2 * weight).sum(dim=1) / total.clamp(min=1.0))
        return torch.where(total > 0, out, torch.full_like(out, float("nan")))

    def at(frame: torch.Tensor) -> torch.Tensor:
        ok = batch.has_gt[rows, frame] & answered[rows, frame]
        return torch.where(ok, err[rows, frame], torch.full_like(err[:, 0], float("nan")))

    z_pred = torch.where(batch.in_flight, path_flight[..., 2], torch.full_like(err, -1e9))
    z_true = torch.where(batch.has_gt, batch.xyz[..., 2], torch.full_like(err, -1e9))
    is_answered = answered.any(dim=1)
    apex = torch.where(is_answered, (z_pred.max(dim=1).values - z_true.max(dim=1).values).abs(),
                       torch.full_like(err[:, 0], float("nan")))
    return {"rmse_cap": torch.sqrt((capped ** 2 * valid).sum(dim=1) / count),
            "rmse_answered": rms(valid * ans, err), "coverage": (valid * ans).sum(1) / count,
            "rmse_seen_cap": rms(seen, capped), "rmse_unseen_cap": rms(future, capped),
            "start_err": at(torch.zeros_like(batch.n_full)), "end_err": at(batch.n_full - 1),
            "apex_err": apex, "answered": is_answered}


# ---------------------------------------------------------------- Main loop


@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    root = args.data_root / args.scenario
    flights = FlightSet(args.fold, args.split, "cpu", root=root)
    model = None
    if args.run is not None:
        state = torch.load(args.run / "best.pt", map_location="cpu")
        info = state["config"]
        model = ArcPrior(info["hypotheses"], dropout=info["dropout"], has_size=not info["no_size"],
                         n_features=info.get("n_features", N_FEATURES))
        model.load_state_dict(state["model"])
        model.eval()
        if info["fold"] != args.fold:
            raise ValueError(f"run is fold {info['fold']}, not {args.fold}")
    needs_train = "m5" in args.methods
    train = FlightSet(args.fold, "train", "cpu", root=root) if needs_train else None
    has_camera = getattr(flights, "has_camera", torch.ones(flights.size, dtype=torch.bool))
    out = args.out_root / args.scenario / args.tag
    out.mkdir(parents=True, exist_ok=True)
    stem = out / f"fold_{args.fold}_{args.split}"
    if Path(f"{stem}_flights.csv.gz").exists():
        raise FileExistsError(f"{stem}_flights.csv.gz exists")
    rows, frames = [], []
    for share in args.shares:
        nearest = NearestFlights(train, share) if needs_train else None
        for begin in range(0, flights.size, args.batch):
            index = torch.arange(begin, min(begin + args.batch, flights.size))
            batch = build_batch3(flights, index, cut_at(flights.n[index], share))
            anchored = batch.has_pitch_start | args.answer_unanchored
            answerable = batch.has_obs & has_camera[index] & anchored
            decoded = decode3(model(batch.tokens, batch.padding), batch) if model else None
            fits = {}
            for name in args.methods:
                cfg = None
                if name == "lm_fitg":
                    cfg = replace(LM_FITG_CFG, start_shift_m=args.box)
                    theta, score = best_of(physics_starts(batch), batch, cfg)
                    fits[name] = {"theta": theta, "score": score, "cfg": cfg, "prior": None}
                elif name == "m5":
                    fits[name] = {"path_world": nearest.predict(batch),
                                  "score": torch.full((len(index),), float("nan"))}
                elif name == "net":
                    pick = decoded["logit"].argmax(dim=1)
                    theta = with_g(decoded["theta"][torch.arange(len(index)), pick])
                    _, score = refine(theta, batch, RefineConfig(n_iters=0))
                    fits[name] = {"theta": theta, "score": score, "cfg": None, "prior": None}
                elif name.startswith("ap:"):
                    fits[name] = arcprior_fit(decoded, batch, name, args.box)
                else:
                    raise ValueError(f"unknown method {name}")

            for name, fit in fits.items():
                if "path_world" in fit:
                    world = fit["path_world"]
                    flight_path = torch.einsum("bij,btj->bti", batch.to_flight,
                                               world - batch.origin[:, None])
                else:
                    flight_path = path_of(fit["theta"], batch.n_frames)
                    world = to_world3(flight_path, batch)
                answered = answer_mask(batch, answerable)
                table = flight_table(flight_path, batch, answered)
                omega = torch.full((len(index),), float("nan"))
                if args.omega and fit.get("cfg") is not None:
                    omega = observability(fit["theta"], batch, fit["cfg"], fit["prior"],
                                          batch.n_frames, batch.in_flight)["omega"].float()
                for r, i in enumerate(index.tolist()):
                    row = {"method": name, "share": share, "view": int(flights.view[i]),
                           "half": int(flights.half[i]), "group": int(flights.group[i]),
                           "n": int(flights.n[i]), "cut": int(batch.cut[r]),
                           "start_high": bool(flights.start_high[i]),
                           "end_high": bool(flights.end_high[i]),
                           "z_start": float(flights.z_start[i]), "z_end": float(flights.z_end[i]),
                           "apex_true": float(np.nanmax(flights.xyz[i, :flights.n[i], 2].numpy())),
                           "fit_score": float(fit["score"][r]) if bool(table["answered"][r])
                           else float("nan"),
                           "omega": float(omega[r]),
                           **{k: float(v[r]) for k, v in table.items()}}
                    if "theta" in fit:
                        row.update(k3=float(fit["theta"][r, 6]), kl=float(fit["theta"][r, 7]),
                                   ks=float(fit["theta"][r, 8]), g=float(fit["theta"][r, 9]))
                    if "hypothesis" in fit:
                        row["hypothesis"] = int(fit["hypothesis"][r])
                    rows.append(row)
                    if share == 1.0 and args.frames:
                        n = int(flights.n[i])
                        values = world[r, :n].numpy() * Config.DM_PER_M
                        values[~answered[r, :n].numpy()] = np.nan
                        frames.append(pd.DataFrame({
                            "method": name, "view": int(flights.view[i]),
                            "half": int(flights.half[i]), "group": int(flights.group[i]),
                            "file_name": int(flights.a[i]) + np.arange(n),
                            "x_predicted": values[:, 0], "y_predicted": values[:, 1],
                            "z_predicted": values[:, 2], "fit_score": row["fit_score"],
                            "omega": row["omega"]}))
        print(f"{args.scenario} fold {args.fold} {args.split} share {share}: done", flush=True)
    pd.DataFrame(rows).to_csv(f"{stem}_flights.csv.gz", index=False)
    if frames:
        pd.concat(frames).to_csv(f"{stem}_frames.csv.gz", index=False)
    meta = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    meta.update(view_flights=flights.size, python=platform.python_version(),
                torch=str(torch.__version__), numpy=np.__version__,
                finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    Path(f"{stem}_meta.json").write_text(json.dumps(meta, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario")
    parser.add_argument("--fold", required=True, choices=sorted(Config.FOLDS))
    parser.add_argument("--split", required=True, choices=("train", "val", "test"))
    parser.add_argument("--run", type=Path, default=None, help="ArcPrior run (best.pt)")
    parser.add_argument("--tag", required=True, help="output folder name")
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--shares", nargs="+", type=float, default=[1.0])
    parser.add_argument("--box", type=float, default=0.1, help="start box half-width (m)")
    parser.add_argument("--answer-unanchored", action="store_true")
    parser.add_argument("--omega", action="store_true", help="compute Omega (slower)")
    parser.add_argument("--frames", action="store_true", help="write per-frame predictions")
    parser.add_argument("--data-root", type=Path, default=Config.EVAL_DATA)
    parser.add_argument("--out-root", type=Path, default=Config.EVAL_ROOT / "pred")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    run(args)


if __name__ == "__main__":
    main()
