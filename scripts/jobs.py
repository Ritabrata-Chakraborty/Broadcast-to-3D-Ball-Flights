"""Job lists of the evaluation stages, the whole-flight-share choice and the freeze.

Stages, in order (each writes outputs/eval/queue/<stage>.txt for scripts/run_list.sh, or a json):
  tune       fold 1, seed 0: train Base with whole-flight shares 0.5 / 0.6 / 0.7 and predict on
             the validation sets (network alone and ArcPrior with the weak-direction prior)
  pfull      choose the share: mean of E-full and E-part of ArcPrior on fold-1 validation
  base       train Base on GT inputs for every fold and seed (fold 1 seed 0 = the chosen tune run)
  finetune   fine-tune each Base on real inputs (published heads) and on the (det,cal) inputs
  val        validation predictions: every network x input setting with the prior candidates
             (seed 0), and LM-fitg + kNN-flights, on the full and partial test sets
  select     per-fold prior choice per network, input setting and test set (select_prior.py)
  freeze     outputs/eval/freeze.json from the selections, before any test prediction
  test       test predictions of the frozen arms (all seeds) and LM-fitg + kNN-flights
  baselines  Grad et al. and SNv3D-size (own boxes, track) on the full-flight test sets
  fixed      Table 1: Base, LM-fitg and Grad et al. on the fixed-condition test sets
  trainsplit train-split predictions of the networks (seed 0, clean track, correct heights)
Run in the ball3d env: scripts/py scripts/jobs.py <stage>; then scripts/run_list.sh <file> <jobs>.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as Config
from arcprior.metrics import load, selection_score

FOLDS = ("1", "2", "3", "4")
SEEDS = (0, 1, 2)
SHARES = (0.5, 0.6, 0.7)                 # whole-flight training shares tried on fold 1
# (network, input setting): Base on every input, each fine-tune on its own input, real-ft also on
# GT inputs (forgetting check).
COMBOS = (("base", "gt"), ("base", "real"), ("base", "heads"), ("real_ft", "real"),
          ("real_ft", "gt"), ("heads_ft", "heads"))
CANDIDATES = ("ap:none:0:obj", "ap:subspace:1:obj", "ap:subspace:3:obj", "ap:gauss:1:obj")
PART_SHARES = "0.3 0.4 0.5 0.6 0.7 0.8 0.9"
ROOT = Config.EVAL_ROOT
TRAIN = "scripts/py --env snv3d scripts/train_arcprior.py --net-device cuda --threads 3"
COMMON = "--answer-unanchored --batch 128 --threads 3"


def p_full() -> float:
    return json.loads((ROOT / "select" / "p_full.json").read_text())["chosen"]


def run_dir(model: str, fold: str, seed: int) -> str:
    """Shell expression for a training run (fold 1 seed 0 Base is the chosen tuning run)."""
    name = model
    if (model, fold, seed) == ("base", "1", 0):
        name = f"base_pf{round(100 * p_full())}"
    return f"$(ls -d outputs/eval/runs/*_f{fold}_{name}_s{seed} | head -1)"


def predict(dataset: str, fold: str, split: str, tag: str, methods: str, kind: str,
            run: str | None = None, omega: bool = False, frames: bool = False) -> str:
    shares = "1.0" if kind == "full" else PART_SHARES
    cmd = (f"scripts/py --env snv3d scripts/predict.py {dataset} --fold {fold} --split {split} "
           f"--tag {tag} --methods {methods} --shares {shares} {COMMON}")
    if run:
        cmd += f" --run {run}"
    if omega:
        cmd += " --omega"
    if frames and kind == "full":
        cmd += " --frames"
    return cmd


def physics(fold: str, split: str) -> list[str]:
    return [predict(f"{setting}_{kind}", fold, split, "physics", "lm_fitg m5", kind, omega=True,
                    frames=True)
            for setting in ("gt", "real", "heads") for kind in ("full", "part")]


def stage_tune() -> list[str]:
    lines = [f"{TRAIN} --fold 1 --name base_pf{round(100 * s)} --full-share {s} --seed 0"
             for s in SHARES]
    for s in SHARES:
        run = f"$(ls -d outputs/eval/runs/*_f1_base_pf{round(100 * s)}_s0 | head -1)"
        for kind in ("full", "part"):
            lines.append(predict(f"gt_{kind}", "1", "val", f"base_pf{round(100 * s)}_s0",
                                 "net ap:subspace:1:obj", kind, run))
    return lines


def stage_pfull() -> None:
    scores = {}
    for s in SHARES:
        tag = f"base_pf{round(100 * s)}_s0"
        parts = [selection_score(load([ROOT / "pred" / f"gt_{kind}" / tag /
                                       "fold_1_val_flights.csv.gz"]), "ap:subspace:1:obj",
                                 kind == "part")
                 for kind in ("full", "part")]
        scores[s] = sum(parts) / 2.0
    chosen = min(scores, key=scores.get)
    target = ROOT / "select" / "p_full.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"scores": scores, "chosen": chosen}, indent=1))
    print(scores, "->", chosen)


def stage_base() -> list[str]:
    share = p_full()
    return [f"{TRAIN} --fold {f} --name base --full-share {share} --seed {s}"
            for f in FOLDS for s in SEEDS if (f, s) != ("1", 0)]


def stage_finetune() -> list[str]:
    share = p_full()
    return [f"{TRAIN} --fold {f} --name {name} --scenario {setting}_src "
            f"--init {run_dir('base', f, s)} --replay-scenario gt_src --replay-share 0.3 "
            f"--steps 1500 --lr 1e-4 --eval-every 250 "
            f"--full-share {share} --seed {s}"
            for f in FOLDS for s in SEEDS for name, setting in (("real_ft", "real"),
                                                                ("heads_ft", "heads"))]


def stage_val() -> list[str]:
    lines = []
    for fold in FOLDS:
        for model, setting in COMBOS:
            for kind in ("full", "part"):
                lines.append(predict(f"{setting}_{kind}", fold, "val", f"{model}_s0",
                                     " ".join(CANDIDATES), kind, run_dir(model, fold, 0)))
        lines += physics(fold, "val")
    return lines


def stage_select() -> list[str]:
    lines = []
    for model, setting in COMBOS:
        for kind in ("full", "part"):
            candidates = " ".join(f"{setting}_{kind}/{model}_s0:{m}" for m in CANDIDATES)
            lines.append(f"scripts/py scripts/select_prior.py {model}_{setting}_{kind} "
                         f"--candidates {candidates}" + (" --partial" if kind == "part" else ""))
    return lines


def chosen(name: str) -> dict:
    report = json.loads((ROOT / "select" / f"{name}.json").read_text())
    return {f: row["chosen"].split(":", 1)[1] for f, row in report["folds"].items()}


def stage_freeze() -> None:
    path = ROOT / "freeze.json"
    if path.exists():
        raise FileExistsError(path)
    sets, choices = {}, {}
    for setting in ("gt", "real", "heads"):
        for kind in ("full", "part"):
            name = f"{setting}_{kind}"
            arms = {}
            for model, combo_setting in COMBOS:
                if combo_setting != setting:
                    continue
                pick = chosen(f"{model}_{setting}_{kind}")
                choices[f"{model}_{name}"] = pick
                arms[f"ArcPrior {model}"] = {f: [f"{name}/{model}_s{k}:{pick[f]}" for k in SEEDS]
                                             for f in FOLDS}
            arms["LM-fitg"] = {f: [f"{name}/physics:lm_fitg"] for f in FOLDS}
            arms["kNN-flights"] = {f: [f"{name}/physics:m5"] for f in FOLDS}
            if kind == "full" and setting in ("gt", "real"):
                arms["Grad et al."] = {f: [f"{name}/grad:b1"] for f in FOLDS}
                arms["SNv3D-size"] = {f: [f"{name}/snv3d_opt:snv3d"] for f in FOLDS}
                arms["SNv3D-size (track)"] = {f: [f"{name}/snv3d:snv3d"] for f in FOLDS}
            sets[name] = arms
    freeze = {"written": time.strftime("%Y-%m-%d %H:%M:%S"), "p_full": p_full(),
              "candidates": CANDIDATES, "choices": choices, "sets": sets}
    path.write_text(json.dumps(freeze, indent=1))
    print("frozen:", path)


def frozen() -> dict:
    return json.loads((ROOT / "freeze.json").read_text())


def stage_test() -> list[str]:
    freeze = frozen()
    lines = []
    for fold in FOLDS:
        for model, setting in COMBOS:
            for kind in ("full", "part"):
                method = freeze["choices"][f"{model}_{setting}_{kind}"][fold]
                for seed in SEEDS:
                    lines.append(predict(f"{setting}_{kind}", fold, "test", f"{model}_s{seed}",
                                         method, kind, run_dir(model, fold, seed), omega=True,
                                         frames=seed == 0))
        lines += physics(fold, "test")
    return lines


def stage_baselines() -> list[str]:
    lines = []
    for setting in ("gt", "real"):
        unrotate = " --unrotate-dev-views" if setting == "gt" else ""
        for fold in FOLDS:
            lines.append(f"scripts/py scripts/run_grad.py {setting}_full --fold {fold} --split "
                         f"test --tag grad --workers 3{unrotate}")
            for boxes in ("opt", "track"):
                lines.append(f"scripts/py scripts/run_snv3d.py {setting}_full --fold {fold} "
                             f"--splits test --boxes {boxes}")
    return lines


def stage_fixed() -> list[str]:
    choices = frozen()["choices"]["base_gt_full"]
    lines = []
    for level in ("correct", "wrong", "none"):
        name = f"gt_fix{level}"
        for fold in FOLDS:
            for seed in SEEDS:
                lines.append(predict(name, fold, "test", f"base_s{seed}", choices[fold], "full",
                                     run_dir("base", fold, seed)))
            lines.append(predict(name, fold, "test", "physics", "lm_fitg", "full"))
            if level == "none":
                lines.append(f"scripts/py scripts/run_grad.py {name} --fold {fold} --split test "
                             f"--tag grad --workers 3 --unrotate-dev-views")
    return lines


def stage_trainsplit() -> list[str]:
    freeze = frozen()
    lines = []
    for fold in FOLDS:
        for model, setting in (("base", "gt"), ("real_ft", "real"), ("heads_ft", "heads")):
            for kind in ("full", "part"):
                method = freeze["choices"][f"{model}_{setting}_{kind}"][fold]
                lines.append(predict(f"{setting}_{kind}", fold, "train", f"{model}_s0", method,
                                     kind, run_dir(model, fold, 0)))
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    stages = {"tune": stage_tune, "base": stage_base, "finetune": stage_finetune,
              "val": stage_val, "select": stage_select, "test": stage_test,
              "baselines": stage_baselines, "fixed": stage_fixed, "trainsplit": stage_trainsplit}
    parser.add_argument("stage", choices=(*stages, "pfull", "freeze"))
    args = parser.parse_args()
    if args.stage == "pfull":
        stage_pfull()
        return
    if args.stage == "freeze":
        stage_freeze()
        return
    lines = stages[args.stage]()
    target = ROOT / "queue" / f"{args.stage}.txt"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n")
    print(len(lines), "jobs ->", target)


if __name__ == "__main__":
    main()
