<h1 align="center">From Broadcast Video to 3D Soccer Ball Flights</h1>
<h3 align="center">A Learned Flight Prior and a Benchmark across Input Conditions</h3>

<p align="center">
  <a href="https://scholar.google.com/citations?user=JikCewwAAAAJ&hl=en"><b>Ritabrata Chakraborty</b></a><br>
  School of Engineering and Applied Science, University of Pennsylvania
</p>

<p align="center">
  <a href="https://ritabrata-chakraborty.github.io/Broadcast-to-3D-Ball-Flights/"><img src="https://img.shields.io/badge/Project-Page-0F6B5C?logo=googlechrome&logoColor=white" alt="Project page"></a>
  <img src="https://img.shields.io/badge/Paper-On%20Request-B31B1B?logo=googlescholar&logoColor=white" alt="Paper on request">
  <img src="https://img.shields.io/badge/CVPR%202027-Under%20Review-6A4C93" alt="Under review at CVPR 2027">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white" alt="Python 3.11">
  <img src="https://img.shields.io/badge/PyTorch-CUDA-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
</p>

## Abstract

From one broadcast camera, a soccer ball's flight is fixed only up to its depth along the viewing rays, and published monocular methods reconstruct only the frames they see. We present, to our knowledge, the first soccer pipeline from single-camera video to physically valid 3D ball flights that is built from published detection and calibration networks and evaluated end to end against 3D ground truth: a ball detector, a tracker and a calibration network, and ArcPrior, a transformer over per-frame viewing rays whose flight hypotheses and spreads guide a least-squares solver. The user marks only the contact frames; contact heights are optional, and ArcPrior also predicts the unseen part of a partially seen flight. On the dataset of Grad *et al.*, the only public match we know of with 3D ball ground truth and broadcast video, we benchmark published methods and ours across controlled input conditions (dataset and video inputs, clean and masked tracks, correct, wrong and missing contact heights, full and partial flights), and study the ball detector, tracker and calibration networks. Without contact heights, ArcPrior is at 2.24 m against 3.11 m for the optimiser of Grad *et al.* on the same flights; on the unseen frames of partial flights it is at 6.4 against 9.1 m for physics alone; from video, with fine-tuned detection and calibration heads, it reaches 2.67 m on broadcast. All results come from four cross-validation folds on one match.

## Input Settings

| Setting | Where the per-frame rays come from |
|---|---|
| **GT** | The dataset's own 2D ball track and cameras (inputs, not its 3D ground truth) |
| **Real** | Video: YOLO11-L ball detector (SoccerNet-v3D weights), our tracker, PnLCalib MV |
| **Real, heads (det,cal)** | The same with the detector's and PnLCalib's heads fine-tuned on labels of the other half |

In every setting the user gives the contact frames that bound a flight and, optionally, the contact heights; test sets mix clean and masked tracks with correct, wrong and missing heights. The baselines, the optimiser of Grad *et al.* and SNv3D-size (SoccerNet-v3D's size-based localisation), run unchanged.

## Repository Layout

| Path | Contents |
|---|---|
| `config.py` | Paths, the LP match, flight selection, folds, model bounds, evaluation constants |
| `src/arcprior/` | `flight.py` (RK4 flight model), `rays.py` (datasets, rays, projection), `network.py` (ArcPrior tokens, transformer, losses), `contacts.py` (contact heights, corruption, mixed test sets), `solver.py` (Levenberg–Marquardt, priors), `observability.py` (Ω), `oracle.py` (oracle 3D fit), `metrics.py` (E-full, E-part, bootstrap, Holm) |
| `src/snv3d/` | SoccerNet-v3D size-based localisation |
| `scripts/` | One script per pipeline stage (below); `run_pipeline.sh` runs them in order |
| `tests/` | Unit tests |
| `env/` | Pinned requirements of the two environments |
| `tools/build_page.py` | Builder of the encrypted project page |

## External Code, Data and Weights

Nothing below is in this repository; download it and place it as shown.

| What | Source | Place at |
|---|---|---|
| Code and data of Grad *et al.* (CVPRW 2026): LP match, video, cameras, 3D ground truth | [github.com/lukaszgrad/ball3d](https://github.com/lukaszgrad/ball3d), data release linked in its `DATA.md` | `external/ball3d/`, data in `external/ball3d/data/LP/` |
| SoccerNet-v3D ball detectors (Gutiérrez-Pérez and Agudo, CVPRW 2025) | [SoccerNet-v3D releases](https://github.com/mguti97/SoccerNet-v3D/releases/tag/v1.0.0): `yolo-sn-ball.pt`, `yolo-sn-ball-opt.pt` | `weights/snv3d/` |
| PnLCalib and its MV weights | [github.com/mguti97/PnLCalib](https://github.com/mguti97/PnLCalib) and its releases | `third_party/PnLCalib/`, `weights/pnlcalib/{MV_kp,MV_lines}` |
| Calibration study only: SoccerMaster, SigLIP 2, SoccerNet 2022/2023, WC14, TSWC, ISSIA | see the SoccerMaster and PnLCalib READMEs | `external/SoccerMaster/`, `weights/soccermaster/`, `weights/siglip2/`, `data/` |

The data and video of Grad *et al.* are for non-commercial research only; do not redistribute them. Figures drawn on video frames must not be redistributed either.

## Environment

Two conda environments (Python 3.11):

```bash
# ball3d (CPU): the pinned stack of Grad et al. (their optimiser and evaluator), reports, figures
mamba create -n ball3d python=3.11
uv pip install --python ~/miniforge3/envs/ball3d/bin/python -r env/ball3d-requirements.txt
uv pip install --python ~/miniforge3/envs/ball3d/bin/python pytest ruff tabulate

# snv3d (GPU): YOLO11, PnLCalib, ArcPrior training and prediction
mamba create -n snv3d python=3.11
~/miniforge3/envs/snv3d/bin/pip install -r env/snv3d-requirements.txt   # torch for your CUDA first
```

Run every script through `scripts/py [--env snv3d] <script>`; it sets `PYTHONPATH=src:external/ball3d/src` and expects the environments under `~/miniforge3/envs/`. Every script's docstring gives its inputs, outputs and options (`--help`). Everything was run on one RTX 3060 laptop GPU (6 GB).

```bash
scripts/py -m pytest -q tests                  # ball3d env: tests that need its stack
scripts/py --env snv3d -m pytest -q tests      # snv3d env: tests that need torch
```

## Reproducing the Paper

`scripts/run_pipeline.sh <stage>...` runs the stages below in order; all outputs go to `outputs/`. Each stage expects the outputs of the ones before it, and no script overwrites existing outputs.

| Stage | Scripts | Output |
|---|---|---|
| `flights` | `build_flights.py`: airborne flights (apex ≥ 1 m) mapped to every view, oracle fits | `outputs/flights/` |
| `video` | `extract_video.py`: YOLO11 boxes and PnLCalib MV cameras; the size-optimised boxes of SNv3D-size | `outputs/extract/{published,snv3d_opt}/` |
| `folds` | `build_folds.py`: four blocked time folds with GT inputs and with the published heads (Track-CV tuned per fold) | `outputs/folds/{gt,real}/` |
| `heads` | `head_data_{broadcast,static}.py`, `train_yolo_head.py`, `train_pnlcalib_head.py`: heads fine-tuned per half | `outputs/heads/` |
| `heads_video` | `extract_video.py` with the heads, `link_heads.py`, `build_folds.py heads` | `outputs/folds/heads/` |
| `tracks` | `drop_contacts.py`, `track_phys.py`: contact-free video inputs, Track-Phys | `outputs/folds/{real,heads}_{cv,phys}/` |
| `testsets` | `input_sources.py`, `build_test_sets.py`: mixed test sets per input setting, fixed levels for Table 1 | `outputs/eval/data/` |
| `train` | `jobs.py tune / pfull / base / finetune` → `train_arcprior.py` | `outputs/eval/runs/` |
| `evaluate` | `jobs.py val / select / freeze / test / baselines / fixed / trainsplit` → `predict.py`, `select_prior.py`, `run_grad.py`, `run_snv3d.py` | `outputs/eval/{pred,select,freeze.json}` |
| `report` | `report.py`, `table_heights.py`, `input_quality.py`, `figures.py` | `outputs/eval/report/`, `outputs/figures/` |
| `calibration` | `run_calibration.sh`, `calib_*.py`: PnLCalib MV vs SoccerMaster | `outputs/calibration/` |

The per-fold prior and the whole-flight training share are chosen on validation and written to `outputs/eval/freeze.json` before any test prediction is computed. Where each result of the paper comes from:

| Paper | Source |
|---|---|
| Table 1 (contact heights) | `outputs/eval/report/test/table1.csv`, `table1_gaps.csv` |
| Table 2 (full and partial flights), Tables S2–S5 | `outputs/eval/report/test/` (`table_full.csv`, `table_part.csv`, `uncapped_full.csv`, `map_full.csv`, `gaps_*.csv`, `conditions_*.csv`), `report/train/`, `report/val/` |
| Table S6 (prior chosen per fold) | `outputs/eval/freeze.json` |
| Table S7 (PnLCalib MV vs SoccerMaster) | `outputs/calibration/RESULTS.md` |
| Tables S8–S9 (calibration and tracking from video) | `outputs/eval/report/inputs/` |
| Figures 4–5, S1–S4, S6–S7 | `outputs/figures/` |

## Citation

```bibtex
@misc{chakraborty2026broadcast3dflights,
  title  = {From Broadcast Video to 3D Soccer Ball Flights: A Learned Flight Prior and a Benchmark
            across Input Conditions},
  author = {Chakraborty, Ritabrata},
  year   = {2026},
  note   = {Under review}
}
```

## Acknowledgements

The dataset, flight models and evaluator are those of Grad *et al.*; the ball detectors and the size-based localisation come from SoccerNet-v3D, and the calibration network from PnLCalib. Their code is used unchanged.
