#!/usr/bin/env bash
# Every stage of the paper's results, in order: flights, video inputs, heads, folds, test sets,
# training, selection and freeze, test predictions, baselines, tables and figures.
# Usage: scripts/run_pipeline.sh <stage> [<stage> ...]   (stages are listed in main below)
# Each stage expects the outputs of the stages before it; existing outputs are never overwritten.
set -euo pipefail

# ---------------------------------------------------------------- Paths and parameters
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
JOBS="${JOBS:-4}"                         # parallel jobs for command lists
WORKERS="${WORKERS:-8}"                   # CPU workers for the dataset builders
OUT="outputs"

# ---------------------------------------------------------------- Utilities
log() { echo "[$(date '+%F %T')] $*"; }
gpu() { scripts/py --env snv3d "$@"; }
cpu() { scripts/py "$@"; }
queue() {   # stage of jobs.py -> run its command list
    cpu scripts/jobs.py "$1"
    scripts/run_list.sh "$OUT/eval/queue/$1.txt" "$JOBS"
}

# ---------------------------------------------------------------- Stages
flights() {          # airborne flights (apex >= 1 m), splits and oracle fits
    cpu scripts/build_flights.py --workers "$WORKERS"
}

video() {            # published YOLO11 + PnLCalib MV; SNv3D-size's size-optimised boxes
    gpu scripts/extract_video.py
    gpu scripts/extract_video.py --no-calib --yolo weights/snv3d/yolo-sn-ball-opt.pt \
        --out "$OUT/extract/snv3d_opt"
}

folds() {            # GT and published-head datasets of the four folds (Track-CV tuned per fold)
    cpu scripts/build_folds.py gt --workers "$WORKERS"
    cpu scripts/build_folds.py real --workers "$WORKERS"
}

heads() {            # fine-tuned YOLO Detect head and PnLCalib keypoint head, per half
    gpu scripts/head_data_broadcast.py
    for fold in a b; do
        gpu scripts/train_yolo_head.py --fold "$fold"
        gpu scripts/train_pnlcalib_head.py --fold "$fold"
    done
    for half in half_1 half_2; do
        gpu scripts/head_data_static.py --half "$half"
        gpu scripts/train_yolo_head.py --fold "$half" --batch 4 \
            --data "$OUT/heads/static/data/$half/data.yaml" --out "$OUT/heads/static/yolo"
        gpu scripts/train_pnlcalib_head.py --fold "$half" \
            --data "$OUT/heads/static/data/$half" --out "$OUT/heads/static/pnlcalib"
    done
}

heads_video() {      # extraction with the fine-tuned heads; each half's heads in one folder
    for fold in a b; do
        gpu scripts/extract_video.py --views 0 \
            --yolo "$OUT/heads/broadcast/yolo/$fold/weights/best.pt" \
            --kp-head "$OUT/heads/broadcast/pnlcalib/$fold/MV_kp_head" \
            --out "$OUT/extract/heads_broadcast_$fold"
    done
    for half in half_1 half_2; do
        gpu scripts/extract_video.py --views 1 2 3 4 5 \
            --yolo "$OUT/heads/static/yolo/$half/weights/best.pt" \
            --kp-head "$OUT/heads/static/pnlcalib/$half/MV_kp_head" \
            --out "$OUT/extract/heads_static_$half"
    done
    cpu scripts/link_heads.py
    cpu scripts/build_folds.py heads --head-map "$OUT/extract/heads_map.json" --workers "$WORKERS"
}

tracks() {           # contact-free video inputs; Track-Phys (radius tuned on the published heads)
    cpu scripts/drop_contacts.py real --out real_cv
    cpu scripts/drop_contacts.py heads --out heads_cv
    gpu scripts/track_phys.py real_cv --out real_phys --no-contacts
    gpu scripts/track_phys.py heads_cv --out heads_phys --no-contacts --radius-from real_phys
}

testsets() {         # input sources per setting, mixed test sets, and Table 1's fixed levels
    cpu scripts/input_sources.py
    local src="--source-root $OUT/eval/data"
    gpu scripts/build_test_sets.py gt_src --name gt $src --ball3d-anchors dev
    gpu scripts/build_test_sets.py real_src --name real $src --ball3d-anchors ground
    gpu scripts/build_test_sets.py heads_src --name heads $src --ball3d-anchors ground
    for level in correct wrong none; do
        gpu scripts/build_test_sets.py gt_src --name gt $src --ball3d-anchors dev --kinds full \
            --fixed "$level"
    done
}

train() {            # whole-flight share on fold 1, Base for every fold and seed, fine-tunes
    queue tune
    cpu scripts/jobs.py pfull
    queue base
    queue finetune
}

evaluate() {         # validation, prior selection, freeze, then the single test read
    queue val
    queue select
    cpu scripts/jobs.py freeze
    queue test
    queue baselines
    queue fixed
    queue trainsplit
}

report() {           # tables and figures of the paper (figures with frames: never commit)
    cpu scripts/report.py --split test
    cpu scripts/report.py --split val
    cpu scripts/table_heights.py
    gpu scripts/input_quality.py
    cpu scripts/figures.py --frames
}

calibration() {      # PnLCalib MV vs SoccerMaster (supplement); see scripts/run_calibration.sh
    for stage in inference tune speed repro solve; do
        scripts/run_calibration.sh "$stage"
    done
    gpu scripts/calib_report.py
}

# ---------------------------------------------------------------- Main
main() {
    local stages=(flights video folds heads heads_video tracks testsets train evaluate report
                  calibration)
    if [[ $# -eq 0 ]]; then
        echo "usage: $0 <stage>...  (stages: ${stages[*]})"
        exit 1
    fi
    for stage in "$@"; do
        [[ " ${stages[*]} " == *" $stage "* ]] || { echo "unknown stage: $stage"; exit 1; }
        log "stage $stage"
        "$stage"
    done
    log "done: $*"
}

main "$@"
