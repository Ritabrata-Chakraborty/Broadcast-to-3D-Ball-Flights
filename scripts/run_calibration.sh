#!/usr/bin/env bash
# PnLCalib MV vs SoccerMaster calibration study (supplement), one GPU job at a time.
# Usage: scripts/run_calibration.sh inference | tune | repro | speed | solve
set -euo pipefail

# ---------------------------------------------------------------- Paths and parameters
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
ALL="SN23-valid SN23-test SN22-test-center WC14-test TSWC-test ISSIA LP-broadcast LP-static"
SCORED="SN23-test SN22-test-center WC14-test TSWC-test ISSIA LP-broadcast LP-static"
FOCAL="WC14-test TSWC-test ISSIA LP-broadcast LP-static"
CHOSEN="outputs/calibration/chosen_thresholds.json"

# ---------------------------------------------------------------- Utilities
log() { echo "[$(date '+%F %T')] $*"; }
py_for() { if [[ "$1" == soccermaster* ]]; then echo soccermaster; else echo snv3d; fi; }
chosen() { python3 -c "import json;print(json.load(open('$CHOSEN'))['$1']['$2'])"; }

solve_score() {  # model kp line datasets...
    local model="$1" kp="$2" line="$3"
    shift 3
    scripts/py --env snv3d scripts/calib_solve.py --model "$model" --kp-th "$kp" --line-th "$line" \
        --datasets "$@"
    scripts/py --env snv3d scripts/calib_eval.py --model "$model" \
        --tag "$(python3 -c "print(f'kp{$kp:g}_ln{$line:g}')")" --datasets "$@"
}

# ---------------------------------------------------------------- Stages
inference() {
    scripts/py scripts/calib_common.py
    log "SoccerMaster"
    scripts/py --env soccermaster scripts/calib_infer.py --model soccermaster --datasets $ALL
    log "PnLCalib SV"
    scripts/py --env snv3d scripts/calib_infer.py --model pnlcalib --datasets $ALL
    log "PnLCalib MV"
    scripts/py --env snv3d scripts/calib_infer.py --model pnlcalib_mv --datasets $ALL
    log "PnLCalib fine-tuned"
    scripts/py --env snv3d scripts/calib_infer.py --model pnlcalib_ft_wc14 --datasets WC14-test
    scripts/py --env snv3d scripts/calib_infer.py --model pnlcalib_ft_tswc --datasets TSWC-test
}

speed() {
    for model in pnlcalib pnlcalib_mv soccermaster soccermaster_d1; do
        scripts/py --env "$(py_for "$model")" scripts/calib_speed.py --model "$model"
    done
}

solve() {
    for model in pnlcalib pnlcalib_mv soccermaster soccermaster_d1; do
        solve_score "$model" "$(chosen "$model" kp_th)" "$(chosen "$model" line_th)" $SCORED
    done
    # Ablations: inference.py's thresholds, and PnLCalib's fine-tuned weights as a full camera.
    solve_score pnlcalib 0.3434 0.7867 $FOCAL
    solve_score pnlcalib_ft_wc14 0.1274 0.1439 WC14-test
    solve_score pnlcalib_ft_tswc 0.2432 0.8482 TSWC-test
    cat > outputs/calibration/extra_runs.json <<JSON
[{"model": "pnlcalib", "tag": "kp0.3434_ln0.7867", "dataset": "WC14-test", "label": "PnLCalib SV, inference.py thresholds"},
 {"model": "pnlcalib", "tag": "kp0.3434_ln0.7867", "dataset": "TSWC-test", "label": "PnLCalib SV, inference.py thresholds"},
 {"model": "pnlcalib", "tag": "kp0.3434_ln0.7867", "dataset": "ISSIA", "label": "PnLCalib SV, inference.py thresholds"},
 {"model": "pnlcalib", "tag": "kp0.3434_ln0.7867", "dataset": "LP-broadcast", "label": "PnLCalib SV, inference.py thresholds"},
 {"model": "pnlcalib", "tag": "kp0.3434_ln0.7867", "dataset": "LP-static", "label": "PnLCalib SV, inference.py thresholds"},
 {"model": "pnlcalib_ft_wc14", "tag": "kp0.1274_ln0.1439", "dataset": "WC14-test", "label": "PnLCalib fine-tuned WC14"},
 {"model": "pnlcalib_ft_tswc", "tag": "kp0.2432_ln0.8482", "dataset": "TSWC-test", "label": "PnLCalib fine-tuned TSWC"}]
JSON
}

# ---------------------------------------------------------------- Main
main() {
    case "${1:-}" in
        inference) inference ;;
        tune) scripts/py --env snv3d scripts/calib_tune.py ;;
        repro) scripts/py --env snv3d scripts/calib_repro.py ;;
        speed) speed ;;
        solve) solve ;;
        *) echo "usage: $0 inference|tune|repro|speed|solve" >&2; exit 2 ;;
    esac
    log "done: ${1}"
}

main "$@"
