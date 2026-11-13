#!/usr/bin/env bash
# Run each line of a command file, a few at a time; one log per line, a summary log at the end.
# Usage: scripts/run_list.sh <command file> [parallel jobs]
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIST="${1:?command file}"
PARALLEL="${2:-2}"
LOGS="${LOG_ROOT:-$REPO/outputs/logs}/$(basename "$LIST" .txt)"   # LOG_ROOT overrides
mkdir -p "$LOGS"
cd "$REPO"
nl -ba -nrz -w3 "$LIST" | xargs -P "$PARALLEL" -d '\n' -I {} bash -c '
    line="{}"; id="${line%%$'"'"'\t'"'"'*}"; cmd="${line#*$'"'"'\t'"'"'}"
    if bash -c "$cmd" > "'"$LOGS"'/$id.log" 2>&1; then echo "$(date "+%F %T") done $id"
    else echo "$(date "+%F %T") FAILED $id: $cmd"; fi'
echo "$(date '+%F %T') list finished"
