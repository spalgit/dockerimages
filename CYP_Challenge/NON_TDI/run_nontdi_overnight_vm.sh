#!/usr/bin/env bash
# Full run of BOTH trunks, scratch first then CheMeleon.
#
#   bash run_nontdi_overnight_vm.sh            # 9 h + 15 h caps (~24 h)
#   bash run_nontdi_overnight_vm.sh 3 7        # custom: 3 h scratch, 7 h chemeleon
#
# The hours are a SAFETY CAP, not a target. Each run is bounded deterministically
# by MAX_EPOCHS=400 x 15 members, measured at ~7 h (scratch) and ~13 h (CheMeleon)
# on an RTX 4060 laptop, so on comparable hardware both complete with headroom and
# the caps never bite. They exist so that a slower GPU degrades into "most members
# finished" rather than an open-ended run.
#
# Each trunk writes to its own output directory. Completed members are cached, so
# if a cap does bite, just re-run this script to finish the rest.
#
# Run it with `bash`, not `source`. Under SLURM (detected via $SLURM_JOB_ID) it
# stays in the foreground, because a batch script that returns takes its children
# down with it.
set -euo pipefail

H_SCRATCH="${1:-9}"
H_CHEMELEON="${2:-15}"
BASE="$HOME/dockerimages/CYP_Challenge/NON_TDI"
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
SCRIPT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/chemprop_nontdi_8task_ensemble.py"

for f in cyp-challenge-TRAIN_direct_inhibition_with_single_shot_LABELLED.csv \
         cyp-challenge-TEST-BLINDED_ligprepped.csv; do
    [[ -f "$BASE/$f" ]] || { echo "MISSING: $BASE/$f"; exit 1; }
done

run () {                      # $1 = trunk, $2 = hours
    local out="$BASE/outputs_8task_${1}"
    mkdir -p "$out"
    echo "=== $1 trunk, ${2}h cap -> $out  ($(date)) ==="
    python "$SCRIPT" --trunk "$1" --max-hours "$2" \
        --data-dir "$BASE" --output-dir "$out" 2>&1 | tee -a "$out/run.log"
}

main () {
    run scratch   "$H_SCRATCH"
    run chemeleon "$H_CHEMELEON"
    echo "=== both runs finished $(date) ==="
}

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    # Under SLURM the batch script MUST stay in the foreground: when it exits,
    # Slurm tears down the job step and kills every child. Tee so the output
    # lands in both Slurm's .out file and the persistent log.
    echo "SLURM job ${SLURM_JOB_ID} on $(hostname) -- running in foreground"
    main 2>&1 | tee -a "$BASE/overnight.log"
elif [[ -n "${CYP_FOREGROUND:-}" || -n "${CYP_DETACHED:-}" ]]; then
    # Either the user asked for the foreground, or we ARE the detached copy.
    main 2>&1 | tee -a "$BASE/overnight.log"
else
    # Detach properly. `&` alone is not enough: closing the terminal or dropping
    # the SSH connection sends SIGHUP to the whole session. setsid puts the run in
    # a new session with no controlling terminal, nohup ignores SIGHUP, and stdin
    # is closed so it can never block waiting on a terminal that is gone.
    mkdir -p "$BASE"
    CYP_DETACHED=1 setsid nohup bash "$SELF" "$H_SCRATCH" "$H_CHEMELEON" \
        > "$BASE/overnight.log" 2>&1 < /dev/null &
    disown || true
    sleep 1
    echo "detached: pid $(pgrep -f "CYP_DETACHED|chemprop_nontdi_8task" | head -1 || echo "$!")"
    echo "  log    : $BASE/overnight.log"
    echo "  watch  : tail -f $BASE/overnight.log"
    echo "  stop   : pkill -f chemprop_nontdi_8task_ensemble.py"
    echo "You can close this terminal now."
fi
