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
set -euo pipefail

H_SCRATCH="${1:-9}"
H_CHEMELEON="${2:-15}"
BASE="$HOME/dockerimages/CYP_Challenge/NON_TDI"
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

{
    run scratch   "$H_SCRATCH"
    run chemeleon "$H_CHEMELEON"
    echo "=== both runs finished $(date) ==="
} > "$BASE/overnight.log" 2>&1 &

echo "started pid $! -- tail -f $BASE/overnight.log"
