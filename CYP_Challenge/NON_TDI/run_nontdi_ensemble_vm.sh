#!/usr/bin/env bash
# Launch the 8-task ChemProp ensemble on the VM.
#   bash run_nontdi_ensemble_vm.sh            # full run, logs to nohup_ensemble.log
# Re-running after an interruption skips members that already finished.
set -euo pipefail

BASE="$HOME/dockerimages/CYP_Challenge/NON_TDI"
OUT="$BASE/outputs_8task_ensemble"
mkdir -p "$OUT"

for f in cyp-challenge-TRAIN_direct_inhibition_with_single_shot_LABELLED.csv \
         cyp-challenge-TEST-BLINDED_ligprepped.csv; do
    [[ -f "$BASE/$f" ]] || { echo "MISSING: $BASE/$f"; exit 1; }
done

nohup python chemprop_nontdi_8task_ensemble.py \
    --data-dir "$BASE" --output-dir "$OUT" \
    > "$OUT/run.log" 2>&1 &
echo "started pid $! -- tail -f $OUT/run.log"
