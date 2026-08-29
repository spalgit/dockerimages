#!/bin/bash
# =============================================================================
# tdi_ensemble_potency_gated.py -- ChemProp + XGBoost potency-gated TDI ensemble
# with the Optuna search enabled (it is on by default; the flags below size it).
#
#   sbatch submit_tdi_gated_hpo.slurm
#
# Smoke-test first -- ~10 min/isoform, exercises every code path:
#   python -u tdi_ensemble_potency_gated.py --quick --out-dir /tmp/gated_quick
# =============================================================================
#SBATCH --job-name=tdi_gated_hpo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1                 # ChemProp needs a GPU; check: sinfo -o "%P %G"
#SBATCH --partition=LocalQ           # must be a partition that actually has GPUs
#SBATCH --time=36:00:00
#SBATCH --output=tdi_gated_hpo_%j.out
#SBATCH --error=tdi_gated_hpo_%j.err

# -e/-o pipefail, but NOT -u: `conda activate` trips on unset internal variables.
set -eo pipefail

# --- environment -------------------------------------------------------------
# module purge
# module load miniconda3

# Batch shells are non-interactive, so conda must be sourced before `activate`.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate chemprop

cd "${SLURM_SUBMIT_DIR}"

# --- paths -------------------------------------------------------------------
# Config.data_dir defaults to $CYP_DATA_DIR, so export it rather than passing
# --data-dir: if the variable is empty, argparse eats the *next* flag as its value.
export CYP_DATA_DIR="${SLURM_SUBMIT_DIR}/data"        # <-- adjust to your data dir
OUT="${SLURM_SUBMIT_DIR}/TDI/output/gated_hpo"
mkdir -p "$OUT"

[ -d "$CYP_DATA_DIR" ] || { echo "CYP_DATA_DIR not found: $CYP_DATA_DIR" >&2; exit 1; }

# --- sanity checks (cheap, and they save 24h of CPU-only training) -----------
echo "host: $(hostname)   job: ${SLURM_JOB_ID}   started: $(date)"
nvidia-smi || echo "WARNING: nvidia-smi failed -- the run will fall back to CPU"
python -c "import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())"
python -c "import optuna, xgboost; print('optuna', optuna.__version__, 'xgboost', xgboost.__version__)"

# --- run ---------------------------------------------------------------------
# --n-trials 40   Optuna trials (default 20) over the ChemProp search space:
#                 depth, message_hidden_dim, ffn_hidden_dim, ffn_n_layers, dropout,
#                 max_lr, lr ratios, warmup_epochs, aggregation, batch_norm,
#                 aux_weight, gate_weight
# --hpo-folds 3   scaffold folds scored per trial (keeps each trial affordable)
# Do NOT pass --n-trials 0: study.best_trial is read unconditionally and would raise.
python -u tdi_ensemble_potency_gated.py \
    --isoform both \
    --out-dir "$OUT" \
    --n-trials 40 \
    --hpo-folds 3 \
    --folds 5 \
    --max-epochs 60

echo "finished: $(date)"

# =============================================================================
# If 24h is tight, run the two isoforms as independent array tasks instead --
# add  #SBATCH --array=0-1  above and replace the python call with:
#
#   ISOS=(CYP2D6 CYP3A4)
#   python -u tdi_ensemble_potency_gated.py \
#       --isoform "${ISOS[$SLURM_ARRAY_TASK_ID]}" \
#       --out-dir "$OUT" --n-trials 40 --hpo-folds 3 --folds 5 --max-epochs 60
#
# If a run times out, restart with  --reuse-cache-from "$OUT"  to copy the
# feature/stack caches and skip re-featurisation.
# =============================================================================
