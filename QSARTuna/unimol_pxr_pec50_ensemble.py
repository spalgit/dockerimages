"""
Uni-Mol PXR pEC50 regression — multi-seed ensemble.

Trains N independent Uni-Mol models on the PXR training set and averages
their predictions on the external test set.

Key differences from the ChemProp scripts:
  - 3D molecular representation (ETKDG conformers) instead of 2D graph
  - Pretrained on 209M molecules from PubChem — fine-tuned here on PXR data
  - Each MolTrain call runs 5-fold scaffold CV internally and saves one
    model checkpoint per fold; MolPredict averages across all 5 fold models
  - Running N seeds gives an N×5 ensemble of diverse models

Note on weighting:
  unimol_tools does not expose sample weights in its public API, so all
  training compounds are treated with equal weight here. The 3D pretrained
  representation typically compensates for this vs. the weighted ChemProp
  models, but the ChemProp+counter-weight model is still worth including
  in the final blended submission.

Usage:
    conda activate chemprop
    python ~/dockerimages/QSARTuna/unimol_pxr_pec50_ensemble.py
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

from unimol_tools import MolPredict, MolTrain

# ── Paths ──────────────────────────────────────────────────────────────────────
TRAIN_PATH = Path(
    "/home/spal/dockerimages/QSARTuna/PXR_For_QSARTuna_Modelling/"
    "processed_Openadmet_REAL_PXR_train_AND_test_main_with_side_info_"
    "AND_counter_screen_weighted.csv"
)
TEST_PATH = (
    Path.home()
    / "dockerimages/QSARTuna/PXR/Prediction_OpenAdmet_ChemProp_Only_OpenADMET_Data.csv"
)
ENSEMBLE_BASE_DIR = Path.home() / "unimol_pxr_ensemble"
OUTPUT_PREDS      = Path.home() / "unimol_pxr_ensemble_test_predictions.csv"

# ── Column names ───────────────────────────────────────────────────────────────
TRAIN_SMILES_COL = "SMILES"
TRAIN_TARGET_COL = "pEC50"
TEST_SMILES_COL  = "SMILES"
TEST_TARGET_COL  = "pEC50"
TEST_NAME_COL    = "Molecule Name"

# ── Training settings ──────────────────────────────────────────────────────────
EPOCHS         = 40
LEARNING_RATE  = 1e-4
BATCH_SIZE     = 16
EARLY_STOPPING = 15     # patience in epochs
KFOLD          = 5      # folds for internal scaffold CV
ENSEMBLE_SEEDS = [42, 123, 456, 789, 1337]


# ── Utilities ──────────────────────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def report_metrics(actual: np.ndarray, predicted: np.ndarray, label: str = "") -> dict:
    mae  = float(np.mean(np.abs(actual - predicted)))
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2   = float(1.0 - ss_res / ss_tot)
    range_actual = actual.max() - actual.min()
    rae  = float(mae / (range_actual / 2.0)) if range_actual > 0 else float("nan")
    rho, _ = stats.spearmanr(actual, predicted)
    tau, _ = stats.kendalltau(actual, predicted)
    tag = f"  [{label}]" if label else ""
    print(
        f"{tag}  MAE={mae:.4f}  RMSE={rmse:.4f}  RAE={rae:.4f}  "
        f"R²={r2:.4f}  Spearman={rho:.4f}  Kendall={tau:.4f}"
    )
    return dict(mae=mae, rmse=rmse, rae=rae, r2=r2, spearman=rho, kendall=tau)


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — Load data
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nLoading training data:\n  {TRAIN_PATH}")
df_train = pd.read_csv(TRAIN_PATH)
print(f"  {len(df_train)} rows  |  "
      f"pEC50 range: {df_train[TRAIN_TARGET_COL].min():.2f} – "
      f"{df_train[TRAIN_TARGET_COL].max():.2f}")

print(f"\nLoading test data:\n  {TEST_PATH}")
df_test = pd.read_csv(TEST_PATH)
print(f"  {len(df_test)} rows")

# unimol_tools expects a dict with 'SMILES' and 'target' keys
train_data = {
    "SMILES": df_train[TRAIN_SMILES_COL].tolist(),
    "target": df_train[TRAIN_TARGET_COL].tolist(),
}

# Pass true test labels so MolPredict can report metrics internally
test_data = {
    "SMILES": df_test[TEST_SMILES_COL].tolist(),
    "target": df_test[TEST_TARGET_COL].tolist(),
}

test_targets = df_test[TEST_TARGET_COL].values
test_names   = df_test[TEST_NAME_COL].values

# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — Train ensemble members
# ══════════════════════════════════════════════════════════════════════════════
ENSEMBLE_BASE_DIR.mkdir(parents=True, exist_ok=True)
all_test_preds    = []
per_model_metrics = []

for i, seed in enumerate(ENSEMBLE_SEEDS):
    save_path = str(ENSEMBLE_BASE_DIR / f"model_seed{seed}")
    print(f"\n{'='*70}")
    print(f"Ensemble member {i + 1}/{len(ENSEMBLE_SEEDS)}  |  seed={seed}")
    print(f"{'='*70}")

    # Check if already trained — skip if checkpoints exist
    existing = list(Path(save_path).glob("model_*.pth")) if Path(save_path).exists() else []
    if len(existing) == KFOLD:
        print(f"  Found {len(existing)} existing checkpoints — skipping training.")
    else:
        set_seed(seed)
        clf = MolTrain(
            task="regression",
            data_type="molecule",
            epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            early_stopping=EARLY_STOPPING,
            metrics="mse",           # used for early stopping; MAE computed below
            split="scaffold",        # scaffold-based K-fold — no scaffold leakage
            kfold=KFOLD,
            save_path=save_path,
            smiles_col="SMILES",
            target_normalize="auto", # StandardScaler on targets during training
            use_cuda=True,
            use_amp=True,            # mixed precision — faster on modern GPUs
            conf_cache_level=1,      # cache conformers to disk; speeds up re-runs
        )
        clf.fit(train_data)
        print(f"\n  Training complete. CV predictions available.")

    # Predict on test set using all K fold models (averaged internally)
    predictor = MolPredict(load_model=save_path)
    preds = predictor.predict(test_data, metrics="mae")
    preds_flat = np.array(preds).flatten()
    all_test_preds.append(preds_flat)

    m = report_metrics(test_targets, preds_flat, label=f"seed={seed}")
    m["seed"] = seed
    per_model_metrics.append(m)

# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — Ensemble prediction (mean across all seeds)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Ensemble results")
print(f"{'='*70}")

all_preds_array  = np.stack(all_test_preds, axis=0)   # (n_seeds, n_test)
ensemble_preds   = all_preds_array.mean(axis=0)
ensemble_metrics = report_metrics(test_targets, ensemble_preds, label="ENSEMBLE")

# ── Save predictions ───────────────────────────────────────────────────────────
pred_cols = {f"pEC50_seed{s}": p for s, p in zip(ENSEMBLE_SEEDS, all_test_preds)}
df_out = pd.DataFrame({
    "Molecule Name":  test_names,
    "SMILES":         df_test[TEST_SMILES_COL].values,
    "pEC50_actual":   test_targets,
    "pEC50_ensemble": ensemble_preds,
    "residual":       test_targets - ensemble_preds,
    **pred_cols,
})
df_out.to_csv(OUTPUT_PREDS, index=False)
print(f"\nPredictions saved to {OUTPUT_PREDS}")

# ── Per-model summary ──────────────────────────────────────────────────────────
df_members = pd.DataFrame(per_model_metrics)
print("\nPer-member test-set metrics:")
print(df_members.to_string(index=False))
print(f"\nMean individual MAE : {df_members['mae'].mean():.4f} ± {df_members['mae'].std():.4f}")
print(f"Ensemble MAE        : {ensemble_metrics['mae']:.4f}")
print(f"Gain vs mean indiv. : {df_members['mae'].mean() - ensemble_metrics['mae']:.4f}")

m = ensemble_metrics
print(
    f"\n[Leaderboard format]  "
    f"MAE={m['mae']:.4f}  RAE={m['rae']:.4f}  "
    f"R²={m['r2']:.4f}  Spearman={m['spearman']:.4f}  Kendall={m['kendall']:.4f}"
)
