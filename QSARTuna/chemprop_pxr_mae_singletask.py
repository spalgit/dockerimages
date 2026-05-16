"""
ChemProp single-task PXR pEC50 regression — MAE training criterion.

Identical architecture to the rank-50 OpenAdmet submission
(PXR_chemprop_SingleTask_All_data_Claude) but trained with MAE loss
instead of MSE, and for up to 200 epochs to ensure full convergence.

Architecture (fixed — no grid search):
  BondMessagePassing : depth=3, d_h=300
  MeanAggregation
  RegressionFFN      : hidden_dim=300, n_layers=2, dropout=0.2
  batch_norm=True
  criterion          : MAE   ← key change vs rank-50 model

Pipeline:
  1. 5-fold stratified CV — records best epoch per fold, estimates MAE.
  2. Final model trained on ALL training data for mean_best_epoch × 1.1
     epochs (no val set; early stopping not needed because duration is
     derived from CV).
  3. Predict on competition test set; write submission CSV.

Training data : processed_Openadmet_REAL_ChemBL_PXR_train_AND_test_main_
                with_side_info_AND_counter_screen.csv  (~4 384 compounds)
Test set      : ~/dockerimages/QSARTuna/PXR/test.csv  (513 molecules, no labels)

Usage:
    conda activate chemprop
    cd ~/dockerimages/QSARTuna
    python chemprop_pxr_mae_singletask.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightning import pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from scipy import stats
from sklearn.model_selection import StratifiedKFold

from chemprop import data, featurizers, models, nn

# ── Paths ──────────────────────────────────────────────────────────────────────
TRAIN_PATH = Path(
    "/home/spal/dockerimages/QSARTuna/PXR_For_QSARTuna_Modelling/"
    "processed_Openadmet_REAL_ChemBL_PXR_train_AND_test_main_with_side_info_"
    "AND_counter_screen.csv"
)
TEST_PATH      = Path.home() / "dockerimages/QSARTuna/PXR/test.csv"
MODEL_PKL_PATH = Path.home() / "pxr_chemprop_mae_singletask_final.pkl"
CV_RESULTS_PATH = Path.home() / "pxr_chemprop_mae_singletask_cv.csv"
SUBMISSION_PATH = Path.home() / "OpenAdmet/Submission_ChemProp_MAE_SingleTask.csv"

TRAIN_SMILES_COL = "SMILES"
TRAIN_TARGET_COL = "pEC50"
TEST_SMILES_COL  = "SMILES"
TEST_NAME_COL    = "Molecule Name"

# ── Architecture (mirrors rank-50 OpenAdmet model) ─────────────────────────────
MP_DEPTH      = 3
MP_HIDDEN_DIM = 300
FFN_HIDDEN_DIM  = 300
FFN_NUM_LAYERS  = 2      # intended value; YAML key was silently ignored by OpenAdmet
DROPOUT         = 0.2
BATCH_NORM      = True

# ── Training ───────────────────────────────────────────────────────────────────
N_FOLDS       = 5
N_STRATA_BINS = 5
CV_MAX_EPOCHS = 200
CV_PATIENCE   = 30       # generous patience to allow MAE loss to converge
FINAL_EPOCH_BUFFER = 1.1 # multiply mean CV best-epoch by this for final run
NUM_WORKERS   = 0

# Moderate LR schedule — slower than default to suit longer runs
INIT_LR  = 1e-4
MAX_LR   = 5e-4
FINAL_LR = 1e-5


# ── Best-epoch tracker ─────────────────────────────────────────────────────────
class BestEpochTracker(pl.Callback):
    def __init__(self):
        self.best_val_loss = float("inf")
        self.best_epoch    = 0

    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = float(trainer.callback_metrics.get("val_loss", float("inf")))
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch    = trainer.current_epoch


# ── Model factory ──────────────────────────────────────────────────────────────
def build_mpnn(target_scaler) -> models.MPNN:
    feat = featurizers.SimpleMoleculeMolGraphFeaturizer()
    mp   = nn.BondMessagePassing(
               d_v=feat.atom_fdim,
               d_e=feat.bond_fdim,
               depth=MP_DEPTH,
               d_h=MP_HIDDEN_DIM,
           )
    agg  = nn.MeanAggregation()
    output_transform = nn.UnscaleTransform.from_standard_scaler(target_scaler)
    ffn  = nn.RegressionFFN(
               input_dim=mp.output_dim,
               hidden_dim=FFN_HIDDEN_DIM,
               n_layers=FFN_NUM_LAYERS,
               dropout=DROPOUT,
               criterion=nn.metrics.MAE(),   # ← MAE training loss
               output_transform=output_transform,
           )
    return models.MPNN(
        mp, agg, ffn,
        batch_norm=BATCH_NORM,
        metrics=[nn.metrics.MAE(), nn.metrics.RMSE()],
        init_lr=INIT_LR,
        max_lr=MAX_LR,
        final_lr=FINAL_LR,
    )


# ── Data loading ───────────────────────────────────────────────────────────────
def load_datapoints(smiles_list, targets_list=None):
    points, skipped = [], 0
    iterable = zip(smiles_list, targets_list) if targets_list is not None \
               else ((s, None) for s in smiles_list)
    for smi, y in iterable:
        y_arr = np.array([[y]], dtype=float) if y is not None else np.array([[np.nan]])
        dp = data.MoleculeDatapoint.from_smi(smi, y_arr)
        if dp.mol is None:
            skipped += 1
            continue
        points.append(dp)
    if skipped:
        print(f"  Skipped {skipped} unparseable SMILES.")
    return points


# ── CV fold ────────────────────────────────────────────────────────────────────
def run_fold(fold_train, fold_val, featurizer, max_epochs, patience):
    train_dset = data.MoleculeDataset(fold_train, featurizer)
    scaler     = train_dset.normalize_targets()

    val_dset   = data.MoleculeDataset(fold_val, featurizer)
    val_dset.normalize_targets(scaler)

    train_loader = data.build_dataloader(train_dset, num_workers=NUM_WORKERS)
    val_loader   = data.build_dataloader(val_dset,   num_workers=NUM_WORKERS, shuffle=False)

    mpnn          = build_mpnn(scaler)
    epoch_tracker = BestEpochTracker()

    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        accelerator="auto",
        devices=1,
        max_epochs=max_epochs,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=patience, mode="min"),
            epoch_tracker,
        ],
    )
    trainer.fit(mpnn, train_loader, val_loader)

    # Collect predictions on the validation fold for metric reporting
    mpnn.eval()
    raw = trainer.predict(mpnn, val_loader)
    preds = torch.cat(raw).numpy().flatten()
    actuals = np.array([dp.y[0] for dp in fold_val])

    mae  = float(np.mean(np.abs(actuals - preds)))
    rmse = float(np.sqrt(np.mean((actuals - preds) ** 2)))
    r2   = float(1 - np.sum((actuals - preds)**2) / np.sum((actuals - actuals.mean())**2))
    sp   = float(stats.spearmanr(actuals, preds).statistic)

    return epoch_tracker.best_val_loss, epoch_tracker.best_epoch, mae, rmse, r2, sp


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — Load training data
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nLoading training data:\n  {TRAIN_PATH}")
df_train    = pd.read_csv(TRAIN_PATH)
all_points  = load_datapoints(
    df_train[TRAIN_SMILES_COL].values,
    df_train[TRAIN_TARGET_COL].values,
)
pec50_vals = np.array([dp.y[0] for dp in all_points])
print(f"  Usable compounds : {len(all_points)}")
print(f"  pEC50 range      : {pec50_vals.min():.2f} – {pec50_vals.max():.2f}  "
      f"(mean {pec50_vals.mean():.2f})")

# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — Load competition test set (no labels)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nLoading competition test set:\n  {TEST_PATH}")
df_test     = pd.read_csv(TEST_PATH)
test_points = load_datapoints(df_test[TEST_SMILES_COL].values)
test_names  = df_test[TEST_NAME_COL].values
print(f"  Test molecules   : {len(test_points)}")

featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — 5-fold stratified CV (skip if results file already exists)
# ══════════════════════════════════════════════════════════════════════════════
if CV_RESULTS_PATH.exists():
    print(f"\nCV results found — loading from {CV_RESULTS_PATH}")
    df_cv = pd.read_csv(CV_RESULTS_PATH)
else:
    strata = pd.qcut(pec50_vals, q=N_STRATA_BINS, labels=False,
                     duplicates="drop").astype(int)
    print(f"\nStratification ({N_STRATA_BINS} equal-frequency bins):")
    for b in range(strata.max() + 1):
        m = strata == b
        print(f"  Bin {b}: n={m.sum():4d}  "
              f"pEC50 {pec50_vals[m].min():.2f}–{pec50_vals[m].max():.2f}")

    print(f"\n{'='*60}")
    print(f"5-fold stratified CV  |  max_epochs={CV_MAX_EPOCHS}  patience={CV_PATIENCE}")
    print(f"Architecture: depth={MP_DEPTH}  mp_hidden={MP_HIDDEN_DIM}  "
          f"ffn_hidden={FFN_HIDDEN_DIM}  ffn_layers={FFN_NUM_LAYERS}  "
          f"dropout={DROPOUT}  criterion=MAE")
    print(f"{'='*60}")

    skf     = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    indices = np.arange(len(all_points))
    rows    = []

    for fold_num, (tr_idx, va_idx) in enumerate(skf.split(indices, strata)):
        fold_train = [all_points[i] for i in tr_idx]
        fold_val   = [all_points[i] for i in va_idx]

        val_loss, best_epoch, mae, rmse, r2, sp = run_fold(
            fold_train, fold_val, featurizer,
            max_epochs=CV_MAX_EPOCHS, patience=CV_PATIENCE,
        )
        print(f"  Fold {fold_num+1}: best_epoch={best_epoch:3d}  "
              f"MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}  Spearman={sp:.4f}")
        rows.append(dict(fold=fold_num+1, best_epoch=best_epoch,
                         mae=mae, rmse=rmse, r2=r2, spearman=sp))

    df_cv = pd.DataFrame(rows)
    df_cv.to_csv(CV_RESULTS_PATH, index=False)
    print(f"\nCV results saved to {CV_RESULTS_PATH}")

print(f"\nCV summary:")
print(df_cv.to_string(index=False))
print(f"\n  Mean MAE      : {df_cv['mae'].mean():.4f} ± {df_cv['mae'].std():.4f}")
print(f"  Mean RMSE     : {df_cv['rmse'].mean():.4f} ± {df_cv['rmse'].std():.4f}")
print(f"  Mean R2       : {df_cv['r2'].mean():.4f} ± {df_cv['r2'].std():.4f}")
print(f"  Mean Spearman : {df_cv['spearman'].mean():.4f} ± {df_cv['spearman'].std():.4f}")

mean_best_epoch = df_cv["best_epoch"].mean()
final_epochs    = max(int(mean_best_epoch * FINAL_EPOCH_BUFFER), 10)
print(f"\n  Mean best epoch : {mean_best_epoch:.1f}  →  final training epochs: {final_epochs}")

# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — Final model: all training data, no validation set
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"Training final model on all {len(all_points)} compounds for {final_epochs} epochs")
print(f"{'='*60}")

all_train_dset = data.MoleculeDataset(all_points, featurizer)
final_scaler   = all_train_dset.normalize_targets()
final_loader   = data.build_dataloader(all_train_dset, num_workers=NUM_WORKERS)

final_mpnn = build_mpnn(final_scaler)
print("\nArchitecture:")
print(final_mpnn)

final_trainer = pl.Trainer(
    logger=False,
    enable_checkpointing=False,
    enable_progress_bar=True,
    accelerator="auto",
    devices=1,
    max_epochs=final_epochs,
)
final_trainer.fit(final_mpnn, final_loader)

# ══════════════════════════════════════════════════════════════════════════════
# Step 5 — Save model
# ══════════════════════════════════════════════════════════════════════════════
torch.save(final_mpnn, MODEL_PKL_PATH)
print(f"\nModel saved to {MODEL_PKL_PATH}")

# ══════════════════════════════════════════════════════════════════════════════
# Step 6 — Predict on competition test set and write submission
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nPredicting on {len(test_points)} competition test molecules...")
final_mpnn.eval()

test_dset   = data.MoleculeDataset(test_points, featurizer)
test_loader = data.build_dataloader(test_dset, num_workers=NUM_WORKERS, shuffle=False)

raw_preds = final_trainer.predict(final_mpnn, test_loader)
preds     = torch.cat(raw_preds).numpy().flatten()

df_submission = pd.DataFrame({
    "Molecule Name": test_names[: len(test_points)],
    "SMILES":        df_test[TEST_SMILES_COL].values[: len(test_points)],
    "pEC50":         preds,
})
df_submission.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission saved to {SUBMISSION_PATH}")
print(f"\nFirst 5 predictions:")
print(df_submission.head().to_string(index=False))
