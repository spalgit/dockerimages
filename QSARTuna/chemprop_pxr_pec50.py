"""
CheMeleon-finetuned Chemprop model for PXR pEC50 regression.

Pipeline:
  1. 5-fold STRATIFIED CV grid search to select FFN hyperparameters.
       Stratification bins the continuous pEC50 target into quantile-based
       groups so that every fold's validation set reflects the full activity
       distribution — important for PXR data which spans inactive (<4) through
       highly active (>7) compounds.
  2. Retrain a final model on ALL training data for the average number of
       epochs that gave the best validation loss across folds.
  3. Save final model as a .pkl file (via torch.save).
  4. Predict on an external test set and report performance metrics.

Usage:
    conda activate chemprop
    python ~/chemprop_pxr_pec50.py

To load the saved model for inference in another script:
    import torch
    from chemprop import data, featurizers
    model = torch.load("~/pxr_chemeleon_final.pkl", weights_only=False)
    model.eval()
"""

import itertools
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
import torch
from lightning import pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold   # replaces plain KFold

from chemprop import data, featurizers, models, nn

# ── Configuration ─────────────────────────────────────────────────────────────
TRAIN_PATH = Path(
    "/home/spal/dockerimages/QSARTuna/PXR_For_QSARTuna_Modelling/"
    "processed_Openadmet_REAL_PXR_train_AND_test_main_with_side_info_"
    "AND_counter_screen_weighted.csv"
)
TEST_PATH = Path(
    "/home/spal/OpenAdmet/Prediction_OpenAdmet_ChemProp_Only_OpenADMET_Data.csv"
)
CHEMELEON_PATH  = Path.home() / "chemeleon_mp.pt"
MODEL_PKL_PATH  = Path.home() / "pxr_chemeleon_final.pkl"
CV_RESULTS_PATH = Path.home() / "pxr_cv_results.csv"
OUTPUT_PREDS    = Path.home() / "pxr_external_test_predictions.csv"

TRAIN_SMILES_COL = "SMILES"
TRAIN_TARGET_COL = "pEC50"
TEST_SMILES_COL  = "SMILES"
TEST_TARGET_COL  = "pEC50"
TEST_NAME_COL    = "Molecule Name"

N_FOLDS       = 5
# Number of quantile bins used to stratify pEC50 for fold assignment.
# Using N_FOLDS bins means each bin contributes one compound per fold on
# average, giving balanced activity coverage across all splits.
N_STRATA_BINS = N_FOLDS
CV_MAX_EPOCHS = 30   # upper bound per fold; early stopping usually fires first
CV_PATIENCE   = 5    # consecutive epochs without val_loss improvement before stopping
NUM_WORKERS   = 0

# Hyperparameter grid — only FFN params are tuned.
# CheMeleon's BondMessagePassing architecture is fixed by the checkpoint.
PARAM_GRID = {
    "ffn_hidden_dim": [300, 512],   # width of the hidden FFN layers
    "ffn_n_layers":   [2, 3],       # depth of the FFN (not the message-passing depth)
    "dropout":        [0.0, 0.2],   # dropout applied after each FFN hidden layer
}


# ── Custom callback — records the epoch with the lowest validation loss ────────
class BestEpochTracker(pl.Callback):
    """
    Tracks which epoch produced the lowest val_loss.
    Used after CV to set the final model's training budget:
      final_epochs = mean(best_epochs across folds) × 1.1
    This avoids needing a validation set when retraining on all data.
    """

    def __init__(self):
        self.best_val_loss = float("inf")
        self.best_epoch = 0

    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = float(trainer.callback_metrics.get("val_loss", float("inf")))
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = trainer.current_epoch


# ── Helpers ───────────────────────────────────────────────────────────────────
def build_datapoints(smiles_arr, targets_arr):
    """
    Convert parallel SMILES and target arrays into MoleculeDatapoint objects.
    Compounds whose SMILES RDKit cannot parse are silently skipped.
    """
    points, skipped = [], 0
    for smi, y in zip(smiles_arr, targets_arr):
        dp = data.MoleculeDatapoint.from_smi(smi, y)
        if dp.mol is None:
            skipped += 1
            continue
        points.append(dp)
    if skipped:
        print(f"  Skipped {skipped} unparseable SMILES.")
    return points


def build_mpnn(chemeleon_hyper, chemeleon_state, ffn_hidden_dim, ffn_n_layers,
               dropout, scaler):
    """
    Build a fresh MPNN for each fold/run.

    The CheMeleon state dict is CLONED before loading so that every call
    starts from identical pre-trained weights, regardless of how the previous
    fold updated them during finetuning.

    The only hard constraint linking CheMeleon to the FFN is:
        ffn input_dim == mp.output_dim  (both 2048 for CheMeleon)
    All other FFN parameters (hidden_dim, n_layers, dropout) are free to tune.
    """
    mp = nn.BondMessagePassing(**chemeleon_hyper)
    mp.load_state_dict({k: v.clone() for k, v in chemeleon_state.items()})

    agg = nn.MeanAggregation()

    # UnscaleTransform reverses the StandardScaler applied to training targets,
    # so model outputs are always in the original pEC50 units.
    output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
    ffn = nn.RegressionFFN(
        input_dim=mp.output_dim,   # must equal 2048 for CheMeleon
        hidden_dim=ffn_hidden_dim,
        n_layers=ffn_n_layers,
        dropout=dropout,
        output_transform=output_transform,
    )

    return models.MPNN(
        mp, agg, ffn,
        batch_norm=False,          # not recommended with CheMeleon
        metrics=[nn.metrics.RMSE(), nn.metrics.MAE()],
    )


def run_fold(fold_train, fold_val, chemeleon_hyper, chemeleon_state,
             params, featurizer, max_epochs, patience):
    """
    Train one stratified CV fold and return (best_val_rmse, best_epoch).

    Target normalisation is fit on the fold's training split only and then
    applied to the validation split — exactly as it would be in production.
    This prevents any leakage of validation statistics into training.
    """
    # Fit scaler on train fold only
    train_dset = data.MoleculeDataset(fold_train, featurizer)
    scaler = train_dset.normalize_targets()

    # Apply the same scaler to val (no refit)
    val_dset = data.MoleculeDataset(fold_val, featurizer)
    val_dset.normalize_targets(scaler)

    train_loader = data.build_dataloader(train_dset, num_workers=NUM_WORKERS)
    val_loader   = data.build_dataloader(val_dset,   num_workers=NUM_WORKERS, shuffle=False)

    mpnn = build_mpnn(chemeleon_hyper, chemeleon_state, **params, scaler=scaler)

    epoch_tracker = BestEpochTracker()
    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,   # suppressed to keep CV output readable
        accelerator="auto",
        devices=1,
        max_epochs=max_epochs,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=patience, mode="min"),
            epoch_tracker,
        ],
    )
    trainer.fit(mpnn, train_loader, val_loader)

    return epoch_tracker.best_val_loss, epoch_tracker.best_epoch


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1 — Download CheMeleon weights (skipped if already present)
# ═══════════════════════════════════════════════════════════════════════════════
if not CHEMELEON_PATH.exists():
    print("Downloading CheMeleon model weights from Zenodo...")
    urlretrieve(
        "https://zenodo.org/records/15460715/files/chemeleon_mp.pt",
        CHEMELEON_PATH,
    )
    print(f"  Saved to {CHEMELEON_PATH}")
else:
    print(f"CheMeleon weights found at {CHEMELEON_PATH}")

# ═══════════════════════════════════════════════════════════════════════════════
# Step 2 — Load CheMeleon checkpoint once; reuse for every fold via cloning
# ═══════════════════════════════════════════════════════════════════════════════
print("\nLoading CheMeleon checkpoint...")
chemeleon_ckpt  = torch.load(CHEMELEON_PATH, weights_only=True)
chemeleon_hyper = chemeleon_ckpt["hyper_parameters"]   # BondMessagePassing kwargs
chemeleon_state = chemeleon_ckpt["state_dict"]          # pre-trained weights tensor dict
print(f"  Hyper-parameters: {chemeleon_hyper}")

featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

# ═══════════════════════════════════════════════════════════════════════════════
# Step 3 — Load datasets
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\nLoading training data:\n  {TRAIN_PATH}")
df_train  = pd.read_csv(TRAIN_PATH)
all_train = build_datapoints(df_train[TRAIN_SMILES_COL].values,
                              df_train[[TRAIN_TARGET_COL]].values)
print(f"  Usable training datapoints: {len(all_train)}")
print(f"  pEC50 range: "
      f"{df_train[TRAIN_TARGET_COL].min():.2f} – {df_train[TRAIN_TARGET_COL].max():.2f}")

print(f"\nLoading external test set:\n  {TEST_PATH}")
df_test  = pd.read_csv(TEST_PATH)
all_test = build_datapoints(df_test[TEST_SMILES_COL].values,
                             df_test[[TEST_TARGET_COL]].values)
print(f"  Usable test datapoints: {len(all_test)}")

# ═══════════════════════════════════════════════════════════════════════════════
# Step 4 — Build stratification labels for the training set
#
# StratifiedKFold requires discrete class labels, but pEC50 is continuous.
# Solution: bin the pEC50 values into N_STRATA_BINS quantile-based groups.
#
# pd.qcut divides the range so each bin contains roughly the same number of
# compounds (equal-frequency binning), rather than equal-width intervals.
# This is preferred because PXR data tends to have many weakly active
# compounds and fewer highly active ones — equal-width bins would leave the
# high-activity bins sparsely populated and poorly represented in each fold.
# ═══════════════════════════════════════════════════════════════════════════════
pec50_values = np.array([d.y[0] for d in all_train])

strata = pd.qcut(
    pec50_values,
    q=N_STRATA_BINS,
    labels=False,        # return integer bin indices (0 … N_STRATA_BINS-1)
    duplicates="drop",   # silently merge bins if boundary values are identical
).astype(int)

print(f"\nStratification bins (equal-frequency, {N_STRATA_BINS} bins):")
for bin_id in range(strata.max() + 1):
    mask = strata == bin_id
    vals = pec50_values[mask]
    print(f"  Bin {bin_id}: n={mask.sum():4d}  "
          f"pEC50 {vals.min():.2f} – {vals.max():.2f}")

# ═══════════════════════════════════════════════════════════════════════════════
# Step 5 — 5-fold stratified CV grid search
#
# StratifiedKFold ensures that the proportion of compounds from each
# activity bin is (approximately) the same in every train and val fold.
# shuffle=True randomises fold assignment within each stratum.
# ═══════════════════════════════════════════════════════════════════════════════
param_combos = [
    dict(zip(PARAM_GRID.keys(), combo))
    for combo in itertools.product(*PARAM_GRID.values())
]

print(f"\n{'='*60}")
print(f"5-fold stratified CV grid search — {len(param_combos)} parameter combinations")
print(f"{'='*60}")

skf     = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
indices = np.arange(len(all_train))

cv_results = []

for combo_idx, params in enumerate(param_combos):
    print(f"\n[{combo_idx + 1}/{len(param_combos)}] {params}")
    fold_rmses, fold_epochs = [], []

    # StratifiedKFold.split takes (X, y) where y are the stratum labels
    for fold_num, (train_idx, val_idx) in enumerate(skf.split(indices, strata)):
        fold_train = [all_train[i] for i in train_idx]
        fold_val   = [all_train[i] for i in val_idx]

        rmse, best_epoch = run_fold(
            fold_train, fold_val,
            chemeleon_hyper, chemeleon_state,
            params, featurizer,
            max_epochs=CV_MAX_EPOCHS,
            patience=CV_PATIENCE,
        )
        fold_rmses.append(rmse)
        fold_epochs.append(best_epoch)
        print(f"  Fold {fold_num + 1}: val_RMSE={rmse:.4f}  best_epoch={best_epoch}")

    mean_rmse  = float(np.mean(fold_rmses))
    std_rmse   = float(np.std(fold_rmses))
    mean_epoch = int(np.mean(fold_epochs))
    print(f"  → Mean val RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}  "
          f"Mean best epoch: {mean_epoch}")

    cv_results.append({
        **params,
        "mean_val_rmse":   mean_rmse,
        "std_val_rmse":    std_rmse,
        "mean_best_epoch": mean_epoch,
    })

df_cv = pd.DataFrame(cv_results).sort_values("mean_val_rmse").reset_index(drop=True)
df_cv.to_csv(CV_RESULTS_PATH, index=False)
print(f"\nCV results saved to {CV_RESULTS_PATH}")
print(f"\n{df_cv.to_string(index=False)}")

# ── Select best hyperparameters ───────────────────────────────────────────────
best_row = df_cv.iloc[0]
best_params = {
    "ffn_hidden_dim": int(best_row["ffn_hidden_dim"]),
    "ffn_n_layers":   int(best_row["ffn_n_layers"]),
    "dropout":        float(best_row["dropout"]),
}
# Add a 10 % epoch buffer: when training on more data the model typically
# needs slightly more steps to converge to the same loss level.
final_epochs = max(int(best_row["mean_best_epoch"] * 1.1), 5)

print(f"\nBest hyperparameters : {best_params}")
print(f"Final model epochs   : {final_epochs}  (mean best epoch × 1.1)")

# ═══════════════════════════════════════════════════════════════════════════════
# Step 6 — Final model: train on ALL training data with best params
#
# No validation split is needed here because:
#   • Hyperparameters were already selected by CV.
#   • The training duration (final_epochs) is set from CV, not from a live
#     val_loss signal, so there is nothing to overfit to.
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"Training final model on all {len(all_train)} training compounds")
print(f"{'='*60}")

all_train_dset = data.MoleculeDataset(all_train, featurizer)
final_scaler   = all_train_dset.normalize_targets()
final_loader   = data.build_dataloader(all_train_dset, num_workers=NUM_WORKERS)

final_mpnn = build_mpnn(
    chemeleon_hyper, chemeleon_state,
    **best_params,
    scaler=final_scaler,
)
print("\nFinal model architecture:")
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

# ═══════════════════════════════════════════════════════════════════════════════
# Step 7 — Save final model as .pkl
#
# torch.save serialises the full model object (weights + architecture) using
# pickle.  Load it back in any script with:
#   model = torch.load("pxr_chemeleon_final.pkl", weights_only=False)
#   model.eval()
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\nSaving final model to {MODEL_PKL_PATH} ...")
torch.save(final_mpnn, MODEL_PKL_PATH)
print("  Saved.")

# ═══════════════════════════════════════════════════════════════════════════════
# Step 8 — Predict on the external test set
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\nPredicting on external test set ({len(all_test)} molecules)...")
final_mpnn.eval()

test_dset   = data.MoleculeDataset(all_test, featurizer)
test_loader = data.build_dataloader(test_dset, num_workers=NUM_WORKERS, shuffle=False)

raw_preds = final_trainer.predict(final_mpnn, test_loader)
preds     = torch.cat(raw_preds).numpy().flatten()

from rdkit.Chem import MolToSmiles as _mts
actual = df_test[TEST_TARGET_COL].values[: len(all_test)]
smiles = [_mts(d.mol) for d in all_test]
names  = df_test[TEST_NAME_COL].values[: len(all_test)]

df_out = pd.DataFrame({
    "Molecule Name":   names,
    "SMILES":          smiles,
    "pEC50_actual":    actual,
    "pEC50_predicted": preds,
    "residual":        np.array(actual) - preds,
})
df_out.to_csv(OUTPUT_PREDS, index=False)
print(f"Predictions saved to {OUTPUT_PREDS}")

rmse = np.sqrt(np.mean((df_out["pEC50_actual"] - df_out["pEC50_predicted"]) ** 2))
mae  = np.mean(np.abs(df_out["pEC50_actual"] - df_out["pEC50_predicted"]))
corr = df_out[["pEC50_actual", "pEC50_predicted"]].corr().iloc[0, 1]
print(f"\nExternal test set  —  RMSE: {rmse:.3f}  MAE: {mae:.3f}  Pearson r: {corr:.3f}")
