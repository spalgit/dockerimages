"""
CheMeleon-finetuned Chemprop model for PXR pEC50 regression.

Pipeline:
  1. 5-fold CV grid search to select FFN hyperparameters.
  2. Retrain a final model on ALL training data for the average number of
     epochs that gave best validation loss across folds.
  3. Save final model as a .pkl file (torch.save).
  4. Predict on an external test set and report performance metrics.

Usage:
    conda activate chemprop
    python ~/chemprop_pxr_pec50.py

To load the saved model later for inference:
    import torch
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
from sklearn.model_selection import KFold

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
CHEMELEON_PATH   = Path.home() / "chemeleon_mp.pt"
MODEL_PKL_PATH   = Path.home() / "pxr_chemeleon_final.pkl"
CV_RESULTS_PATH  = Path.home() / "pxr_cv_results.csv"
OUTPUT_PREDS     = Path.home() / "pxr_external_test_predictions.csv"

TRAIN_SMILES_COL = "SMILES"
TRAIN_TARGET_COL = "pEC50"
TEST_SMILES_COL  = "SMILES"
TEST_TARGET_COL  = "pEC50"
TEST_NAME_COL    = "Molecule Name"

N_FOLDS       = 5
CV_MAX_EPOCHS = 30   # upper bound per fold; early stopping usually kicks in first
CV_PATIENCE   = 5    # early-stopping patience during CV
NUM_WORKERS   = 0

# Hyperparameter grid — only FFN params are tuned; CheMeleon MP is fixed.
PARAM_GRID = {
    "ffn_hidden_dim": [300, 512],
    "ffn_n_layers":   [2, 3],
    "dropout":        [0.0, 0.2],
}


# ── Custom callback — tracks epoch with lowest val loss ───────────────────────
class BestEpochTracker(pl.Callback):
    """Records the epoch index at which val_loss was minimised."""

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
    """Convert SMILES + target arrays into MoleculeDatapoints, skip bad SMILES."""
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
    Build a fresh MPNN each time by cloning CheMeleon weights.
    Cloning is essential so each fold/run starts from identical pre-trained weights.
    """
    mp = nn.BondMessagePassing(**chemeleon_hyper)
    mp.load_state_dict({k: v.clone() for k, v in chemeleon_state.items()})

    agg = nn.MeanAggregation()

    output_transform = nn.UnscaleTransform.from_standard_scaler(scaler)
    ffn = nn.RegressionFFN(
        input_dim=mp.output_dim,   # must match CheMeleon output (2048)
        hidden_dim=ffn_hidden_dim,
        n_layers=ffn_n_layers,
        dropout=dropout,
        output_transform=output_transform,
    )

    return models.MPNN(
        mp, agg, ffn,
        batch_norm=False,
        metrics=[nn.metrics.RMSE(), nn.metrics.MAE()],
    )


def run_fold(fold_train, fold_val, chemeleon_hyper, chemeleon_state,
             params, featurizer, max_epochs, patience):
    """Train one CV fold. Returns (best_val_rmse, best_epoch)."""
    train_dset = data.MoleculeDataset(fold_train, featurizer)
    scaler = train_dset.normalize_targets()

    val_dset = data.MoleculeDataset(fold_val, featurizer)
    val_dset.normalize_targets(scaler)

    train_loader = data.build_dataloader(train_dset, num_workers=NUM_WORKERS)
    val_loader   = data.build_dataloader(val_dset,   num_workers=NUM_WORKERS, shuffle=False)

    mpnn = build_mpnn(chemeleon_hyper, chemeleon_state, **params, scaler=scaler)

    epoch_tracker = BestEpochTracker()
    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,   # keep CV output concise
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
# Step 1 — Download CheMeleon weights
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
# Step 2 — Load CheMeleon checkpoint (once; reused for every fold)
# ═══════════════════════════════════════════════════════════════════════════════
print("\nLoading CheMeleon checkpoint...")
chemeleon_ckpt  = torch.load(CHEMELEON_PATH, weights_only=True)
chemeleon_hyper = chemeleon_ckpt["hyper_parameters"]
chemeleon_state = chemeleon_ckpt["state_dict"]
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

print(f"\nLoading external test set:\n  {TEST_PATH}")
df_test  = pd.read_csv(TEST_PATH)
all_test = build_datapoints(df_test[TEST_SMILES_COL].values,
                             df_test[[TEST_TARGET_COL]].values)
print(f"  Usable test datapoints: {len(all_test)}")

# ═══════════════════════════════════════════════════════════════════════════════
# Step 4 — 5-fold CV grid search
# ═══════════════════════════════════════════════════════════════════════════════
param_combos = [
    dict(zip(PARAM_GRID.keys(), combo))
    for combo in itertools.product(*PARAM_GRID.values())
]

print(f"\n{'='*60}")
print(f"5-fold CV grid search — {len(param_combos)} parameter combinations")
print(f"{'='*60}")

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
indices = np.arange(len(all_train))

cv_results = []

for combo_idx, params in enumerate(param_combos):
    print(f"\n[{combo_idx + 1}/{len(param_combos)}] {params}")
    fold_rmses, fold_epochs = [], []

    for fold_num, (train_idx, val_idx) in enumerate(kf.split(indices)):
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
        "mean_val_rmse":  mean_rmse,
        "std_val_rmse":   std_rmse,
        "mean_best_epoch": mean_epoch,
    })

df_cv = pd.DataFrame(cv_results).sort_values("mean_val_rmse").reset_index(drop=True)
df_cv.to_csv(CV_RESULTS_PATH, index=False)
print(f"\nCV results saved to {CV_RESULTS_PATH}")
print(f"\n{df_cv.to_string(index=False)}")

# Select best hyperparameters
best_row = df_cv.iloc[0]
best_params = {
    "ffn_hidden_dim": int(best_row["ffn_hidden_dim"]),
    "ffn_n_layers":   int(best_row["ffn_n_layers"]),
    "dropout":        float(best_row["dropout"]),
}
# Add 10 % buffer to the average best epoch to account for using more data
final_epochs = max(int(best_row["mean_best_epoch"] * 1.1), 5)

print(f"\nBest hyperparameters: {best_params}")
print(f"Final model will train for {final_epochs} epochs (avg best epoch × 1.1)")

# ═══════════════════════════════════════════════════════════════════════════════
# Step 5 — Final model: train on ALL training data, fixed epochs, no val split
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
# Step 6 — Save final model as .pkl
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\nSaving final model to {MODEL_PKL_PATH} ...")
torch.save(final_mpnn, MODEL_PKL_PATH)
print("  Done.")

# ═══════════════════════════════════════════════════════════════════════════════
# Step 7 — Predict on external test set
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\nPredicting on external test set ({len(all_test)} molecules)...")
final_mpnn.eval()

test_dset   = data.MoleculeDataset(all_test, featurizer)
test_loader = data.build_dataloader(test_dset, num_workers=NUM_WORKERS, shuffle=False)

raw_preds = final_trainer.predict(final_mpnn, test_loader)
preds     = torch.cat(raw_preds).numpy().flatten()

actual = [float(d.y[0]) for d in all_test]
smiles = [d.smi for d in all_test]
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
