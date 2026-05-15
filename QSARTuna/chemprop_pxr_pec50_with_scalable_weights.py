"""
CheMeleon-finetuned Chemprop model for PXR pEC50 regression — weighted variant.

Compound weights are derived from the selectivity difference between the main
PXR assay and the counter screen (pEC50 - pEC50_counter):

  • Large positive diff → compound is selective for PXR → higher training weight
  • Negative diff      → counter screen more potent than main assay (possibly
                          promiscuous / non-selective) → lower training weight
  • No counter screen  → weight set to NEUTRAL_WEIGHT (1.0); no information
                          about selectivity, so treated equally

The raw differences are linearly scaled to [MIN_WEIGHT, MAX_WEIGHT] using the
min and max of all available diff values.  Compounds without a counter screen
(NaN diff) receive NEUTRAL_WEIGHT regardless of the scaling.

Pipeline:
  1. Compute and scale per-compound weights from pEC50_diff.
  2. 5-fold STRATIFIED CV grid search to select FFN hyperparameters.
  3. Retrain a final model on ALL training data using the selected params and
     weights, for the average number of best epochs found during CV.
  4. Save final model as a .pkl file (via torch.save).
  5. Predict on an external test set and report performance metrics.

Usage:
    conda activate chemprop
    python ~/chemprop_pxr_pec50_with_scalable_weights.py

To load the saved model for inference in another script:
    import torch
    model = torch.load("~/pxr_chemeleon_weighted_final.pkl", weights_only=False)
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
from sklearn.model_selection import StratifiedKFold

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
MODEL_PKL_PATH  = Path.home() / "pxr_chemeleon_weighted_final.pkl"
CV_RESULTS_PATH = Path.home() / "pxr_weighted_cv_results.csv"
OUTPUT_PREDS    = Path.home() / "pxr_weighted_external_test_predictions.csv"

TRAIN_SMILES_COL = "SMILES"
TRAIN_TARGET_COL = "pEC50"
DIFF_COL         = "pEC50_diff"   # pEC50 − pEC50_counter; NaN where no counter screen
TEST_SMILES_COL  = "SMILES"
TEST_TARGET_COL  = "pEC50"
TEST_NAME_COL    = "Molecule Name"

# Weight scaling bounds — applied to the full observed pEC50_diff range.
# MIN_WEIGHT is assigned to the least selective compound (most negative diff).
# MAX_WEIGHT is assigned to the most selective compound (most positive diff).
# Compounds without a counter screen receive NEUTRAL_WEIGHT.
MIN_WEIGHT     = 0.5
MAX_WEIGHT     = 2.0
NEUTRAL_WEIGHT = 1.0   # used for the ~55 % of compounds with no counter screen

N_FOLDS       = 5
N_STRATA_BINS = N_FOLDS   # equal-frequency pEC50 bins for stratified splitting
CV_MAX_EPOCHS = 30
CV_PATIENCE   = 5
NUM_WORKERS   = 0

# Hyperparameter grid — FFN only; CheMeleon MP architecture is fixed.
PARAM_GRID = {
    "ffn_hidden_dim": [300, 512],
    "ffn_n_layers":   [2, 3],
    "dropout":        [0.0, 0.2],
}

# Learning rate schedule — deliberately lower than ChemProp default (max_lr=1e-3).
# A high LR overwrites the pretrained CheMeleon message-passing weights too
# aggressively, which caused the rank drop from 50 to 163 in the fast-LR run.
INIT_LR  = 1e-4
MAX_LR   = 2e-4
FINAL_LR = 1e-5


# ── Custom callback — records epoch with lowest val loss ──────────────────────
class BestEpochTracker(pl.Callback):
    """
    Tracks which training epoch produced the lowest val_loss.
    The mean best epoch across CV folds sets the final model's training budget,
    avoiding the need for a validation set when retraining on all data.
    """

    def __init__(self):
        self.best_val_loss = float("inf")
        self.best_epoch = 0

    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = float(trainer.callback_metrics.get("val_loss", float("inf")))
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = trainer.current_epoch


# ── Weight computation ────────────────────────────────────────────────────────
def compute_weights(diff_values: np.ndarray) -> np.ndarray:
    """
    Scale pEC50_diff values linearly to [MIN_WEIGHT, MAX_WEIGHT].

    NaN entries (no counter screen available) receive NEUTRAL_WEIGHT so that
    they contribute equally to training — we simply have no selectivity
    information about them, not evidence that they are unreliable.

    The scaling uses the global min/max of the *available* (non-NaN) diffs so
    that the weight range is consistent across folds and the final model.

    Parameters
    ----------
    diff_values : array of pEC50_diff values (may contain NaN)

    Returns
    -------
    weights : array of per-compound weights, same length as diff_values
    """
    weights = np.full(len(diff_values), NEUTRAL_WEIGHT, dtype=float)

    has_counter = ~np.isnan(diff_values)
    available   = diff_values[has_counter]

    diff_min = available.min()
    diff_max = available.max()

    # Linear map: diff_min → MIN_WEIGHT, diff_max → MAX_WEIGHT
    scaled = MIN_WEIGHT + (MAX_WEIGHT - MIN_WEIGHT) * (
        (available - diff_min) / (diff_max - diff_min)
    )
    weights[has_counter] = scaled
    return weights


# ── Helpers ───────────────────────────────────────────────────────────────────
def build_datapoints(smiles_arr, targets_arr, weights_arr):
    """
    Convert SMILES, targets, and per-compound weights into MoleculeDatapoints.
    Compounds whose SMILES RDKit cannot parse are silently skipped.
    The weight is stored on the datapoint and used by Chemprop's loss function
    to scale each compound's contribution to the gradient.
    """
    points, skipped = [], 0
    for smi, y, w in zip(smiles_arr, targets_arr, weights_arr):
        dp = data.MoleculeDatapoint.from_smi(smi, y, weight=float(w))
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

    CheMeleon weights are CLONED so every call starts from identical pre-trained
    weights, regardless of how the previous fold updated them during finetuning.

    The only hard interface constraint between CheMeleon and the FFN is:
        ffn input_dim == mp.output_dim  (2048 for CheMeleon)
    """
    mp = nn.BondMessagePassing(**chemeleon_hyper)
    mp.load_state_dict({k: v.clone() for k, v in chemeleon_state.items()})

    agg = nn.MeanAggregation()

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
        batch_norm=False,
        metrics=[nn.metrics.RMSE(), nn.metrics.MAE()],
        init_lr=INIT_LR,
        max_lr=MAX_LR,
        final_lr=FINAL_LR,
    )


def run_fold(fold_train, fold_val, chemeleon_hyper, chemeleon_state,
             params, featurizer, max_epochs, patience):
    """
    Train one stratified CV fold and return (best_val_rmse, best_epoch).

    Target normalisation is fit on the fold's training split only and applied
    to the validation split — no leakage of validation statistics into training.
    Per-compound weights are embedded in each MoleculeDatapoint and are
    automatically picked up by Chemprop's weighted loss computation.
    """
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
# Step 2 — Load CheMeleon checkpoint once; cloned for every fold
# ═══════════════════════════════════════════════════════════════════════════════
print("\nLoading CheMeleon checkpoint...")
chemeleon_ckpt  = torch.load(CHEMELEON_PATH, weights_only=True)
chemeleon_hyper = chemeleon_ckpt["hyper_parameters"]
chemeleon_state = chemeleon_ckpt["state_dict"]
print(f"  Hyper-parameters: {chemeleon_hyper}")

featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

# ═══════════════════════════════════════════════════════════════════════════════
# Step 3 — Load training data and compute per-compound weights
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\nLoading training data:\n  {TRAIN_PATH}")
df_train = pd.read_csv(TRAIN_PATH)
print(f"  {len(df_train)} rows")
print(f"  pEC50 range : {df_train[TRAIN_TARGET_COL].min():.2f} – "
      f"{df_train[TRAIN_TARGET_COL].max():.2f}")

# ── Compute scalable weights from pEC50_diff ──────────────────────────────────
diff_values = df_train[DIFF_COL].values   # NaN where no counter screen
train_weights = compute_weights(diff_values)

n_with_counter = (~np.isnan(diff_values)).sum()
print(f"\n  Counter screen available : {n_with_counter} / {len(df_train)} compounds")
print(f"  pEC50_diff range         : {np.nanmin(diff_values):.3f} – "
      f"{np.nanmax(diff_values):.3f}  (mean {np.nanmean(diff_values):.3f})")
print(f"  Weight range             : {train_weights.min():.3f} – "
      f"{train_weights.max():.3f}  (neutral={NEUTRAL_WEIGHT})")

all_train = build_datapoints(
    df_train[TRAIN_SMILES_COL].values,
    df_train[[TRAIN_TARGET_COL]].values,
    train_weights,
)
print(f"  Usable datapoints: {len(all_train)}")

# ═══════════════════════════════════════════════════════════════════════════════
# Step 4 — Load external test set
#
# The test set has no counter screen columns, so weights are not applied there.
# Prediction is purely based on SMILES; test pEC50 values are used only to
# report performance after training is complete.
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\nLoading external test set:\n  {TEST_PATH}")
df_test  = pd.read_csv(TEST_PATH)

# Test compounds receive unit weight — weights only affect training loss,
# not inference, so this is a placeholder that has no effect on predictions.
test_weights = np.ones(len(df_test))
all_test = build_datapoints(
    df_test[TEST_SMILES_COL].values,
    df_test[[TEST_TARGET_COL]].values,
    test_weights,
)
print(f"  Usable test datapoints: {len(all_test)}")

# ═══════════════════════════════════════════════════════════════════════════════
# Step 5 — Build stratification labels
#
# StratifiedKFold needs discrete class labels; pEC50 is continuous.
# pd.qcut creates equal-frequency bins (each bin has ~the same number of
# compounds), which handles the skewed PXR activity distribution better than
# equal-width bins.
# ═══════════════════════════════════════════════════════════════════════════════
pec50_values = np.array([d.y[0] for d in all_train])

strata = pd.qcut(
    pec50_values,
    q=N_STRATA_BINS,
    labels=False,        # integer bin indices 0 … N_STRATA_BINS-1
    duplicates="drop",
).astype(int)

print(f"\nStratification bins (equal-frequency, {N_STRATA_BINS} bins):")
for bin_id in range(strata.max() + 1):
    mask = strata == bin_id
    vals = pec50_values[mask]
    ws   = np.array([all_train[i].weight for i in np.where(mask)[0]])
    print(f"  Bin {bin_id}: n={mask.sum():4d}  "
          f"pEC50 {vals.min():.2f}–{vals.max():.2f}  "
          f"mean_weight={ws.mean():.3f}")

# ═══════════════════════════════════════════════════════════════════════════════
# Step 6 — 5-fold stratified CV grid search
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
# 10 % epoch buffer: more data per epoch when training on the full set means
# the model typically needs slightly more steps to reach equivalent convergence.
final_epochs = max(int(best_row["mean_best_epoch"] * 1.1), 5)

print(f"\nBest hyperparameters : {best_params}")
print(f"Final model epochs   : {final_epochs}  (mean best epoch × 1.1)")

# ═══════════════════════════════════════════════════════════════════════════════
# Step 7 — Final model: train on ALL training data with weights
#
# No validation split needed:
#   • Hyperparameters fixed by CV.
#   • Training duration fixed by CV (final_epochs).
# Per-compound weights from pEC50_diff are embedded in the MoleculeDatapoints
# and are automatically applied by Chemprop's weighted MSE loss.
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"Training final model on all {len(all_train)} training compounds (with weights)")
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
# Step 8 — Save final model as .pkl
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\nSaving final model to {MODEL_PKL_PATH} ...")
torch.save(final_mpnn, MODEL_PKL_PATH)
print("  Saved.")

# ═══════════════════════════════════════════════════════════════════════════════
# Step 9 — Predict on external test set
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
