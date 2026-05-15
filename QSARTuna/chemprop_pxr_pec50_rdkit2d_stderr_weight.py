"""
ChemProp model for PXR pEC50 regression — RDKit 2D extra descriptors +
measurement-uncertainty sample weighting (1/std_error).

This is fundamentally different from multi-task learning with std_error:

  Multi-task approach (tried, hurt rank):
      std_error is an OUTPUT the model tries to predict alongside pEC50.
      Splits model capacity between two objectives; adds noise to the
      pEC50 learning signal.

  This script — sample weighting (NOT multi-task):
      std_error is METADATA about measurement quality.  The model still
      predicts only pEC50.  Each compound's contribution to the MAE loss
      is scaled by 1/std_error so that precisely-measured compounds
      (low std_error, high confidence) steer the gradient more than
      noisy single-replicate measurements (high std_error).

      Weight = 1/std_error, normalised to mean = 1.0 across the training
      set so the effective learning rate is preserved.
      Weight range after normalisation: ~0.16 – 2.34.
      All 4,140 training compounds have std_error → no neutral fallback needed.

Architecture: standard ChemProp MPNN (no foundation model) + RDKit 2D x_d
              + MAE loss + slow LR — same as chemprop_pxr_pec50_rdkit2d_features.py
              except for the weighting.

Pipeline:
  1. Compute per-compound weights: normalised 1/std_error.
  2. Compute 217 RDKit 2D descriptors; drop NaN/inf/zero-variance columns.
  3. Per-fold StandardScaler on x_d (fit on train, apply to val) — no leakage.
  4. 5-fold stratified CV grid search over FFN hyperparameters.
  5. Retrain final model on ALL training data.
  6. Predict on external test set.

Usage:
    conda activate chemprop
    python ~/dockerimages/QSARTuna/chemprop_pxr_pec50_rdkit2d_stderr_weight.py
"""

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightning import pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from chemprop import data, featurizers, models, nn

# ── Configuration ──────────────────────────────────────────────────────────────
TRAIN_PATH = Path(
    "/home/spal/dockerimages/QSARTuna/PXR_For_QSARTuna_Modelling/"
    "processed_Openadmet_REAL_PXR_train_AND_test_main_with_side_info_"
    "AND_counter_screen_weighted.csv"
)
TEST_PATH = Path.home() / "dockerimages/QSARTuna/PXR/Prediction_OpenAdmet_ChemProp_Only_OpenADMET_Data.csv"
MODEL_PKL_PATH  = Path.home() / "pxr_chemprop_rdkit2d_stderr_weight_final.pkl"
CV_RESULTS_PATH = Path.home() / "pxr_rdkit2d_stderr_weight_cv_results.csv"
OUTPUT_PREDS    = Path.home() / "pxr_rdkit2d_stderr_weight_test_predictions.csv"
KEPT_DESCS_PATH = Path.home() / "pxr_rdkit2d_stderr_weight_kept_descriptors.txt"

TRAIN_SMILES_COL = "SMILES"
TRAIN_TARGET_COL = "pEC50"
TEST_SMILES_COL  = "SMILES"
TEST_TARGET_COL  = "pEC50"
TEST_NAME_COL    = "Molecule Name"

N_FOLDS       = 5
N_STRATA_BINS = N_FOLDS
# Training from scratch needs more epochs than fine-tuning a pretrained backbone
CV_MAX_EPOCHS = 50
CV_PATIENCE   = 10
NUM_WORKERS   = 0

# Slow LR — same values used in the Chemeleon weighted variant
INIT_LR  = 1e-4
MAX_LR   = 2e-4
FINAL_LR = 1e-5

STD_ERROR_COL = "std_error"
MIN_STD_ERROR = 0.05   # clip floor — 306 compounds below this; caps weight at 1/0.05=20 before normalisation

PARAM_GRID = {
    "ffn_hidden_dim": [300, 512],
    "ffn_n_layers":   [2, 3],
    "dropout":        [0.0, 0.2],
    "mp_depth":       [3, 4],      # message passing steps; Anvil best model used 4
    "mp_hidden_dim":  [300, 1024], # encoder hidden width; Anvil best model used 2048
}


# ── Callback: track best validation epoch ─────────────────────────────────────
class BestEpochTracker(pl.Callback):
    """Records the epoch with the lowest val_loss across a training run."""

    def __init__(self):
        self.best_val_loss = float("inf")
        self.best_epoch    = 0

    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = float(trainer.callback_metrics.get("val_loss", float("inf")))
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch    = trainer.current_epoch


# ── Sample weight from measurement uncertainty ────────────────────────────────
def compute_weights(std_error_values: np.ndarray) -> np.ndarray:
    """
    Weight = 1 / clip(std_error, MIN_STD_ERROR), normalised to mean = 1.0.

    Clipping prevents a handful of very precise measurements from dominating.
    Normalisation keeps the effective learning rate unchanged.
    All compounds are retained — only the weight magnitude is capped.
    """
    clipped    = np.clip(std_error_values, MIN_STD_ERROR, None)
    raw        = 1.0 / clipped
    return raw / raw.mean()


# ── RDKit 2D descriptor utilities ─────────────────────────────────────────────
_DESC_LIST  = [(name, fn) for name, fn in Descriptors.descList]
ALL_DESC_NAMES = [name for name, _ in _DESC_LIST]


def compute_rdkit_descriptors(mols: list) -> np.ndarray:
    """
    Compute all 217 RDKit 2D descriptors for each mol.
    Returns shape (n_mols, 217). Failed descriptor calls → NaN.
    """
    rows = []
    for mol in mols:
        row = []
        for _, fn in _DESC_LIST:
            try:
                v = fn(mol)
                row.append(float(v) if v is not None else np.nan)
            except Exception:
                row.append(np.nan)
        rows.append(row)
    return np.array(rows, dtype=float)


def select_valid_columns(arr: np.ndarray) -> np.ndarray:
    """
    Boolean column mask: keep columns that are finite for ALL molecules
    and have nonzero variance across the training set.
    This mask is derived from training data only and applied to test data
    to avoid leakage.
    """
    finite = np.all(np.isfinite(arr), axis=0)
    varied = np.var(arr, axis=0) > 0
    return finite & varied


# ── Build MPNN ─────────────────────────────────────────────────────────────────
def build_mpnn(n_descriptors: int, ffn_hidden_dim: int, ffn_n_layers: int,
               dropout: float, mp_depth: int, mp_hidden_dim: int,
               target_scaler) -> models.MPNN:
    """
    Build a standard ChemProp MPNN with RDKit2D descriptors concatenated
    before the FFN.

    x_d is pre-scaled outside the model (per-fold StandardScaler), so no
    X_d_transform is stored here — the model receives already-scaled data.

    Parameters
    ----------
    n_descriptors  : number of kept RDKit descriptor columns
    mp_depth       : number of BondMessagePassing steps (Anvil best used 4)
    mp_hidden_dim  : hidden dimension of the message-passing network (d_h)
    target_scaler  : StandardScaler fitted on fold/full training targets;
                     used to build the output UnscaleTransform
    """
    feat = featurizers.SimpleMoleculeMolGraphFeaturizer()
    mp   = nn.BondMessagePassing(
               d_v=feat.atom_fdim, d_e=feat.bond_fdim,
               depth=mp_depth, d_h=mp_hidden_dim,
           )
    agg  = nn.MeanAggregation()

    output_transform = nn.UnscaleTransform.from_standard_scaler(target_scaler)
    ffn = nn.RegressionFFN(
        input_dim=mp.output_dim + n_descriptors,   # mp_hidden_dim + n_descriptors
        hidden_dim=ffn_hidden_dim,
        n_layers=ffn_n_layers,
        dropout=dropout,
        criterion=nn.metrics.MAE(),
        output_transform=output_transform,
    )
    return models.MPNN(
        mp, agg, ffn,
        batch_norm=True,
        metrics=[nn.metrics.RMSE(), nn.metrics.MAE()],
        init_lr=INIT_LR,
        max_lr=MAX_LR,
        final_lr=FINAL_LR,
    )


# ── Build datapoints ───────────────────────────────────────────────────────────
def make_datapoints(mols, targets, weights, x_d):
    """Zip mol objects, targets, weights, and pre-scaled x_d into datapoints."""
    return [
        data.MoleculeDatapoint(
            mol=mol,
            y=np.array([t], dtype=float),
            weight=float(w),
            x_d=xd.astype(float),
        )
        for mol, t, w, xd in zip(mols, targets, weights, x_d)
    ]


# ── CV fold runner ─────────────────────────────────────────────────────────────
def run_fold(fold_idx, train_mols, train_targets, train_weights, train_x_d,
             val_mols,   val_targets,   val_weights,   val_x_d,
             n_descriptors, params, max_epochs, patience):
    """
    Train one stratified CV fold.

    x_d arrays are already scaled for this fold (fit on fold-train, applied
    to fold-val).  Target normalisation is handled inside this function via
    ChemProp's normalize_targets() so it is fold-local and leak-free.

    Returns (best_val_loss, best_epoch).
    """
    feat = featurizers.SimpleMoleculeMolGraphFeaturizer()

    train_dps = make_datapoints(train_mols, train_targets, train_weights, train_x_d)
    val_dps   = make_datapoints(val_mols,   val_targets,   val_weights,   val_x_d)

    train_dset = data.MoleculeDataset(train_dps, feat)
    val_dset   = data.MoleculeDataset(val_dps,   feat)

    target_scaler = train_dset.normalize_targets()
    val_dset.normalize_targets(target_scaler)

    train_loader = data.build_dataloader(train_dset, num_workers=NUM_WORKERS)
    val_loader   = data.build_dataloader(val_dset,   num_workers=NUM_WORKERS, shuffle=False)

    mpnn = build_mpnn(n_descriptors, **params, target_scaler=target_scaler)

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


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — Load and parse training data
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nLoading training data:\n  {TRAIN_PATH}")
df_train = pd.read_csv(TRAIN_PATH)
print(f"  {len(df_train)} rows  |  sample weights = 1/std_error (normalised to mean=1.0)")

# Parse SMILES → keep only valid mols; collect std_error alongside targets
print("\nParsing training SMILES...")
train_mols, train_smiles, train_targets, train_stderr = [], [], [], []
for smi, y, se in zip(df_train[TRAIN_SMILES_COL].values,
                      df_train[TRAIN_TARGET_COL].values,
                      df_train[STD_ERROR_COL].values):
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        train_mols.append(mol)
        train_smiles.append(smi)
        train_targets.append(y)
        train_stderr.append(se)
    else:
        print(f"  Skipped unparseable SMILES: {smi[:40]}")

train_targets = np.array(train_targets)
train_stderr  = np.array(train_stderr, dtype=float)

# Compute 1/std_error weights (clipped + normalised)
train_weights_stderr = compute_weights(train_stderr)
n_clipped = (train_stderr < MIN_STD_ERROR).sum()

print(f"  Usable training molecules : {len(train_mols)}")
print(f"  std_error range           : {train_stderr.min():.3f} – {train_stderr.max():.3f}  "
      f"(mean={train_stderr.mean():.3f})")
print(f"  Compounds clipped at {MIN_STD_ERROR} std_error : {n_clipped}  "
      f"(weight capped, NOT removed)")
print(f"  Weight range after norm   : {train_weights_stderr.min():.3f} – "
      f"{train_weights_stderr.max():.3f}  (mean={train_weights_stderr.mean():.3f})")

# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — Load and parse test data
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nLoading test data:\n  {TEST_PATH}")
df_test = pd.read_csv(TEST_PATH)
test_mols, test_smiles, test_targets, test_names = [], [], [], []
for _, row in df_test.iterrows():
    mol = Chem.MolFromSmiles(row[TEST_SMILES_COL])
    if mol is not None:
        test_mols.append(mol)
        test_smiles.append(row[TEST_SMILES_COL])
        test_targets.append(row[TEST_TARGET_COL])
        test_names.append(row[TEST_NAME_COL])
test_targets = np.array(test_targets)
print(f"  Usable test molecules: {len(test_mols)}")

# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — Compute RDKit 2D descriptors; select valid columns from training set
# ══════════════════════════════════════════════════════════════════════════════
print("\nComputing RDKit 2D descriptors for training set...")
x_d_train_all = compute_rdkit_descriptors(train_mols)   # (n_train, 217)

col_mask   = select_valid_columns(x_d_train_all)
kept_names = [ALL_DESC_NAMES[i] for i, k in enumerate(col_mask) if k]
print(f"  Kept {col_mask.sum()} / {len(col_mask)} descriptors "
      f"(dropped {(~col_mask).sum()} NaN/inf/zero-var columns)")

x_d_train_raw = x_d_train_all[:, col_mask]   # (n_train, n_kept)
n_descriptors = x_d_train_raw.shape[1]

# Save kept descriptor names for reproducibility
with open(KEPT_DESCS_PATH, "w") as fh:
    fh.write("\n".join(kept_names))
print(f"  Descriptor names saved to {KEPT_DESCS_PATH}")

print("\nComputing RDKit 2D descriptors for test set...")
x_d_test_all = compute_rdkit_descriptors(test_mols)
x_d_test_raw = x_d_test_all[:, col_mask]     # (n_test, n_kept) — same columns

# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — Build stratification labels
# ══════════════════════════════════════════════════════════════════════════════
strata = pd.qcut(
    train_targets, q=N_STRATA_BINS, labels=False, duplicates="drop"
).astype(int)

print(f"\nStratification ({N_STRATA_BINS} equal-frequency bins):")
for b in range(strata.max() + 1):
    mask = strata == b
    print(f"  Bin {b}: n={mask.sum():4d}  "
          f"pEC50 {train_targets[mask].min():.2f}–{train_targets[mask].max():.2f}")

# ══════════════════════════════════════════════════════════════════════════════
# Step 5 — 5-fold stratified CV grid search
# ══════════════════════════════════════════════════════════════════════════════
param_combos = [
    dict(zip(PARAM_GRID.keys(), combo))
    for combo in itertools.product(*PARAM_GRID.values())
]

print(f"\n{'='*60}")
print(f"5-fold CV — {len(param_combos)} hyperparameter combos  |  "
      f"n_descriptors={n_descriptors}")
print(f"{'='*60}")

skf     = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
indices = np.arange(len(train_mols))

cv_results = []

for combo_idx, params in enumerate(param_combos):
    print(f"\n[{combo_idx + 1}/{len(param_combos)}] {params}")
    fold_rmses, fold_epochs = [], []

    for fold_num, (tr_idx, va_idx) in enumerate(skf.split(indices, strata)):
        # Scale x_d per fold: fit on fold-train, apply to fold-val
        xd_scaler = StandardScaler()
        x_d_tr = xd_scaler.fit_transform(x_d_train_raw[tr_idx])
        x_d_va = xd_scaler.transform(x_d_train_raw[va_idx])

        rmse, best_epoch = run_fold(
            fold_num,
            [train_mols[i] for i in tr_idx],
            train_targets[tr_idx],
            train_weights_stderr[tr_idx],
            x_d_tr,
            [train_mols[i] for i in va_idx],
            train_targets[va_idx],
            train_weights_stderr[va_idx],
            x_d_va,
            n_descriptors,
            params,
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

    cv_results.append({**params,
                       "mean_val_rmse":   mean_rmse,
                       "std_val_rmse":    std_rmse,
                       "mean_best_epoch": mean_epoch})

df_cv = pd.DataFrame(cv_results).sort_values("mean_val_rmse").reset_index(drop=True)
df_cv.to_csv(CV_RESULTS_PATH, index=False)
print(f"\nCV results saved to {CV_RESULTS_PATH}")
print(f"\n{df_cv.to_string(index=False)}")

# ── Select best hyperparameters ────────────────────────────────────────────────
best_row    = df_cv.iloc[0]
best_params = {
    "ffn_hidden_dim": int(best_row["ffn_hidden_dim"]),
    "ffn_n_layers":   int(best_row["ffn_n_layers"]),
    "dropout":        float(best_row["dropout"]),
}
final_epochs = max(int(best_row["mean_best_epoch"] * 1.1), 5)
print(f"\nBest hyperparameters : {best_params}")
print(f"Final model epochs   : {final_epochs}")

# ══════════════════════════════════════════════════════════════════════════════
# Step 6 — Final model: train on ALL training data
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"Training final model on all {len(train_mols)} compounds")
print(f"{'='*60}")

# Fit a single x_d scaler on all training data; apply to test set
final_xd_scaler = StandardScaler()
x_d_train_scaled = final_xd_scaler.fit_transform(x_d_train_raw)
x_d_test_scaled  = final_xd_scaler.transform(x_d_test_raw)

feat = featurizers.SimpleMoleculeMolGraphFeaturizer()

all_train_dps = make_datapoints(
    train_mols, train_targets, train_weights_stderr, x_d_train_scaled
)
all_train_dset  = data.MoleculeDataset(all_train_dps, feat)
final_scaler    = all_train_dset.normalize_targets()
final_loader    = data.build_dataloader(all_train_dset, num_workers=NUM_WORKERS)

final_mpnn = build_mpnn(
    n_descriptors, **best_params, target_scaler=final_scaler
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

# ══════════════════════════════════════════════════════════════════════════════
# Step 7 — Save model
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nSaving model to {MODEL_PKL_PATH} ...")
torch.save(final_mpnn, MODEL_PKL_PATH)
print("  Saved.")

# ══════════════════════════════════════════════════════════════════════════════
# Step 8 — Predict on external test set
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nPredicting on test set ({len(test_mols)} molecules)...")
final_mpnn.eval()

# Test data uses x_d scaled with the training scaler (already done above)
test_weights_dummy = np.ones(len(test_mols))
test_dps   = make_datapoints(test_mols, test_targets, test_weights_dummy, x_d_test_scaled)
test_dset  = data.MoleculeDataset(test_dps, feat)
test_loader = data.build_dataloader(test_dset, num_workers=NUM_WORKERS, shuffle=False)

raw_preds = final_trainer.predict(final_mpnn, test_loader)
preds     = torch.cat(raw_preds).numpy().flatten()

df_out = pd.DataFrame({
    "Molecule Name":   test_names,
    "SMILES":          test_smiles,
    "pEC50_actual":    test_targets,
    "pEC50_predicted": preds,
    "residual":        test_targets - preds,
})
df_out.to_csv(OUTPUT_PREDS, index=False)
print(f"Predictions saved to {OUTPUT_PREDS}")

rmse = np.sqrt(np.mean((df_out["pEC50_actual"] - df_out["pEC50_predicted"]) ** 2))
mae  = np.mean(np.abs(df_out["pEC50_actual"]  - df_out["pEC50_predicted"]))
corr = df_out[["pEC50_actual", "pEC50_predicted"]].corr().iloc[0, 1]
print(f"\nExternal test set  —  RMSE: {rmse:.3f}  MAE: {mae:.3f}  Pearson r: {corr:.3f}")
