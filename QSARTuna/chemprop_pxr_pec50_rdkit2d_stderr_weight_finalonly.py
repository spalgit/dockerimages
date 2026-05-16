"""
ChemProp PXR pEC50 — RDKit 2D + stderr weighting — FINAL MODEL ONLY.

Skips the CV grid search entirely. Uses the best hyperparameters already
determined from the full CV run:

    ffn_hidden_dim=512, ffn_n_layers=3, dropout=0.0,
    mp_depth=4, mp_hidden_dim=1024
    mean_best_epoch=14  →  final training epochs = 15

Writes the same output files as the full script so predictions can be
used directly for submission.

Usage:
    conda activate chemprop
    python ~/dockerimages/QSARTuna/chemprop_pxr_pec50_rdkit2d_stderr_weight_finalonly.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightning import pytorch as pl
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.preprocessing import StandardScaler

from chemprop import data, featurizers, models, nn

# ── Configuration ──────────────────────────────────────────────────────────────
TRAIN_PATH = Path(
    "/home/spal/dockerimages/QSARTuna/PXR_For_QSARTuna_Modelling/"
    "processed_Openadmet_REAL_PXR_train_AND_test_main_with_side_info_"
    "AND_counter_screen_weighted.csv"
)
TEST_PATH      = Path.home() / "dockerimages/QSARTuna/PXR/Prediction_OpenAdmet_ChemProp_Only_OpenADMET_Data.csv"
MODEL_PKL_PATH = Path.home() / "pxr_chemprop_rdkit2d_stderr_weight_final.pkl"
OUTPUT_PREDS   = Path.home() / "pxr_rdkit2d_stderr_weight_test_predictions.csv"
KEPT_DESCS_PATH = Path.home() / "pxr_rdkit2d_stderr_weight_kept_descriptors.txt"

TRAIN_SMILES_COL = "SMILES"
TRAIN_TARGET_COL = "pEC50"
TEST_SMILES_COL  = "SMILES"
TEST_TARGET_COL  = "pEC50"
TEST_NAME_COL    = "Molecule Name"
STD_ERROR_COL    = "std_error"

NUM_WORKERS  = 0
MIN_STD_ERROR = 0.05

# Slow LR — same as the full script
INIT_LR  = 1e-4
MAX_LR   = 2e-4
FINAL_LR = 1e-5

# ── Best hyperparameters from CV (mean_val_RMSE = 0.4380) ─────────────────────
BEST_PARAMS = {
    "ffn_hidden_dim": 512,
    "ffn_n_layers":   3,
    "dropout":        0.0,
    "mp_depth":       4,
    "mp_hidden_dim":  1024,
}
FINAL_EPOCHS = 15   # max(int(14 * 1.1), 5)


# ── Sample weight from measurement uncertainty ────────────────────────────────
def compute_weights(std_error_values: np.ndarray) -> np.ndarray:
    clipped = np.clip(std_error_values, MIN_STD_ERROR, None)
    raw     = 1.0 / clipped
    return raw / raw.mean()


# ── RDKit 2D descriptor utilities ─────────────────────────────────────────────
_DESC_LIST     = [(name, fn) for name, fn in Descriptors.descList]
ALL_DESC_NAMES = [name for name, _ in _DESC_LIST]


def compute_rdkit_descriptors(mols: list) -> np.ndarray:
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
    finite = np.all(np.isfinite(arr), axis=0)
    varied = np.var(arr, axis=0) > 0
    return finite & varied


# ── Build MPNN ─────────────────────────────────────────────────────────────────
def build_mpnn(n_descriptors: int, ffn_hidden_dim: int, ffn_n_layers: int,
               dropout: float, mp_depth: int, mp_hidden_dim: int,
               target_scaler) -> models.MPNN:
    feat = featurizers.SimpleMoleculeMolGraphFeaturizer()
    mp   = nn.BondMessagePassing(
               d_v=feat.atom_fdim, d_e=feat.bond_fdim,
               depth=mp_depth, d_h=mp_hidden_dim,
           )
    agg  = nn.MeanAggregation()
    output_transform = nn.UnscaleTransform.from_standard_scaler(target_scaler)
    ffn  = nn.RegressionFFN(
        input_dim=mp.output_dim + n_descriptors,
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
    return [
        data.MoleculeDatapoint(
            mol=mol,
            y=np.array([t], dtype=float),
            weight=float(w),
            x_d=xd.astype(float),
        )
        for mol, t, w, xd in zip(mols, targets, weights, x_d)
    ]


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — Load and parse training data
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nLoading training data:\n  {TRAIN_PATH}")
df_train = pd.read_csv(TRAIN_PATH)
print(f"  {len(df_train)} rows")

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

train_targets        = np.array(train_targets)
train_stderr         = np.array(train_stderr, dtype=float)
train_weights_stderr = compute_weights(train_stderr)

print(f"  Usable training molecules : {len(train_mols)}")
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
x_d_train_all = compute_rdkit_descriptors(train_mols)

col_mask      = select_valid_columns(x_d_train_all)
kept_names    = [ALL_DESC_NAMES[i] for i, k in enumerate(col_mask) if k]
x_d_train_raw = x_d_train_all[:, col_mask]
n_descriptors = x_d_train_raw.shape[1]
print(f"  Kept {n_descriptors} / {len(col_mask)} descriptors")

with open(KEPT_DESCS_PATH, "w") as fh:
    fh.write("\n".join(kept_names))

print("\nComputing RDKit 2D descriptors for test set...")
x_d_test_all = compute_rdkit_descriptors(test_mols)
x_d_test_raw = x_d_test_all[:, col_mask]

# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — Train final model on ALL training data
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"Best hyperparameters (from CV): {BEST_PARAMS}")
print(f"Final training epochs         : {FINAL_EPOCHS}")
print(f"Training on all {len(train_mols)} compounds")
print(f"{'='*60}")

final_xd_scaler  = StandardScaler()
x_d_train_scaled = final_xd_scaler.fit_transform(x_d_train_raw)
x_d_test_scaled  = final_xd_scaler.transform(x_d_test_raw)

feat = featurizers.SimpleMoleculeMolGraphFeaturizer()

all_train_dps  = make_datapoints(train_mols, train_targets, train_weights_stderr, x_d_train_scaled)
all_train_dset = data.MoleculeDataset(all_train_dps, feat)
final_scaler   = all_train_dset.normalize_targets()
final_loader   = data.build_dataloader(all_train_dset, num_workers=NUM_WORKERS)

final_mpnn = build_mpnn(n_descriptors, **BEST_PARAMS, target_scaler=final_scaler)
print("\nFinal model architecture:")
print(final_mpnn)

final_trainer = pl.Trainer(
    logger=False,
    enable_checkpointing=False,
    enable_progress_bar=True,
    accelerator="auto",
    devices=1,
    max_epochs=FINAL_EPOCHS,
)
final_trainer.fit(final_mpnn, final_loader)

# ══════════════════════════════════════════════════════════════════════════════
# Step 5 — Save model
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nSaving model to {MODEL_PKL_PATH} ...")
torch.save(final_mpnn, MODEL_PKL_PATH)
print("  Saved.")

# ══════════════════════════════════════════════════════════════════════════════
# Step 6 — Predict on external test set
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nPredicting on test set ({len(test_mols)} molecules)...")
final_mpnn.eval()

test_weights_dummy = np.ones(len(test_mols))
test_dps    = make_datapoints(test_mols, test_targets, test_weights_dummy, x_d_test_scaled)
test_dset   = data.MoleculeDataset(test_dps, feat)
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
