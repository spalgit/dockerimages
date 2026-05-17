"""
ChemProp PXR pEC50 — Final model on ALL training compounds.

Identical feature engineering and weighting to
chemprop_pxr_pec50_rdkit2d_counter_weight_scaffold_cv.py
(RDKit 2D descriptors, counter-assay weighting, MAE loss, slow LR)
but skips the CV grid search entirely.

Hyperparameters are set to the ChemProp-recommended defaults which also
lie inside the CV param grid.  The final model trains for FINAL_EPOCHS on
all 4,140 training compounds; there is no held-out validation set.

Output:
  ~/pxr_rdkit2d_cw_scaffold_finalonly.pt         — saved model (torch.save)
  ~/pxr_rdkit2d_cw_scaffold_finalonly_descs.txt  — descriptor names kept
  ~/OpenAdmet/Submission_CW_Scaffold_FinalOnly.csv — competition submission

Usage:
    conda activate chemprop
    python ~/dockerimages/QSARTuna/chemprop_pxr_pec50_rdkit2d_counter_weight_scaffold_finalonly.py
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

# ── Paths ───────────────────────────────────────────────────────────────────────
TRAIN_PATH      = Path(
    "/home/spal/dockerimages/QSARTuna/PXR_For_QSARTuna_Modelling/"
    "processed_Openadmet_REAL_PXR_train_AND_test_main_with_side_info_"
    "AND_counter_screen_weighted.csv"
)
TEST_PATH       = Path.home() / "dockerimages/QSARTuna/PXR/test.csv"
MODEL_PATH      = Path.home() / "pxr_rdkit2d_cw_scaffold_finalonly.pt"
KEPT_DESCS_PATH = Path.home() / "pxr_rdkit2d_cw_scaffold_finalonly_descs.txt"
SUBMISSION_PATH = Path.home() / "OpenAdmet/Submission_CW_Scaffold_FinalOnly.csv"

TRAIN_SMILES_COL = "SMILES"
TRAIN_TARGET_COL = "pEC50"
COUNTER_COL      = "pEC50_counter"
TEST_SMILES_COL  = "SMILES"
TEST_NAME_COL    = "Molecule Name"

# ── Weighting ───────────────────────────────────────────────────────────────────
MIN_WEIGHT     = 0.5
MAX_WEIGHT     = 2.0
NEUTRAL_WEIGHT = 1.0

# ── Final-model hyperparameters ─────────────────────────────────────────────────
# These sit inside the scaffold CV param grid and are the ChemProp defaults.
# Chosen to match the architecture family that produced the rank-~50 result.
FFN_HIDDEN_DIM = 300
FFN_N_LAYERS   = 2
DROPOUT        = 0.0
MP_DEPTH       = 3
MP_HIDDEN_DIM  = 300

FINAL_EPOCHS = 150   # slow LR (max 2e-4) needs many steps; no early stopping
NUM_WORKERS  = 0

INIT_LR  = 1e-4
MAX_LR   = 2e-4
FINAL_LR = 1e-5

# ── RDKit descriptor helpers ────────────────────────────────────────────────────
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


# ── Counter-screen weighting ─────────────────────────────────────────────────────
def compute_weights(counter_values: np.ndarray) -> np.ndarray:
    weights     = np.full(len(counter_values), NEUTRAL_WEIGHT, dtype=float)
    has_counter = ~np.isnan(counter_values)
    vals        = counter_values[has_counter]
    c_min, c_max = vals.min(), vals.max()
    weights[has_counter] = MIN_WEIGHT + (MAX_WEIGHT - MIN_WEIGHT) * (
        (c_max - vals) / (c_max - c_min)
    )
    return weights


# ── MPNN factory ────────────────────────────────────────────────────────────────
def build_mpnn(n_descriptors: int, target_scaler) -> models.MPNN:
    feat = featurizers.SimpleMoleculeMolGraphFeaturizer()
    mp   = nn.BondMessagePassing(
               d_v=feat.atom_fdim, d_e=feat.bond_fdim,
               depth=MP_DEPTH, d_h=MP_HIDDEN_DIM,
           )
    agg  = nn.MeanAggregation()
    output_transform = nn.UnscaleTransform.from_standard_scaler(target_scaler)
    ffn  = nn.RegressionFFN(
               input_dim=mp.output_dim + n_descriptors,
               hidden_dim=FFN_HIDDEN_DIM,
               n_layers=FFN_N_LAYERS,
               dropout=DROPOUT,
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
# Step 1 — Load training data
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nLoading training data:\n  {TRAIN_PATH}")
df_train = pd.read_csv(TRAIN_PATH)
print(f"  {len(df_train)} rows")

counter_values = df_train[COUNTER_COL].values
train_weights  = compute_weights(counter_values)
n_with_counter = (~np.isnan(counter_values)).sum()
print(f"  Counter screen available : {n_with_counter}/{len(df_train)} "
      f"({100*n_with_counter/len(df_train):.1f}%)")
print(f"  Weight range             : "
      f"{train_weights.min():.3f} – {train_weights.max():.3f}")

print("\nParsing training SMILES...")
train_mols, train_smiles, train_targets, train_weights_clean = [], [], [], []
for smi, y, w in zip(df_train[TRAIN_SMILES_COL].values,
                     df_train[TRAIN_TARGET_COL].values,
                     train_weights):
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        train_mols.append(mol)
        train_smiles.append(smi)
        train_targets.append(y)
        train_weights_clean.append(w)
    else:
        print(f"  Skipped unparseable SMILES: {smi[:50]}")

train_targets       = np.array(train_targets)
train_weights_clean = np.array(train_weights_clean)
print(f"  Usable training molecules: {len(train_mols)}")

# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — Load test set
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nLoading test set:\n  {TEST_PATH}")
df_test = pd.read_csv(TEST_PATH)
test_mols, test_smiles, test_names = [], [], []
for _, row in df_test.iterrows():
    mol = Chem.MolFromSmiles(row[TEST_SMILES_COL])
    if mol is not None:
        test_mols.append(mol)
        test_smiles.append(row[TEST_SMILES_COL])
        test_names.append(row[TEST_NAME_COL])
print(f"  Usable test molecules: {len(test_mols)}")

# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — RDKit 2D descriptors (column mask from training set only)
# ══════════════════════════════════════════════════════════════════════════════
print("\nComputing RDKit 2D descriptors for training set...")
x_d_train_all = compute_rdkit_descriptors(train_mols)
col_mask      = select_valid_columns(x_d_train_all)
kept_names    = [ALL_DESC_NAMES[i] for i, k in enumerate(col_mask) if k]
print(f"  Kept {col_mask.sum()} / {len(col_mask)} descriptors "
      f"(dropped {(~col_mask).sum()} NaN/inf/zero-var)")

x_d_train_raw = x_d_train_all[:, col_mask]
n_descriptors = x_d_train_raw.shape[1]

with open(KEPT_DESCS_PATH, "w") as fh:
    fh.write("\n".join(kept_names))
print(f"  Descriptor list saved to {KEPT_DESCS_PATH}")

print("Computing RDKit 2D descriptors for test set...")
x_d_test_all = compute_rdkit_descriptors(test_mols)
x_d_test_raw = x_d_test_all[:, col_mask]

# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — Scale descriptors on all training data; apply to test
# ══════════════════════════════════════════════════════════════════════════════
print("\nScaling RDKit 2D descriptors...")
xd_scaler        = StandardScaler()
x_d_train_scaled = xd_scaler.fit_transform(x_d_train_raw)
x_d_test_scaled  = xd_scaler.transform(x_d_test_raw)

# ══════════════════════════════════════════════════════════════════════════════
# Step 5 — Build dataset and train final model on ALL compounds
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*62}")
print(f"Training final model on ALL {len(train_mols)} compounds")
print(f"  ffn_hidden_dim={FFN_HIDDEN_DIM}  ffn_n_layers={FFN_N_LAYERS}  "
      f"dropout={DROPOUT}")
print(f"  mp_depth={MP_DEPTH}  mp_hidden_dim={MP_HIDDEN_DIM}")
print(f"  epochs={FINAL_EPOCHS}  LR: {INIT_LR}→{MAX_LR}→{FINAL_LR}")
print(f"{'='*62}")

feat = featurizers.SimpleMoleculeMolGraphFeaturizer()

all_dps   = make_datapoints(train_mols, train_targets, train_weights_clean, x_d_train_scaled)
all_dset  = data.MoleculeDataset(all_dps, feat)
target_scaler = all_dset.normalize_targets()
train_loader  = data.build_dataloader(all_dset, num_workers=NUM_WORKERS)

mpnn = build_mpnn(n_descriptors, target_scaler)
print("\nArchitecture:")
print(mpnn)

trainer = pl.Trainer(
    logger=False,
    enable_checkpointing=False,
    enable_progress_bar=True,
    accelerator="auto",
    devices=1,
    max_epochs=FINAL_EPOCHS,
)
trainer.fit(mpnn, train_loader)

# ══════════════════════════════════════════════════════════════════════════════
# Step 6 — Save model
# ══════════════════════════════════════════════════════════════════════════════
torch.save(mpnn, MODEL_PATH)
print(f"\nModel saved to {MODEL_PATH}")

# ══════════════════════════════════════════════════════════════════════════════
# Step 7 — Predict on test set and write submission
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nPredicting on {len(test_mols)} test molecules...")
mpnn.eval()

test_dps   = make_datapoints(test_mols,
                             np.zeros(len(test_mols)),
                             np.ones(len(test_mols)),
                             x_d_test_scaled)
test_dset   = data.MoleculeDataset(test_dps, feat)
test_loader = data.build_dataloader(test_dset, num_workers=NUM_WORKERS, shuffle=False)

raw_preds = trainer.predict(mpnn, test_loader)
preds     = torch.cat(raw_preds).numpy().flatten()

df_submission = pd.DataFrame({
    "Molecule Name": test_names,
    "SMILES":        test_smiles,
    "pEC50":         preds,
})
df_submission.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission saved to {SUBMISSION_PATH}")
print(f"\nFirst 5 predictions:")
print(df_submission.head().to_string(index=False))
print(f"\npEC50 summary:")
print(df_submission["pEC50"].describe().to_string())
