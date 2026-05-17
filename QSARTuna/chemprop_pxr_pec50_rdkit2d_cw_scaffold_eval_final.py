"""
ChemProp PXR pEC50 — Scaffold 80/10/10 evaluation then final model on ALL data.

Purpose
───────
Before committing to a submission it is useful to have an unbiased estimate of
model performance on compounds the model has never seen.  This script does that
with a scaffold-aware three-way split, then retrains on all data for the
competition submission.

Two-phase workflow
──────────────────
Phase 1 · Internal evaluation  (scaffold 80 / 10 / 10 split)
  • Bemis-Murcko scaffold groups are kept whole, so val and internal-test
    contain only scaffolds absent from the training set — the most realistic
    proxy for blind-test difficulty.
  • Train on 80 %, use the 10 % val fold for early stopping, then report
    RMSE / MAE / Pearson r / Spearman ρ on the held-out 10 % test fold.
  • Descriptor scaler is fit on the 80 % train split only (no leakage).
  • Saves per-compound internal test predictions for post-hoc analysis.

Phase 2 · Final submission model  (all 4,140 compounds)
  • Retrain with identical hyperparameters on 100 % of training data.
  • No early stopping — fixed epoch budget (FINAL_EPOCHS = 150).
  • Descriptor scaler refit on all training data.
  • Predict competition test.csv and write submission CSV.

Model settings (best from previous stratified-CV HPO)
  ffn_hidden_dim = 300   ffn_n_layers = 3   dropout = 0.2
  mp_depth       = 4     mp_hidden_dim = 1024

Weighting scheme (unchanged from rank-~50 model)
  pEC50_counter_min → weight 2.0  (selective, trust more)
  pEC50_counter_max → weight 0.5  (promiscuous, trust less)
  NaN               → weight 1.0  (no counter data)

Output
  ~/OpenAdmet/eval_internal_test_predictions.csv   — Phase-1 held-out metrics
  ~/OpenAdmet/Submission_CW_Scaffold_EvalFinal.csv — Phase-2 competition submission
  ~/pxr_rdkit2d_cw_scaffold_eval_final.pt          — saved final model
  ~/pxr_rdkit2d_cw_scaffold_eval_final_descs.txt   — kept descriptor names

Usage
  conda activate chemprop
  python ~/dockerimages/QSARTuna/chemprop_pxr_pec50_rdkit2d_cw_scaffold_eval_final.py
"""

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightning import pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

from chemprop import data, featurizers, models, nn

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_CSV   = Path(
    "/home/spal/dockerimages/QSARTuna/PXR_For_QSARTuna_Modelling/"
    "processed_Openadmet_REAL_PXR_train_AND_test_main_with_side_info_"
    "AND_counter_screen_weighted.csv"
)
TEST_CSV          = Path.home() / "dockerimages/QSARTuna/PXR/test.csv"
MODEL_OUT         = Path.home() / "pxr_rdkit2d_cw_scaffold_eval_final.pt"
DESCS_OUT         = Path.home() / "pxr_rdkit2d_cw_scaffold_eval_final_descs.txt"
INTERNAL_PRED_OUT = Path.home() / "OpenAdmet/eval_internal_test_predictions.csv"
SUBMISSION_OUT    = Path.home() / "OpenAdmet/Submission_CW_Scaffold_EvalFinal.csv"

SMILES_COL  = "SMILES"
TARGET_COL  = "pEC50"
COUNTER_COL = "pEC50_counter"
TEST_ID_COL = "Molecule Name"

# ─────────────────────────────────────────────────────────────────────────────
# Counter-screen weighting
# ─────────────────────────────────────────────────────────────────────────────
MIN_WEIGHT     = 0.5
MAX_WEIGHT     = 2.0
NEUTRAL_WEIGHT = 1.0

# ─────────────────────────────────────────────────────────────────────────────
# Architecture (best from previous HPO)
# ─────────────────────────────────────────────────────────────────────────────
FFN_HIDDEN_DIM = 300
FFN_N_LAYERS   = 3
DROPOUT        = 0.2
MP_DEPTH       = 4
MP_HIDDEN_DIM  = 1024

# ─────────────────────────────────────────────────────────────────────────────
# Training settings
# ─────────────────────────────────────────────────────────────────────────────
INIT_LR     = 1e-4
MAX_LR      = 2e-4
FINAL_LR    = 1e-5
NUM_WORKERS = 0

# Phase 1 — evaluation model uses early stopping
EVAL_MAX_EPOCHS = 200
EVAL_PATIENCE   = 30

# Phase 2 — final model trains for a fixed budget on all data
FINAL_EPOCHS = 150

# Scaffold split proportions
TRAIN_FRAC = 0.8
VAL_FRAC   = 0.1
# test_frac  = 0.1 (remainder)


# ─────────────────────────────────────────────────────────────────────────────
# Callback: record the epoch with the lowest validation loss
# ─────────────────────────────────────────────────────────────────────────────
class BestEpochTracker(pl.Callback):
    def __init__(self):
        self.best_val_loss = float("inf")
        self.best_epoch    = 0

    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = float(trainer.callback_metrics.get("val_loss", float("inf")))
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch    = trainer.current_epoch


# ─────────────────────────────────────────────────────────────────────────────
# RDKit 2D descriptor helpers
# ─────────────────────────────────────────────────────────────────────────────
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
    """Keep columns that are finite for ALL molecules and have nonzero variance.
    Always derived from training data only to prevent leakage."""
    finite = np.all(np.isfinite(arr), axis=0)
    varied = np.var(arr, axis=0) > 0
    return finite & varied


# ─────────────────────────────────────────────────────────────────────────────
# Counter-screen weighting
# ─────────────────────────────────────────────────────────────────────────────
def compute_weights(counter_values: np.ndarray) -> np.ndarray:
    """Inverse linear map: low counter potency → high weight (more selective)."""
    weights     = np.full(len(counter_values), NEUTRAL_WEIGHT, dtype=float)
    has_counter = ~np.isnan(counter_values)
    vals        = counter_values[has_counter]
    c_min, c_max = vals.min(), vals.max()
    weights[has_counter] = MIN_WEIGHT + (MAX_WEIGHT - MIN_WEIGHT) * (
        (c_max - vals) / (c_max - c_min)
    )
    return weights


# ─────────────────────────────────────────────────────────────────────────────
# Scaffold three-way split  (80 / 10 / 10)
# ─────────────────────────────────────────────────────────────────────────────
def scaffold_three_way_split(
    mols: list,
    train_frac: float = TRAIN_FRAC,
    val_frac:   float = VAL_FRAC,
    seed:       int   = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split molecule indices into train / val / test by Bemis-Murcko scaffold.

    All molecules sharing the same scaffold land in the same partition, so
    val and test contain only scaffolds unseen during training.  Scaffold
    groups are assigned greedily to the bucket furthest below its target
    size, processing the largest groups first.

    Returns
    -------
    train_idx, val_idx, test_idx : np.ndarray of int
    """
    scaffold_to_idx: dict[str, list[int]] = defaultdict(list)
    for i, mol in enumerate(mols):
        smi = Chem.MolToSmiles(mol)
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                smiles=smi, includeChirality=False
            )
        except Exception:
            scaffold = smi          # fall back to the full SMILES
        scaffold_to_idx[scaffold].append(i)

    n            = len(mols)
    train_target = int(train_frac * n)
    val_target   = int(val_frac * n)
    test_target  = n - train_target - val_target

    groups  = sorted(scaffold_to_idx.values(), key=len, reverse=True)
    buckets = [[], [], []]          # train, val, test
    counts  = [0, 0, 0]
    targets = [train_target, val_target, test_target]

    for group in groups:
        # Assign each scaffold group to whichever bucket is furthest below its target
        deficits = [targets[b] - counts[b] for b in range(3)]
        bucket   = int(np.argmax(deficits))
        buckets[bucket].extend(group)
        counts[bucket] += len(group)

    return (
        np.array(buckets[0]),
        np.array(buckets[1]),
        np.array(buckets[2]),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Model builder
# ─────────────────────────────────────────────────────────────────────────────
def build_mpnn(n_descriptors: int, target_scaler) -> models.MPNN:
    feat = featurizers.SimpleMoleculeMolGraphFeaturizer()
    mp   = nn.BondMessagePassing(
               d_v=feat.atom_fdim, d_e=feat.bond_fdim,
               depth=MP_DEPTH, d_h=MP_HIDDEN_DIM,
           )
    agg  = nn.MeanAggregation()
    ffn  = nn.RegressionFFN(
               input_dim=mp.output_dim + n_descriptors,
               hidden_dim=FFN_HIDDEN_DIM,
               n_layers=FFN_N_LAYERS,
               dropout=DROPOUT,
               criterion=nn.metrics.MAE(),
               output_transform=nn.UnscaleTransform.from_standard_scaler(target_scaler),
           )
    return models.MPNN(
        mp, agg, ffn,
        batch_norm=True,
        metrics=[nn.metrics.RMSE(), nn.metrics.MAE()],
        init_lr=INIT_LR, max_lr=MAX_LR, final_lr=FINAL_LR,
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


def report_metrics(label: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse       = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae        = float(mean_absolute_error(y_true, y_pred))
    r, _       = pearsonr(y_true, y_pred)
    rho, _     = spearmanr(y_true, y_pred)
    print(f"  {label}")
    print(f"    RMSE      = {rmse:.4f}")
    print(f"    MAE       = {mae:.4f}")
    print(f"    Pearson r = {r:.4f}")
    print(f"    Spearman ρ= {rho:.4f}")
    return dict(rmse=rmse, mae=mae, pearson_r=r, spearman_rho=rho)


# ═════════════════════════════════════════════════════════════════════════════
# Step 1 — Load training data
# ═════════════════════════════════════════════════════════════════════════════
print(f"\nLoading training data:\n  {TRAIN_CSV}")
df_train = pd.read_csv(TRAIN_CSV)
print(f"  {len(df_train)} rows")

counter_values = df_train[COUNTER_COL].values
weights_all    = compute_weights(counter_values)
n_counter      = (~np.isnan(counter_values)).sum()
print(f"  Counter screen: {n_counter}/{len(df_train)} compounds "
      f"({100*n_counter/len(df_train):.1f}%)")
print(f"  Weight range  : {weights_all.min():.3f} – {weights_all.max():.3f}")

print("\nParsing SMILES...")
all_mols, all_smiles, all_targets, all_weights = [], [], [], []
for smi, y, w in zip(df_train[SMILES_COL].values,
                     df_train[TARGET_COL].values,
                     weights_all):
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        all_mols.append(mol)
        all_smiles.append(smi)
        all_targets.append(y)
        all_weights.append(w)
    else:
        print(f"  Skipped: {smi[:60]}")

all_targets = np.array(all_targets)
all_weights = np.array(all_weights)
print(f"  Usable molecules: {len(all_mols)}")

# ═════════════════════════════════════════════════════════════════════════════
# Step 2 — Load competition test set
# ═════════════════════════════════════════════════════════════════════════════
print(f"\nLoading competition test set:\n  {TEST_CSV}")
df_test = pd.read_csv(TEST_CSV)
test_mols, test_smiles, test_names = [], [], []
for _, row in df_test.iterrows():
    mol = Chem.MolFromSmiles(row[SMILES_COL])
    if mol is not None:
        test_mols.append(mol)
        test_smiles.append(row[SMILES_COL])
        test_names.append(row[TEST_ID_COL])
print(f"  Test molecules: {len(test_mols)}")

# ═════════════════════════════════════════════════════════════════════════════
# Step 3 — RDKit 2D descriptors
# Column mask is derived from ALL training data (not just the 80 % split) so
# the same descriptor set is used in both phases.
# ═════════════════════════════════════════════════════════════════════════════
print("\nComputing RDKit 2D descriptors for training set...")
x_d_all_raw = compute_rdkit_descriptors(all_mols)
col_mask    = select_valid_columns(x_d_all_raw)
kept_names  = [ALL_DESC_NAMES[i] for i, k in enumerate(col_mask) if k]
x_d_all_raw = x_d_all_raw[:, col_mask]
n_desc      = x_d_all_raw.shape[1]
print(f"  Kept {n_desc} / {len(col_mask)} descriptors")

DESCS_OUT.parent.mkdir(parents=True, exist_ok=True)
DESCS_OUT.write_text("\n".join(kept_names))
print(f"  Descriptor list → {DESCS_OUT}")

print("Computing RDKit 2D descriptors for test set...")
x_d_test_raw = compute_rdkit_descriptors(test_mols)[:, col_mask]

# ═════════════════════════════════════════════════════════════════════════════
# Step 4 — Scaffold 80 / 10 / 10 split
# ═════════════════════════════════════════════════════════════════════════════
print(f"\nBuilding scaffold 80/10/10 split (seed=42)...")
tr_idx, va_idx, te_idx = scaffold_three_way_split(all_mols, seed=42)
n = len(all_mols)
print(f"  Train : {len(tr_idx):4d}  ({100*len(tr_idx)/n:.1f}%)")
print(f"  Val   : {len(va_idx):4d}  ({100*len(va_idx)/n:.1f}%)")
print(f"  Test  : {len(te_idx):4d}  ({100*len(te_idx)/n:.1f}%)  ← scaffold-held-out")

# ═════════════════════════════════════════════════════════════════════════════
# Step 5 — Phase 1: evaluation model
#   • Descriptor scaler fit on 80 % train split only
#   • Early stopping on 10 % val
#   • Metrics reported on 10 % internal test (never seen during training)
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*62}")
print(f"PHASE 1 — Evaluation model  (scaffold 80/10/10)")
print(f"  max_epochs={EVAL_MAX_EPOCHS}  patience={EVAL_PATIENCE}")
print(f"{'='*62}")

# Fit scaler on 80 % train only — applying to val/test without refitting
eval_scaler = StandardScaler()
x_d_tr = eval_scaler.fit_transform(x_d_all_raw[tr_idx])
x_d_va = eval_scaler.transform(x_d_all_raw[va_idx])
x_d_te = eval_scaler.transform(x_d_all_raw[te_idx])

feat = featurizers.SimpleMoleculeMolGraphFeaturizer()

tr_dps = make_datapoints([all_mols[i] for i in tr_idx],
                         all_targets[tr_idx], all_weights[tr_idx], x_d_tr)
va_dps = make_datapoints([all_mols[i] for i in va_idx],
                         all_targets[va_idx], all_weights[va_idx], x_d_va)
te_dps = make_datapoints([all_mols[i] for i in te_idx],
                         all_targets[te_idx], all_weights[te_idx], x_d_te)

tr_dset = data.MoleculeDataset(tr_dps, feat)
va_dset = data.MoleculeDataset(va_dps, feat)
te_dset = data.MoleculeDataset(te_dps, feat)

# Target normalisation derived from the 80 % train set — applied to val too
phase1_target_scaler = tr_dset.normalize_targets()
va_dset.normalize_targets(phase1_target_scaler)

tr_loader = data.build_dataloader(tr_dset, num_workers=NUM_WORKERS)
va_loader = data.build_dataloader(va_dset, num_workers=NUM_WORKERS, shuffle=False)
te_loader = data.build_dataloader(te_dset, num_workers=NUM_WORKERS, shuffle=False)

eval_mpnn    = build_mpnn(n_desc, phase1_target_scaler)
best_tracker = BestEpochTracker()

eval_trainer = pl.Trainer(
    logger=False,
    enable_checkpointing=False,
    enable_progress_bar=True,
    accelerator="auto",
    devices=1,
    max_epochs=EVAL_MAX_EPOCHS,
    callbacks=[
        EarlyStopping(monitor="val_loss", patience=EVAL_PATIENCE, mode="min"),
        best_tracker,
    ],
)
eval_trainer.fit(eval_mpnn, tr_loader, va_loader)
print(f"\n  Stopped at epoch {best_tracker.best_epoch}  "
      f"(best val_loss = {best_tracker.best_val_loss:.4f})")

# Predict on the held-out 10 % internal test set
eval_mpnn.eval()
raw_te   = eval_trainer.predict(eval_mpnn, te_loader)
te_preds = torch.cat(raw_te).numpy().flatten()
te_true  = all_targets[te_idx]

print(f"\n  Internal test metrics  ({len(te_idx)} scaffold-held-out compounds):")
report_metrics("ChemProp + RDKit2D + CW  [scaffold 10% test]", te_true, te_preds)

# Save per-compound predictions for inspection
df_internal = pd.DataFrame({
    "SMILES":     [all_smiles[i] for i in te_idx],
    "pEC50_true": te_true,
    "pEC50_pred": te_preds,
    "residual":   te_true - te_preds,
})
df_internal.to_csv(INTERNAL_PRED_OUT, index=False)
print(f"\n  Internal predictions → {INTERNAL_PRED_OUT}")

# ═════════════════════════════════════════════════════════════════════════════
# Step 6 — Phase 2: final model on ALL 4,140 compounds
#   • Fresh descriptor scaler fit on all training data
#   • Fixed epoch budget, no early stopping
# ═════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*62}")
print(f"PHASE 2 — Final model on ALL {len(all_mols)} compounds")
print(f"  epochs={FINAL_EPOCHS}  (no early stopping)")
print(f"{'='*62}")

final_xd_scaler  = StandardScaler()
x_d_train_scaled = final_xd_scaler.fit_transform(x_d_all_raw)
x_d_test_scaled  = final_xd_scaler.transform(x_d_test_raw)

all_dps  = make_datapoints(all_mols, all_targets, all_weights, x_d_train_scaled)
all_dset = data.MoleculeDataset(all_dps, feat)
final_target_scaler = all_dset.normalize_targets()
final_loader        = data.build_dataloader(all_dset, num_workers=NUM_WORKERS)

final_mpnn = build_mpnn(n_desc, final_target_scaler)

final_trainer = pl.Trainer(
    logger=False,
    enable_checkpointing=False,
    enable_progress_bar=True,
    accelerator="auto",
    devices=1,
    max_epochs=FINAL_EPOCHS,
)
final_trainer.fit(final_mpnn, final_loader)

# ═════════════════════════════════════════════════════════════════════════════
# Step 7 — Save final model
# ═════════════════════════════════════════════════════════════════════════════
torch.save(final_mpnn, MODEL_OUT)
print(f"\nFinal model → {MODEL_OUT}")

# ═════════════════════════════════════════════════════════════════════════════
# Step 8 — Predict on competition test set and write submission
# ═════════════════════════════════════════════════════════════════════════════
print(f"\nPredicting {len(test_mols)} competition test molecules...")
final_mpnn.eval()

test_dps  = make_datapoints(test_mols,
                            np.zeros(len(test_mols)),
                            np.ones(len(test_mols)),
                            x_d_test_scaled)
test_dset   = data.MoleculeDataset(test_dps, feat)
test_loader = data.build_dataloader(test_dset, num_workers=NUM_WORKERS, shuffle=False)

raw_preds = final_trainer.predict(final_mpnn, test_loader)
preds     = torch.cat(raw_preds).numpy().flatten()

df_submission = pd.DataFrame({
    "Molecule Name": test_names,
    "SMILES":        test_smiles,
    "pEC50":         preds,
})
SUBMISSION_OUT.parent.mkdir(parents=True, exist_ok=True)
df_submission.to_csv(SUBMISSION_OUT, index=False)
print(f"Submission → {SUBMISSION_OUT}")
print(f"\nFirst 5 predictions:")
print(df_submission.head().to_string(index=False))
print(f"\npEC50 summary:")
print(df_submission["pEC50"].describe().to_string())
