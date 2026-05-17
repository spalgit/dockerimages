"""
ChemProp PXR pEC50 — Final model on ALL training compounds.
RDKit 2D descriptors + counter-assay weighting + best HPO architecture.

Two-phase workflow
──────────────────
Phase 1 — Internal evaluation (80 / 10 / 10 scaffold split)
  • Scaffold-aware three-way split keeps every Bemis-Murcko scaffold whole
    within one partition, so val and internal-test contain only scaffolds
    unseen during training — a realistic estimate of blind-test performance.
  • Train on 80 %, early-stop on 10 % val, report RMSE / MAE / r on 10 % test.
  • Purpose: decide whether the model is good enough to submit.

Phase 2 — Final submission model (all 4,140 compounds)
  • Retrain on 100 % of training data for FINAL_EPOCHS (no early stopping).
  • Predict competition test.csv and write submission CSV.

Hyperparameters (best from previous stratified-CV HPO):
  ffn_hidden_dim=300  ffn_n_layers=3  dropout=0.2
  mp_depth=4          mp_hidden_dim=1024

Output:
  ~/OpenAdmet/Submission_CW_Scaffold_FinalOnly.csv   — competition submission
  ~/pxr_rdkit2d_cw_scaffold_finalonly.pt             — saved final model
  ~/pxr_rdkit2d_cw_scaffold_finalonly_descs.txt      — kept descriptor names
  ~/OpenAdmet/internal_test_predictions.csv          — Phase-1 metrics

Usage:
    conda activate chemprop
    python ~/dockerimages/QSARTuna/chemprop_pxr_pec50_rdkit2d_counter_weight_scaffold_finalonly.py
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

# ── Paths ────────────────────────────────────────────────────────────────────────
TRAIN_PATH      = Path(
    "/home/spal/dockerimages/QSARTuna/PXR_For_QSARTuna_Modelling/"
    "processed_Openadmet_REAL_PXR_train_AND_test_main_with_side_info_"
    "AND_counter_screen_weighted.csv"
)
TEST_PATH        = Path.home() / "dockerimages/QSARTuna/PXR/test.csv"
MODEL_PATH       = Path.home() / "pxr_rdkit2d_cw_scaffold_finalonly.pt"
KEPT_DESCS_PATH  = Path.home() / "pxr_rdkit2d_cw_scaffold_finalonly_descs.txt"
SUBMISSION_PATH  = Path.home() / "OpenAdmet/Submission_CW_Scaffold_FinalOnly.csv"
INTERNAL_PRED_PATH = Path.home() / "OpenAdmet/internal_test_predictions.csv"

TRAIN_SMILES_COL = "SMILES"
TRAIN_TARGET_COL = "pEC50"
COUNTER_COL      = "pEC50_counter"
TEST_SMILES_COL  = "SMILES"
TEST_NAME_COL    = "Molecule Name"

# ── Counter-screen weighting ─────────────────────────────────────────────────────
MIN_WEIGHT     = 0.5
MAX_WEIGHT     = 2.0
NEUTRAL_WEIGHT = 1.0

# ── Best HPO hyperparameters (from previous stratified-CV grid search) ───────────
FFN_HIDDEN_DIM = 300
FFN_N_LAYERS   = 3
DROPOUT        = 0.2
MP_DEPTH       = 4
MP_HIDDEN_DIM  = 1024

# ── Training settings ────────────────────────────────────────────────────────────
# Phase 1 (eval model — 80 % data, early stopping)
EVAL_MAX_EPOCHS = 200
EVAL_PATIENCE   = 30

# Phase 2 (final model — 100 % data, fixed budget)
FINAL_EPOCHS    = 150

INIT_LR     = 1e-4
MAX_LR      = 2e-4
FINAL_LR    = 1e-5
NUM_WORKERS = 0

# ── Scaffold 80 / 10 / 10 split fractions ────────────────────────────────────────
TRAIN_FRAC = 0.8
VAL_FRAC   = 0.1
# test_frac  = 0.1 (remainder)


# ── Callback: track best validation epoch ────────────────────────────────────────
class BestEpochTracker(pl.Callback):
    def __init__(self):
        self.best_val_loss = float("inf")
        self.best_epoch    = 0

    def on_validation_epoch_end(self, trainer, pl_module):
        val_loss = float(trainer.callback_metrics.get("val_loss", float("inf")))
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch    = trainer.current_epoch


# ── RDKit 2D descriptor helpers ──────────────────────────────────────────────────
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
    """Column mask: finite for ALL molecules and nonzero variance (train-derived)."""
    finite = np.all(np.isfinite(arr), axis=0)
    varied = np.var(arr, axis=0) > 0
    return finite & varied


# ── Counter-screen weighting ──────────────────────────────────────────────────────
def compute_weights(counter_values: np.ndarray) -> np.ndarray:
    weights     = np.full(len(counter_values), NEUTRAL_WEIGHT, dtype=float)
    has_counter = ~np.isnan(counter_values)
    vals        = counter_values[has_counter]
    c_min, c_max = vals.min(), vals.max()
    weights[has_counter] = MIN_WEIGHT + (MAX_WEIGHT - MIN_WEIGHT) * (
        (c_max - vals) / (c_max - c_min)
    )
    return weights


# ── Scaffold three-way split ──────────────────────────────────────────────────────
def scaffold_three_way_split(
    mols: list,
    train_frac: float = TRAIN_FRAC,
    val_frac: float   = VAL_FRAC,
    seed: int         = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split molecule indices into train / val / test using Bemis-Murcko scaffolds.

    Every molecule sharing the same scaffold lands in the same partition,
    preventing any scaffold from leaking across the train / val / test boundary.
    Scaffold groups are assigned greedily to the bucket furthest below its
    target count, largest groups first.

    Returns (train_idx, val_idx, test_idx) as numpy index arrays.
    """
    scaffold_to_idx: dict[str, list[int]] = defaultdict(list)
    for i, mol in enumerate(mols):
        smi = Chem.MolToSmiles(mol)
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                smiles=smi, includeChirality=False
            )
        except Exception:
            scaffold = smi
        scaffold_to_idx[scaffold].append(i)

    n            = len(mols)
    train_target = int(train_frac * n)
    val_target   = int(val_frac * n)
    test_target  = n - train_target - val_target

    # Shuffle same-size groups for reproducibility without systematic bias
    rng    = np.random.RandomState(seed)
    groups = sorted(scaffold_to_idx.values(), key=len, reverse=True)

    train_idx, val_idx, test_idx = [], [], []
    counts = [0, 0, 0]
    targets = [train_target, val_target, test_target]

    for group in groups:
        deficits = [targets[b] - counts[b] for b in range(3)]
        bucket   = int(np.argmax(deficits))
        if bucket == 0:
            train_idx.extend(group)
        elif bucket == 1:
            val_idx.extend(group)
        else:
            test_idx.extend(group)
        counts[bucket] += len(group)

    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


# ── MPNN factory ─────────────────────────────────────────────────────────────────
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


def print_metrics(label: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r, _ = pearsonr(y_true, y_pred)
    rho, _ = spearmanr(y_true, y_pred)
    print(f"  {label:20s}  RMSE={rmse:.4f}  MAE={mae:.4f}  "
          f"Pearson r={r:.4f}  Spearman ρ={rho:.4f}")


# ══════════════════════════════════════════════════════════════════════════════════
# Step 1 — Load training data
# ══════════════════════════════════════════════════════════════════════════════════
print(f"\nLoading training data:\n  {TRAIN_PATH}")
df_train = pd.read_csv(TRAIN_PATH)
print(f"  {len(df_train)} rows")

counter_values = df_train[COUNTER_COL].values
weights_all    = compute_weights(counter_values)
n_with_counter = (~np.isnan(counter_values)).sum()
print(f"  Counter screen available : {n_with_counter}/{len(df_train)} "
      f"({100*n_with_counter/len(df_train):.1f}%)")
print(f"  Weight range             : {weights_all.min():.3f} – {weights_all.max():.3f}")

print("\nParsing SMILES...")
all_mols, all_smiles, all_targets, all_weights = [], [], [], []
for smi, y, w in zip(df_train[TRAIN_SMILES_COL].values,
                     df_train[TRAIN_TARGET_COL].values,
                     weights_all):
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        all_mols.append(mol)
        all_smiles.append(smi)
        all_targets.append(y)
        all_weights.append(w)
    else:
        print(f"  Skipped: {smi[:50]}")

all_targets = np.array(all_targets)
all_weights = np.array(all_weights)
print(f"  Usable molecules: {len(all_mols)}")

# ══════════════════════════════════════════════════════════════════════════════════
# Step 2 — Load competition test set
# ══════════════════════════════════════════════════════════════════════════════════
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

# ══════════════════════════════════════════════════════════════════════════════════
# Step 3 — RDKit 2D descriptors (column mask derived from ALL training data)
# ══════════════════════════════════════════════════════════════════════════════════
print("\nComputing RDKit 2D descriptors for training set...")
x_d_all_raw = compute_rdkit_descriptors(all_mols)
col_mask    = select_valid_columns(x_d_all_raw)
kept_names  = [ALL_DESC_NAMES[i] for i, k in enumerate(col_mask) if k]
print(f"  Kept {col_mask.sum()} / {len(col_mask)} descriptors")

x_d_all_raw   = x_d_all_raw[:, col_mask]
n_descriptors = x_d_all_raw.shape[1]

with open(KEPT_DESCS_PATH, "w") as fh:
    fh.write("\n".join(kept_names))
print(f"  Descriptor list saved to {KEPT_DESCS_PATH}")

print("Computing RDKit 2D descriptors for test set...")
x_d_test_all = compute_rdkit_descriptors(test_mols)[:, col_mask]

# ══════════════════════════════════════════════════════════════════════════════════
# Step 4 — Scaffold 80 / 10 / 10 split
# ══════════════════════════════════════════════════════════════════════════════════
print(f"\nBuilding scaffold 80/10/10 split...")
tr_idx, va_idx, te_idx = scaffold_three_way_split(all_mols, seed=42)
print(f"  Train : {len(tr_idx):4d} ({100*len(tr_idx)/len(all_mols):.1f}%)")
print(f"  Val   : {len(va_idx):4d} ({100*len(va_idx)/len(all_mols):.1f}%)")
print(f"  Test  : {len(te_idx):4d} ({100*len(te_idx)/len(all_mols):.1f}%)")

# ══════════════════════════════════════════════════════════════════════════════════
# Step 5 — Phase 1: evaluation model (80 % train, early-stop on 10 % val,
#           report metrics on held-out 10 % internal test)
# ══════════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*62}")
print(f"Phase 1 — Evaluation model (scaffold 80/10/10 split)")
print(f"  max_epochs={EVAL_MAX_EPOCHS}  patience={EVAL_PATIENCE}")
print(f"{'='*62}")

# Fit descriptor scaler on the TRAIN split only (no leakage into val / test)
eval_xd_scaler = StandardScaler()
x_d_tr = eval_xd_scaler.fit_transform(x_d_all_raw[tr_idx])
x_d_va = eval_xd_scaler.transform(x_d_all_raw[va_idx])
x_d_te = eval_xd_scaler.transform(x_d_all_raw[te_idx])

feat = featurizers.SimpleMoleculeMolGraphFeaturizer()

tr_dps   = make_datapoints([all_mols[i] for i in tr_idx],
                            all_targets[tr_idx], all_weights[tr_idx], x_d_tr)
va_dps   = make_datapoints([all_mols[i] for i in va_idx],
                            all_targets[va_idx], all_weights[va_idx], x_d_va)
te_dps   = make_datapoints([all_mols[i] for i in te_idx],
                            all_targets[te_idx], all_weights[te_idx], x_d_te)

tr_dset  = data.MoleculeDataset(tr_dps, feat)
va_dset  = data.MoleculeDataset(va_dps, feat)
te_dset  = data.MoleculeDataset(te_dps, feat)

eval_scaler = tr_dset.normalize_targets()
va_dset.normalize_targets(eval_scaler)

tr_loader = data.build_dataloader(tr_dset, num_workers=NUM_WORKERS)
va_loader = data.build_dataloader(va_dset, num_workers=NUM_WORKERS, shuffle=False)
te_loader = data.build_dataloader(te_dset, num_workers=NUM_WORKERS, shuffle=False)

eval_mpnn    = build_mpnn(n_descriptors, eval_scaler)
epoch_tracker = BestEpochTracker()

eval_trainer = pl.Trainer(
    logger=False,
    enable_checkpointing=False,
    enable_progress_bar=True,
    accelerator="auto",
    devices=1,
    max_epochs=EVAL_MAX_EPOCHS,
    callbacks=[
        EarlyStopping(monitor="val_loss", patience=EVAL_PATIENCE, mode="min"),
        epoch_tracker,
    ],
)
eval_trainer.fit(eval_mpnn, tr_loader, va_loader)
print(f"\n  Early stop at epoch {epoch_tracker.best_epoch} "
      f"(best val_loss={epoch_tracker.best_val_loss:.4f})")

# Metrics on the held-out internal test set
eval_mpnn.eval()
raw_te = eval_trainer.predict(eval_mpnn, te_loader)
te_preds = torch.cat(raw_te).numpy().flatten()
te_true  = all_targets[te_idx]

print(f"\n  Internal test set metrics ({len(te_idx)} scaffold-held-out compounds):")
print_metrics("ChemProp+RDKit2D", te_true, te_preds)

df_internal = pd.DataFrame({
    "SMILES":    [all_smiles[i] for i in te_idx],
    "pEC50_true": te_true,
    "pEC50_pred": te_preds,
    "residual":   te_true - te_preds,
})
INTERNAL_PRED_PATH.parent.mkdir(parents=True, exist_ok=True)
df_internal.to_csv(INTERNAL_PRED_PATH, index=False)
print(f"  Internal predictions saved to {INTERNAL_PRED_PATH}")

# ══════════════════════════════════════════════════════════════════════════════════
# Step 6 — Phase 2: final model on ALL 4,140 compounds (no early stopping)
# ══════════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*62}")
print(f"Phase 2 — Final model on ALL {len(all_mols)} compounds  "
      f"({FINAL_EPOCHS} epochs, no early stopping)")
print(f"{'='*62}")

# Fit a fresh scaler on ALL training data
final_xd_scaler  = StandardScaler()
x_d_train_scaled = final_xd_scaler.fit_transform(x_d_all_raw)
x_d_test_scaled  = final_xd_scaler.transform(x_d_test_all)

all_dps   = make_datapoints(all_mols, all_targets, all_weights, x_d_train_scaled)
all_dset  = data.MoleculeDataset(all_dps, feat)
final_scaler  = all_dset.normalize_targets()
final_loader  = data.build_dataloader(all_dset, num_workers=NUM_WORKERS)

final_mpnn = build_mpnn(n_descriptors, final_scaler)

final_trainer = pl.Trainer(
    logger=False,
    enable_checkpointing=False,
    enable_progress_bar=True,
    accelerator="auto",
    devices=1,
    max_epochs=FINAL_EPOCHS,
)
final_trainer.fit(final_mpnn, final_loader)

# ══════════════════════════════════════════════════════════════════════════════════
# Step 7 — Save final model
# ══════════════════════════════════════════════════════════════════════════════════
torch.save(final_mpnn, MODEL_PATH)
print(f"\nModel saved to {MODEL_PATH}")

# ══════════════════════════════════════════════════════════════════════════════════
# Step 8 — Predict on competition test set and write submission
# ══════════════════════════════════════════════════════════════════════════════════
print(f"\nPredicting on {len(test_mols)} competition test molecules...")
final_mpnn.eval()

test_dps   = make_datapoints(test_mols,
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
SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
df_submission.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission saved to {SUBMISSION_PATH}")
print(f"\nFirst 5 predictions:")
print(df_submission.head().to_string(index=False))
print(f"\npEC50 summary:")
print(df_submission["pEC50"].describe().to_string())
