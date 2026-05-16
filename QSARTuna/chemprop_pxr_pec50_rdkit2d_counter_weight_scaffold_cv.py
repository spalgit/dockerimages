"""
ChemProp PXR pEC50 — RDKit 2D descriptors + counter-assay weighting
                     with 5-fold SCAFFOLD cross-validation.

Identical to chemprop_pxr_pec50_rdkit2d_counter_weight.py in all modelling
choices (architecture grid, weighting scheme, MAE criterion, slow LR) but
replaces StratifiedKFold with Bemis-Murcko scaffold K-fold splitting.

Why scaffold CV?
  Stratified split lets the same scaffold appear in both train and val,
  so val loss is optimistic and hyperparameter selection can favour configs
  that overfit scaffold-specific patterns.  Scaffold splitting keeps every
  scaffold whole within one fold, giving a realistic estimate of how well
  the model generalises to the novel chemical series present in the
  competition test set.

Two additional tweaks vs the original:
  • CV_MAX_EPOCHS 50  → 200  (slow LR of 2e-4 needs many steps to converge)
  • CV_PATIENCE    10  → 30   (avoids stopping before the plateau is reached)

Weighting scheme (unchanged):
  pEC50_counter_min → MAX_WEIGHT=2.0  (selective compounds, trust more)
  pEC50_counter_max → MIN_WEIGHT=0.5  (promiscuous, trust less)
  NaN               → NEUTRAL_WEIGHT=1.0

Output:
  ~/pxr_rdkit2d_cw_scaffold_cv_results.csv   — grid-search results
  ~/pxr_rdkit2d_cw_scaffold_final.pkl        — saved final model
  ~/pxr_rdkit2d_cw_scaffold_kept_descs.txt   — descriptor names kept
  ~/OpenAdmet/Submission_CW_Scaffold_CV.csv  — competition submission

Usage:
    conda activate chemprop
    python ~/dockerimages/QSARTuna/chemprop_pxr_pec50_rdkit2d_counter_weight_scaffold_cv.py
"""

import itertools
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
from sklearn.preprocessing import StandardScaler

from chemprop import data, featurizers, models, nn

# ── Paths ──────────────────────────────────────────────────────────────────────
TRAIN_PATH = Path(
    "/home/spal/dockerimages/QSARTuna/PXR_For_QSARTuna_Modelling/"
    "processed_Openadmet_REAL_PXR_train_AND_test_main_with_side_info_"
    "AND_counter_screen_weighted.csv"
)
TEST_PATH       = Path.home() / "dockerimages/QSARTuna/PXR/test.csv"
MODEL_PKL_PATH  = Path.home() / "pxr_rdkit2d_cw_scaffold_final.pkl"
CV_RESULTS_PATH = Path.home() / "pxr_rdkit2d_cw_scaffold_cv_results.csv"
KEPT_DESCS_PATH = Path.home() / "pxr_rdkit2d_cw_scaffold_kept_descs.txt"
SUBMISSION_PATH = Path.home() / "OpenAdmet/Submission_CW_Scaffold_CV.csv"

TRAIN_SMILES_COL = "SMILES"
TRAIN_TARGET_COL = "pEC50"
COUNTER_COL      = "pEC50_counter"
TEST_SMILES_COL  = "SMILES"
TEST_NAME_COL    = "Molecule Name"

# ── Weighting ──────────────────────────────────────────────────────────────────
MIN_WEIGHT     = 0.5
MAX_WEIGHT     = 2.0
NEUTRAL_WEIGHT = 1.0

# ── CV / training settings ─────────────────────────────────────────────────────
N_FOLDS       = 5
CV_MAX_EPOCHS = 200   # generous budget — slow LR (2e-4) needs many steps
CV_PATIENCE   = 30    # generous patience to reach true convergence
NUM_WORKERS   = 0

INIT_LR  = 1e-4
MAX_LR   = 2e-4
FINAL_LR = 1e-5

PARAM_GRID = {
    "ffn_hidden_dim": [300, 512],
    "ffn_n_layers":   [2, 3],
    "dropout":        [0.0, 0.2],
    "mp_depth":       [3, 4],
    "mp_hidden_dim":  [300, 1024],
}


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


# ── RDKit 2D descriptors ───────────────────────────────────────────────────────
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
    """Keep columns finite for all molecules and with nonzero variance (train only)."""
    finite = np.all(np.isfinite(arr), axis=0)
    varied = np.var(arr, axis=0) > 0
    return finite & varied


# ── Counter-screen weighting ───────────────────────────────────────────────────
def compute_weights(counter_values: np.ndarray) -> np.ndarray:
    weights     = np.full(len(counter_values), NEUTRAL_WEIGHT, dtype=float)
    has_counter = ~np.isnan(counter_values)
    vals        = counter_values[has_counter]
    c_min, c_max = vals.min(), vals.max()
    weights[has_counter] = MIN_WEIGHT + (MAX_WEIGHT - MIN_WEIGHT) * (
        (c_max - vals) / (c_max - c_min)
    )
    return weights


# ── Scaffold K-fold splitter ───────────────────────────────────────────────────
def scaffold_kfold_indices(mols: list, n_splits: int = 5):
    """
    Yield (train_indices, val_indices) for each of n_splits scaffold folds.

    All molecules sharing the same Bemis-Murcko scaffold are placed in the
    same fold.  Scaffold groups are assigned greedily to the smallest fold
    at each step to keep fold sizes as balanced as possible.
    """
    scaffold_to_idx = defaultdict(list)
    for i, mol in enumerate(mols):
        smi = Chem.MolToSmiles(mol)
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(
                smiles=smi, includeChirality=False
            )
        except Exception:
            scaffold = smi
        scaffold_to_idx[scaffold].append(i)

    # Sort largest group first for stable greedy assignment
    groups = sorted(scaffold_to_idx.values(), key=len, reverse=True)

    fold_buckets = [[] for _ in range(n_splits)]
    fold_sizes   = [0]  * n_splits
    for group in groups:
        smallest = int(np.argmin(fold_sizes))
        fold_buckets[smallest].extend(group)
        fold_sizes[smallest] += len(group)

    for val_fold in range(n_splits):
        val_idx   = np.array(fold_buckets[val_fold])
        train_idx = np.array([i for f in range(n_splits)
                               if f != val_fold
                               for i in fold_buckets[f]])
        yield train_idx, val_idx


# ── MPNN factory ──────────────────────────────────────────────────────────────
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


# ── Datapoint builder ──────────────────────────────────────────────────────────
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


# ── Single fold runner ─────────────────────────────────────────────────────────
def run_fold(train_mols, train_targets, train_weights, train_x_d,
             val_mols,   val_targets,   val_weights,   val_x_d,
             n_descriptors, params, max_epochs, patience):
    """Train one scaffold CV fold; return (best_val_mae, best_epoch)."""
    feat = featurizers.SimpleMoleculeMolGraphFeaturizer()

    train_dps = make_datapoints(train_mols, train_targets, train_weights, train_x_d)
    val_dps   = make_datapoints(val_mols,   val_targets,   val_weights,   val_x_d)

    train_dset = data.MoleculeDataset(train_dps, feat)
    val_dset   = data.MoleculeDataset(val_dps,   feat)

    target_scaler = train_dset.normalize_targets()
    val_dset.normalize_targets(target_scaler)

    train_loader = data.build_dataloader(train_dset, num_workers=NUM_WORKERS)
    val_loader   = data.build_dataloader(val_dset,   num_workers=NUM_WORKERS, shuffle=False)

    mpnn          = build_mpnn(n_descriptors, **params, target_scaler=target_scaler)
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
print(f"  {len(df_train)} rows")

counter_values = df_train[COUNTER_COL].values
train_weights  = compute_weights(counter_values)
n_with_counter = (~np.isnan(counter_values)).sum()
print(f"  Counter screen available : {n_with_counter}/{len(df_train)} "
      f"({100*n_with_counter/len(df_train):.1f}%)")
print(f"  pEC50_counter range      : "
      f"{np.nanmin(counter_values):.2f} – {np.nanmax(counter_values):.2f}  "
      f"(mean {np.nanmean(counter_values):.2f})")
print(f"  Weight range             : "
      f"{train_weights.min():.3f} – {train_weights.max():.3f}  "
      f"(neutral={NEUTRAL_WEIGHT})")

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
        print(f"  Skipped unparseable SMILES: {smi[:40]}")

train_targets       = np.array(train_targets)
train_weights_clean = np.array(train_weights_clean)
print(f"  Usable training molecules: {len(train_mols)}")

# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — Load competition test set (SMILES only, no labels)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nLoading competition test set:\n  {TEST_PATH}")
df_test    = pd.read_csv(TEST_PATH)
test_mols, test_smiles, test_names = [], [], []
for _, row in df_test.iterrows():
    mol = Chem.MolFromSmiles(row[TEST_SMILES_COL])
    if mol is not None:
        test_mols.append(mol)
        test_smiles.append(row[TEST_SMILES_COL])
        test_names.append(row[TEST_NAME_COL])
print(f"  Usable test molecules: {len(test_mols)}")

# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — RDKit 2D descriptors (column mask derived from training set only)
# ══════════════════════════════════════════════════════════════════════════════
print("\nComputing RDKit 2D descriptors for training set...")
x_d_train_all = compute_rdkit_descriptors(train_mols)
col_mask      = select_valid_columns(x_d_train_all)
kept_names    = [ALL_DESC_NAMES[i] for i, k in enumerate(col_mask) if k]
print(f"  Kept {col_mask.sum()} / {len(col_mask)} descriptors "
      f"(dropped {(~col_mask).sum()} NaN/inf/zero-var columns)")

x_d_train_raw = x_d_train_all[:, col_mask]
n_descriptors = x_d_train_raw.shape[1]

with open(KEPT_DESCS_PATH, "w") as fh:
    fh.write("\n".join(kept_names))
print(f"  Descriptor names saved to {KEPT_DESCS_PATH}")

print("Computing RDKit 2D descriptors for test set...")
x_d_test_all = compute_rdkit_descriptors(test_mols)
x_d_test_raw = x_d_test_all[:, col_mask]

# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — Build scaffold folds once; reuse for all hyperparameter combos
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nBuilding {N_FOLDS}-fold scaffold split...")
scaffold_folds = list(scaffold_kfold_indices(train_mols, n_splits=N_FOLDS))
for i, (tr_idx, va_idx) in enumerate(scaffold_folds):
    print(f"  Fold {i+1}: train={len(tr_idx)}  val={len(va_idx)}")

# ══════════════════════════════════════════════════════════════════════════════
# Step 5 — Scaffold CV grid search (skip if results already exist)
# ══════════════════════════════════════════════════════════════════════════════
if CV_RESULTS_PATH.exists():
    print(f"\nCV results found at {CV_RESULTS_PATH} — skipping grid search.")
    df_cv = pd.read_csv(CV_RESULTS_PATH).sort_values("mean_val_mae").reset_index(drop=True)
    print(f"\n{df_cv.to_string(index=False)}")
else:
    param_combos = [
        dict(zip(PARAM_GRID.keys(), combo))
        for combo in itertools.product(*PARAM_GRID.values())
    ]

    print(f"\n{'='*60}")
    print(f"5-fold scaffold CV — {len(param_combos)} combos  |  "
          f"n_descriptors={n_descriptors}  "
          f"max_epochs={CV_MAX_EPOCHS}  patience={CV_PATIENCE}")
    print(f"{'='*60}")

    cv_results = []

    for combo_idx, params in enumerate(param_combos):
        print(f"\n[{combo_idx+1}/{len(param_combos)}] {params}")
        fold_maes, fold_epochs = [], []

        for fold_num, (tr_idx, va_idx) in enumerate(scaffold_folds):
            # Fit x_d scaler on fold-train only; apply to fold-val
            xd_scaler = StandardScaler()
            x_d_tr    = xd_scaler.fit_transform(x_d_train_raw[tr_idx])
            x_d_va    = xd_scaler.transform(x_d_train_raw[va_idx])

            val_mae, best_epoch = run_fold(
                [train_mols[i] for i in tr_idx],
                train_targets[tr_idx],
                train_weights_clean[tr_idx],
                x_d_tr,
                [train_mols[i] for i in va_idx],
                train_targets[va_idx],
                train_weights_clean[va_idx],
                x_d_va,
                n_descriptors,
                params,
                max_epochs=CV_MAX_EPOCHS,
                patience=CV_PATIENCE,
            )
            fold_maes.append(val_mae)
            fold_epochs.append(best_epoch)
            print(f"  Fold {fold_num+1}: val_MAE={val_mae:.4f}  best_epoch={best_epoch}")

        mean_mae   = float(np.mean(fold_maes))
        std_mae    = float(np.std(fold_maes))
        mean_epoch = int(np.mean(fold_epochs))
        print(f"  → Mean val MAE: {mean_mae:.4f} ± {std_mae:.4f}  "
              f"Mean best epoch: {mean_epoch}")

        cv_results.append({**params,
                           "mean_val_mae":   mean_mae,
                           "std_val_mae":    std_mae,
                           "mean_best_epoch": mean_epoch})

    df_cv = pd.DataFrame(cv_results).sort_values("mean_val_mae").reset_index(drop=True)
    df_cv.to_csv(CV_RESULTS_PATH, index=False)
    print(f"\nCV results saved to {CV_RESULTS_PATH}")
    print(f"\n{df_cv.to_string(index=False)}")

# ── Select best hyperparameters ────────────────────────────────────────────────
best_row    = df_cv.iloc[0]
best_params = {
    "ffn_hidden_dim": int(best_row["ffn_hidden_dim"]),
    "ffn_n_layers":   int(best_row["ffn_n_layers"]),
    "dropout":        float(best_row["dropout"]),
    "mp_depth":       int(best_row["mp_depth"]),
    "mp_hidden_dim":  int(best_row["mp_hidden_dim"]),
}
final_epochs = max(int(best_row["mean_best_epoch"] * 1.1), 5)
print(f"\nBest hyperparameters : {best_params}")
print(f"Final model epochs   : {final_epochs}  (mean best epoch × 1.1)")

# ══════════════════════════════════════════════════════════════════════════════
# Step 6 — Final model: all training data, best hyperparameters
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"Training final model on all {len(train_mols)} compounds for {final_epochs} epochs")
print(f"{'='*60}")

final_xd_scaler  = StandardScaler()
x_d_train_scaled = final_xd_scaler.fit_transform(x_d_train_raw)
x_d_test_scaled  = final_xd_scaler.transform(x_d_test_raw)

feat = featurizers.SimpleMoleculeMolGraphFeaturizer()

all_train_dps  = make_datapoints(
    train_mols, train_targets, train_weights_clean, x_d_train_scaled
)
all_train_dset = data.MoleculeDataset(all_train_dps, feat)
final_scaler   = all_train_dset.normalize_targets()
final_loader   = data.build_dataloader(all_train_dset, num_workers=NUM_WORKERS)

final_mpnn = build_mpnn(n_descriptors, **best_params, target_scaler=final_scaler)
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
torch.save(final_mpnn, MODEL_PKL_PATH)
print(f"\nModel saved to {MODEL_PKL_PATH}")

# ══════════════════════════════════════════════════════════════════════════════
# Step 8 — Predict on competition test set and write submission
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nPredicting on {len(test_mols)} competition test molecules...")
final_mpnn.eval()

test_weights_dummy = np.ones(len(test_mols))
test_dps    = make_datapoints(test_mols,
                              np.zeros(len(test_mols)),   # dummy targets
                              test_weights_dummy,
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
df_submission.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission saved to {SUBMISSION_PATH}")
print(f"\nFirst 5 predictions:")
print(df_submission.head().to_string(index=False))
