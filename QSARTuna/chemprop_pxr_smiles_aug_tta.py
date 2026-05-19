"""
ChemProp PXR pEC50 — SMILES enumeration augmentation + test-time averaging (TTA).

Extends chemprop_pxr_pec50_rdkit2d_counter_weight_ensemble.py with two changes:

  1. Training SMILES augmentation: each training molecule contributes its
     canonical SMILES plus N_TRAIN_AUG randomly-enumerated SMILES, all sharing
     the same label and counter-screen weight. Dataset grows ~(N_TRAIN_AUG+1)×,
     making the model robust to arbitrary SMILES representations.

  2. Test-time augmentation (TTA): for each test molecule N_TTA random SMILES
     are predicted and the results are averaged across all (n_seeds × N_TTA)
     combinations. This reduces variance at zero additional training cost.

Architecture / weighting: identical to the ensemble baseline.
CV hyperparameters: re-used from ~/pxr_rdkit2d_cw_ensemble_cv_results.csv if
present (saves ~4 h of grid search). Delete that file to force a fresh CV run
with the augmented training data.

Note on internal test metrics: TEST_PATH contains ChemProp-only model predictions
as its pEC50 column (not ground truth). Reported metrics compare this model
against that baseline; the true leaderboard score comes from submission.

Usage:
    conda activate chemprop
    python ~/dockerimages/QSARTuna/chemprop_pxr_smiles_aug_tta.py
"""

import itertools
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightning import pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from rdkit import Chem
from rdkit.Chem import Descriptors
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from chemprop import data, featurizers, models, nn

# ── Paths ──────────────────────────────────────────────────────────────────────
TRAIN_PATH = Path(
    "/home/spal/dockerimages/QSARTuna/PXR_For_QSARTuna_Modelling/"
    "processed_Openadmet_REAL_PXR_train_AND_test_main_with_side_info_"
    "AND_counter_screen_weighted.csv"
)
TEST_PATH = (
    Path.home()
    / "dockerimages/QSARTuna/PXR/Prediction_OpenAdmet_ChemProp_Only_OpenADMET_Data.csv"
)
# Re-use existing CV results to skip the ~4 h grid search.
CV_RESULTS_PATH   = Path.home() / "pxr_rdkit2d_cw_ensemble_cv_results.csv"
ENSEMBLE_DIR      = Path.home() / "pxr_smiles_aug_tta_models"
OUTPUT_PREDS      = Path.home() / "pxr_smiles_aug_tta_predictions.csv"
OUTPUT_SUBMISSION = Path.home() / "pxr_smiles_aug_tta_submission.csv"
KEPT_DESCS_PATH   = Path.home() / "pxr_rdkit2d_cw_ensemble_kept_descriptors.txt"

# ── Column names ───────────────────────────────────────────────────────────────
TRAIN_SMILES_COL = "SMILES"
TRAIN_TARGET_COL = "pEC50"
COUNTER_COL      = "pEC50_counter"
TEST_SMILES_COL  = "SMILES"
TEST_TARGET_COL  = "pEC50"
TEST_NAME_COL    = "Molecule Name"

# ── Counter-screen weighting (unchanged from baseline) ─────────────────────────
MIN_WEIGHT     = 0.5
MAX_WEIGHT     = 2.0
NEUTRAL_WEIGHT = 1.0

# ── CV settings ────────────────────────────────────────────────────────────────
N_FOLDS       = 5
N_STRATA_BINS = N_FOLDS
CV_MAX_EPOCHS = 50
CV_PATIENCE   = 10
NUM_WORKERS   = 0

# ── Learning rates ─────────────────────────────────────────────────────────────
INIT_LR  = 1e-4
MAX_LR   = 2e-4
FINAL_LR = 1e-5

# ── Ensemble settings ──────────────────────────────────────────────────────────
ENSEMBLE_SEEDS = [42, 123, 456, 789, 1337, 2024, 31415, 99999]

# ── SMILES augmentation settings ──────────────────────────────────────────────
# N_TRAIN_AUG: extra random SMILES per training mol (canonical always included).
# Total training size becomes n_train × (1 + N_TRAIN_AUG).
# Each extra SMILES adds ~1 epoch-equivalent of compute per seed.
# 4 = 5× dataset, good balance of augmentation vs runtime.
N_TRAIN_AUG = 4

# N_TTA: number of SMILES representations averaged per test molecule at inference.
# Includes the canonical SMILES (index 0). Diminishing returns above 20.
N_TTA = 20

# ── Hard-coded best hyperparameters (optional shortcut) ───────────────────────
# Set these to skip CV entirely. Copy the top row of pxr_rdkit2d_cw_ensemble_cv_results.csv
# from your VM. Leave as None to fall back to loading the CSV or running CV.
#
# Example (fill in your actual values):
#   BEST_PARAMS_OVERRIDE = {
#       "ffn_hidden_dim": 1024,
#       "ffn_n_layers":   3,
#       "dropout":        0.0,
#       "mp_depth":       4,
#       "mp_hidden_dim":  1024,
#   }
#   FINAL_EPOCHS_OVERRIDE = 25   # mean_best_epoch × 1.1, rounded up
BEST_PARAMS_OVERRIDE  = None   # ← replace None with dict to activate
FINAL_EPOCHS_OVERRIDE = None   # ← replace None with int to activate

# ── Hyperparameter grid (used only when CV re-runs) ───────────────────────────
PARAM_GRID = {
    "ffn_hidden_dim": [300, 512, 1024],
    "ffn_n_layers":   [2, 3],
    "dropout":        [0.0, 0.2],
    "mp_depth":       [3, 4],
    "mp_hidden_dim":  [300, 1024],
}


# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


# ── SMILES augmentation ────────────────────────────────────────────────────────
def _random_smiles_mol(mol: Chem.Mol):
    """Return (random_smi, rand_mol) or (None, None) if RDKit fails."""
    try:
        smi = Chem.MolToSmiles(mol, doRandom=True)
        m   = Chem.MolFromSmiles(smi)
        if m is not None:
            return smi, m
    except Exception:
        pass
    return None, None


def augment_training_data(
    mols: list,
    targets: np.ndarray,
    weights: np.ndarray,
    x_d: np.ndarray,
    n_aug: int,
):
    """
    Expand training data by adding n_aug randomly-enumerated SMILES per molecule.
    x_d (RDKit 2D descriptors) is molecule-level — augmented copies share the
    same descriptor vector as their source.
    Returns augmented (mols, targets, weights, x_d).
    """
    aug_mols, aug_tgt, aug_wt, aug_xd = [], [], [], []
    failed = 0
    for mol, t, w, xd in zip(mols, targets, weights, x_d):
        aug_mols.append(mol)
        aug_tgt.append(t)
        aug_wt.append(w)
        aug_xd.append(xd)

        added, attempts = 0, 0
        while added < n_aug and attempts < n_aug * 5:
            _, rand_mol = _random_smiles_mol(mol)
            if rand_mol is not None:
                aug_mols.append(rand_mol)
                aug_tgt.append(t)
                aug_wt.append(w)
                aug_xd.append(xd)
                added += 1
            attempts += 1
        failed += n_aug - added

    if failed:
        print(f"  Warning: could not generate {failed} augmented SMILES "
              f"(SMILES too simple to enumerate)")
    print(f"  Augmented training set: {len(mols)} → {len(aug_mols)} molecules "
          f"({len(aug_mols)/len(mols):.1f}×)")
    return aug_mols, np.array(aug_tgt), np.array(aug_wt), np.array(aug_xd)


def generate_tta_pool(mols: list, x_d: np.ndarray, n_tta: int):
    """
    For each test molecule generate n_tta SMILES (canonical + randoms).
    Returns:
        tta_mols    list of RDKit mol objects (length ≈ n_test × n_tta)
        tta_x_d     matching descriptor rows
        tta_indices np.ndarray mapping each TTA entry → original mol index
    """
    tta_mols, tta_xd, tta_idx = [], [], []
    for i, (mol, xd) in enumerate(zip(mols, x_d)):
        tta_mols.append(mol)           # canonical
        tta_xd.append(xd)
        tta_idx.append(i)

        added, attempts = 0, 0
        while added < n_tta - 1 and attempts < (n_tta - 1) * 5:
            _, rand_mol = _random_smiles_mol(mol)
            if rand_mol is not None:
                tta_mols.append(rand_mol)
                tta_xd.append(xd)
                tta_idx.append(i)
                added += 1
            attempts += 1

    return tta_mols, np.array(tta_xd), np.array(tta_idx)


# ── MPNN builder ───────────────────────────────────────────────────────────────
def build_mpnn(
    n_descriptors: int,
    ffn_hidden_dim: int,
    ffn_n_layers: int,
    dropout: float,
    mp_depth: int,
    mp_hidden_dim: int,
    target_scaler,
) -> models.MPNN:
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


# ── Metrics ────────────────────────────────────────────────────────────────────
def report_metrics(actual: np.ndarray, predicted: np.ndarray, label: str = "") -> dict:
    mae  = float(np.mean(np.abs(actual - predicted)))
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2   = float(1.0 - ss_res / ss_tot)
    range_actual = actual.max() - actual.min()
    rae  = float(mae / (range_actual / 2.0)) if range_actual > 0 else float("nan")
    rho, _ = stats.spearmanr(actual, predicted)
    tau, _ = stats.kendalltau(actual, predicted)
    tag = f"  [{label}]" if label else ""
    print(
        f"{tag}  MAE={mae:.4f}  RMSE={rmse:.4f}  RAE={rae:.4f}  "
        f"R²={r2:.4f}  Spearman={rho:.4f}  Kendall={tau:.4f}"
    )
    return dict(mae=mae, rmse=rmse, rae=rae, r2=r2, spearman=rho, kendall=tau)


# ── One CV fold (no augmentation — keeps CV runtime same as baseline) ─────────
def run_fold(
    fold_idx,
    train_mols, train_targets, train_weights, train_x_d,
    val_mols,   val_targets,   val_weights,   val_x_d,
    n_descriptors, params, max_epochs, patience,
):
    feat = featurizers.SimpleMoleculeMolGraphFeaturizer()

    # No augmentation during CV: hyperparameter selection doesn't require it
    # and augmenting here would make CV ~5× slower without changing which
    # architecture wins.
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
      f"{train_weights.min():.3f} – {train_weights.max():.3f}  "
      f"(neutral={NEUTRAL_WEIGHT})")
print(f"\nSMILES augmentation: N_TRAIN_AUG={N_TRAIN_AUG} → "
      f"{1+N_TRAIN_AUG}× training set size")
print(f"TTA at inference   : N_TTA={N_TTA} SMILES per test molecule")

print("\nParsing training SMILES...")
train_mols, train_smiles, train_targets, train_weights_clean = [], [], [], []
for smi, y, w in zip(
    df_train[TRAIN_SMILES_COL].values,
    df_train[TRAIN_TARGET_COL].values,
    train_weights,
):
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
# Step 2 — Load test data
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
print("  (Note: test pEC50 = ChemProp-only predictions, not ground truth)")

# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — RDKit 2D descriptors (column selection from training set only)
# ══════════════════════════════════════════════════════════════════════════════
print("\nComputing RDKit 2D descriptors for training set...")
x_d_train_all = compute_rdkit_descriptors(train_mols)
col_mask      = select_valid_columns(x_d_train_all)
kept_names    = [ALL_DESC_NAMES[i] for i, k in enumerate(col_mask) if k]
print(f"  Kept {col_mask.sum()} / {len(col_mask)} descriptors")

x_d_train_raw = x_d_train_all[:, col_mask]
n_descriptors = x_d_train_raw.shape[1]

with open(KEPT_DESCS_PATH, "w") as fh:
    fh.write("\n".join(kept_names))
print(f"  Descriptor names saved to {KEPT_DESCS_PATH}")

print("\nComputing RDKit 2D descriptors for test set...")
x_d_test_all = compute_rdkit_descriptors(test_mols)
x_d_test_raw = x_d_test_all[:, col_mask]

# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — Stratification labels for CV
# ══════════════════════════════════════════════════════════════════════════════
strata = pd.qcut(
    train_targets, q=N_STRATA_BINS, labels=False, duplicates="drop"
).astype(int)

# ══════════════════════════════════════════════════════════════════════════════
# Step 5 — CV grid search (or load existing results)
# ══════════════════════════════════════════════════════════════════════════════
if BEST_PARAMS_OVERRIDE is not None:
    print("\nBEST_PARAMS_OVERRIDE is set — skipping CV entirely.")
    df_cv = None
elif CV_RESULTS_PATH.exists():
    print(f"\nCV results found at {CV_RESULTS_PATH} — skipping grid search.")
    df_cv = pd.read_csv(CV_RESULTS_PATH).sort_values("mean_val_loss").reset_index(drop=True)
    print(f"\n{df_cv.to_string(index=False)}")
else:
    param_combos = [
        dict(zip(PARAM_GRID.keys(), combo))
        for combo in itertools.product(*PARAM_GRID.values())
    ]

    print(f"\n{'='*70}")
    print(f"5-fold CV with SMILES augmentation — {len(param_combos)} combos  |  "
          f"n_descriptors={n_descriptors}")
    print(f"{'='*70}")

    skf     = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    indices = np.arange(len(train_mols))
    cv_results = []

    for combo_idx, params in enumerate(param_combos):
        print(f"\n[{combo_idx + 1}/{len(param_combos)}] {params}")
        fold_losses, fold_epochs = [], []

        for fold_num, (tr_idx, va_idx) in enumerate(skf.split(indices, strata)):
            set_seed(42)
            xd_scaler = StandardScaler()
            x_d_tr = xd_scaler.fit_transform(x_d_train_raw[tr_idx])
            x_d_va = xd_scaler.transform(x_d_train_raw[va_idx])

            val_loss, best_epoch = run_fold(
                fold_num,
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
            fold_losses.append(val_loss)
            fold_epochs.append(best_epoch)
            print(f"  Fold {fold_num + 1}: val_loss={val_loss:.4f}  best_epoch={best_epoch}")

        mean_loss  = float(np.mean(fold_losses))
        std_loss   = float(np.std(fold_losses))
        mean_epoch = int(np.mean(fold_epochs))
        print(f"  → Mean val loss: {mean_loss:.4f} ± {std_loss:.4f}  "
              f"Mean best epoch: {mean_epoch}")

        cv_results.append({
            **params,
            "mean_val_loss":   mean_loss,
            "std_val_loss":    std_loss,
            "mean_best_epoch": mean_epoch,
        })

    df_cv = (
        pd.DataFrame(cv_results)
        .sort_values("mean_val_loss")
        .reset_index(drop=True)
    )
    df_cv.to_csv(CV_RESULTS_PATH, index=False)
    print(f"\nCV results saved to {CV_RESULTS_PATH}")
    print(f"\n{df_cv.to_string(index=False)}")

# ── Best hyperparameters ───────────────────────────────────────────────────────
if BEST_PARAMS_OVERRIDE is not None:
    best_params  = BEST_PARAMS_OVERRIDE
    final_epochs = FINAL_EPOCHS_OVERRIDE if FINAL_EPOCHS_OVERRIDE is not None else 20
    print("\nUsing hard-coded best hyperparameters (CV skipped).")
else:
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
print(f"Final model epochs   : {final_epochs}")
print(f"Ensemble size        : {len(ENSEMBLE_SEEDS)} seeds")

# ══════════════════════════════════════════════════════════════════════════════
# Step 6 — Fit x_d scaler on all training data; scale test descriptors
# ══════════════════════════════════════════════════════════════════════════════
final_xd_scaler  = StandardScaler()
x_d_train_scaled = final_xd_scaler.fit_transform(x_d_train_raw)
x_d_test_scaled  = final_xd_scaler.transform(x_d_test_raw)

# ══════════════════════════════════════════════════════════════════════════════
# Step 7 — Pre-generate TTA pool for test set
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nGenerating TTA pool: {N_TTA} SMILES × {len(test_mols)} test mols ...")
tta_mols, tta_x_d, tta_indices = generate_tta_pool(test_mols, x_d_test_scaled, N_TTA)
print(f"  TTA pool size: {len(tta_mols)} entries "
      f"(avg {len(tta_mols)/len(test_mols):.1f} per mol)")

feat = featurizers.SimpleMoleculeMolGraphFeaturizer()

# Dummy targets and weights for the TTA dataloader
tta_targets_dummy = np.zeros(len(tta_mols))
tta_weights_dummy = np.ones(len(tta_mols))
tta_dps   = make_datapoints(tta_mols, tta_targets_dummy, tta_weights_dummy, tta_x_d)
tta_dset  = data.MoleculeDataset(tta_dps, feat)
tta_loader = data.build_dataloader(tta_dset, num_workers=NUM_WORKERS, shuffle=False)

# ══════════════════════════════════════════════════════════════════════════════
# Step 8 — Train ensemble members on augmented training data
# ══════════════════════════════════════════════════════════════════════════════
ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)
all_tta_preds   = []   # shape: (n_seeds, n_tta_pool)
per_model_metrics = []

print(f"\nAugmenting full training set for final ensemble training ...")
aug_train_mols, aug_train_targets, aug_train_weights, aug_x_d_train = augment_training_data(
    train_mols, train_targets, train_weights_clean, x_d_train_scaled, N_TRAIN_AUG
)

for i, seed in enumerate(ENSEMBLE_SEEDS):
    print(f"\n{'='*70}")
    print(f"Ensemble member {i + 1}/{len(ENSEMBLE_SEEDS)}  |  seed={seed}")
    print(f"{'='*70}")

    set_seed(seed)

    all_train_dps  = make_datapoints(
        aug_train_mols, aug_train_targets, aug_train_weights, aug_x_d_train
    )
    all_train_dset = data.MoleculeDataset(all_train_dps, feat)
    target_scaler  = all_train_dset.normalize_targets()
    train_loader   = data.build_dataloader(all_train_dset, num_workers=NUM_WORKERS)

    mpnn = build_mpnn(n_descriptors, **best_params, target_scaler=target_scaler)

    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        accelerator="auto",
        devices=1,
        max_epochs=final_epochs,
    )
    trainer.fit(mpnn, train_loader)

    model_path = ENSEMBLE_DIR / f"model_seed{seed}.pt"
    torch.save(mpnn, model_path)
    print(f"  Saved → {model_path}")

    # Predict on the full TTA pool in one forward pass
    mpnn.eval()
    predict_trainer = pl.Trainer(
        logger=False,
        enable_progress_bar=False,
        accelerator="auto",
        devices=1,
    )
    raw_preds = predict_trainer.predict(mpnn, tta_loader)
    preds_flat = torch.cat(raw_preds).numpy().flatten()  # length = n_tta_pool
    all_tta_preds.append(preds_flat)

    # Per-member metric: average TTA for this seed only, then compare vs test baseline
    seed_per_mol = np.array([
        preds_flat[tta_indices == j].mean() for j in range(len(test_mols))
    ])
    m = report_metrics(test_targets, seed_per_mol, label=f"seed={seed} (TTA avg)")
    m["seed"] = seed
    per_model_metrics.append(m)

# ══════════════════════════════════════════════════════════════════════════════
# Step 9 — Average all (seed × TTA) predictions per test molecule
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Final ensemble + TTA results")
print(f"{'='*70}")

all_tta_array = np.stack(all_tta_preds, axis=0)  # (n_seeds, n_tta_pool)

# Average across seeds first, then across TTA entries per molecule
mean_across_seeds = all_tta_array.mean(axis=0)   # (n_tta_pool,)
final_preds = np.array([
    mean_across_seeds[tta_indices == j].mean() for j in range(len(test_mols))
])

ensemble_metrics = report_metrics(test_targets, final_preds, label="ENSEMBLE + TTA")

# Also report canonical-only ensemble (no TTA) for comparison
canonical_mask = np.array([
    (tta_indices == j).nonzero()[0][0] for j in range(len(test_mols))
])
canonical_preds = all_tta_array[:, canonical_mask].mean(axis=0)
report_metrics(test_targets, canonical_preds, label="ENSEMBLE canonical (no TTA)")

# ── Save detailed predictions ──────────────────────────────────────────────────
seed_cols = {}
for seed, preds_flat in zip(ENSEMBLE_SEEDS, all_tta_preds):
    per_mol = np.array([
        preds_flat[tta_indices == j].mean() for j in range(len(test_mols))
    ])
    seed_cols[f"pEC50_seed{seed}_tta"] = per_mol

df_out = pd.DataFrame({
    "Molecule Name":       test_names,
    "SMILES":              test_smiles,
    "pEC50_chemprop_only": test_targets,   # baseline (not ground truth)
    "pEC50_ensemble_tta":  final_preds,
    "pEC50_canonical_ens": canonical_preds,
    "residual_vs_baseline": test_targets - final_preds,
    **seed_cols,
})
df_out.to_csv(OUTPUT_PREDS, index=False)
print(f"\nDetailed predictions saved to {OUTPUT_PREDS}")

# ── Submission-format CSV ──────────────────────────────────────────────────────
df_sub = pd.DataFrame({
    "Molecule Name": test_names,
    "SMILES":        test_smiles,
    "pEC50":         final_preds,
})
df_sub.to_csv(OUTPUT_SUBMISSION, index=False)
print(f"Submission CSV saved to   {OUTPUT_SUBMISSION}")

# ── Per-member summary ─────────────────────────────────────────────────────────
df_members = pd.DataFrame(per_model_metrics)
print("\nPer-member test-set metrics (vs ChemProp-only baseline):")
print(df_members.to_string(index=False))
print(f"\nMean individual MAE  : {df_members['mae'].mean():.4f} ± {df_members['mae'].std():.4f}")
print(f"Ensemble + TTA MAE   : {ensemble_metrics['mae']:.4f}")

m = ensemble_metrics
print(
    f"\n[Leaderboard format]  "
    f"MAE={m['mae']:.4f}  RAE={m['rae']:.4f}  "
    f"R²={m['r2']:.4f}  Spearman={m['spearman']:.4f}  Kendall={m['kendall']:.4f}"
)
print("\nSubmit ~/pxr_smiles_aug_tta_submission.csv to the leaderboard.")
