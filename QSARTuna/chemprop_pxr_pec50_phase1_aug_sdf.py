"""
ChemProp PXR pEC50 — Phase-1-augmented training, SDF inputs.

Training data:
  - train_ci_filtered.sdf          (4132 compounds, MOE-prepped)
  - test_phase1.csv + test SDF     (253 Phase-1 labelled compounds)
    CI filter applied: pEC50 <= 3.5 AND CI_upper > 4.0 are removed.
    (In practice 0 compounds are removed from the 253.)

Weighting: sample_weight × counter_screen.
  1. sample_weight (1/std_error): derived fresh from std_error for both
     datasets, then jointly clipped at p99 and normalised to [0.5, 2.0].
  2. pEC50_counter: inversely mapped to [0.5, 2.0] for train_ci_filtered
     compounds; phase-1 compounds have no counter data → neutral 1.0.

Test data:
  - 260 compounds from test_OpenADMET_Data_prepped.sdf NOT in Phase 1.

Usage:
    conda activate chemprop
    python ~/dockerimages/QSARTuna/chemprop_pxr_pec50_phase1_aug_sdf.py
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
from scipy import stats
from sklearn.model_selection import StratifiedKFold

from chemprop import data, featurizers, models, nn

# ── Paths ──────────────────────────────────────────────────────────────────────
TRAIN_SDF     = Path("/home/spal/dockerimages/QSARTuna/PXR/train_ci_filtered_correct_Prot_state.sdf")
PHASE1_CSV    = Path("/home/spal/dockerimages/QSARTuna/PXR/test_phase1.csv")
TEST_SDF      = Path("/home/spal/dockerimages/QSARTuna/PXR/test_OpenADMET_Data_prepped.sdf")

CV_RESULTS_PATH = Path.home() / "pxr_chemprop_phase1_aug_cv_results.csv"
ENSEMBLE_DIR    = Path.home() / "pxr_ensemble_models_phase1_aug"
OUTPUT_PREDS    = Path.home() / "pxr_chemprop_phase1_aug_test_predictions.csv"

# ── SD / CSV column names ──────────────────────────────────────────────────────
TAG_TARGET   = "pEC50"
TAG_SE       = "std_error"
TAG_COUNTER  = "pEC50_counter"
TAG_NAME     = "Name"

COL_NAME     = "Molecule Name"
COL_SMILES   = "SMILES"
COL_TARGET   = "pEC50"
COL_SE       = "pEC50_std.error (-log10(molarity))"
COL_CI_UPPER = "pEC50_ci.upper (-log10(molarity))"

# ── CI filter thresholds (same as train_ci_filtered.sdf) ──────────────────────
CI_PECS50_CUTOFF = 3.5
CI_UPPER_CUTOFF  = 4.0

# ── Weight normalisation ───────────────────────────────────────────────────────
WEIGHT_MIN     = 0.5
WEIGHT_MAX     = 2.0
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

# ── Ensemble seeds ─────────────────────────────────────────────────────────────
ENSEMBLE_SEEDS = [42, 123, 456, 789, 1337, 2024, 31415, 99999]

# ── Hyperparameter grid ────────────────────────────────────────────────────────
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


def normalize_weights(raw_weights: np.ndarray) -> np.ndarray:
    """Clip at p99 then min-max normalise to [WEIGHT_MIN, WEIGHT_MAX]."""
    w_min  = raw_weights.min()
    w_clip = float(np.percentile(raw_weights, 99))
    clipped = np.clip(raw_weights, w_min, w_clip)
    if w_clip > w_min:
        return WEIGHT_MIN + (WEIGHT_MAX - WEIGHT_MIN) * (clipped - w_min) / (w_clip - w_min)
    return np.full(len(raw_weights), (WEIGHT_MIN + WEIGHT_MAX) / 2.0, dtype=float)


def counter_weights(counter_values: np.ndarray) -> np.ndarray:
    """Inversely map pEC50_counter to [WEIGHT_MIN, WEIGHT_MAX]; NaN → NEUTRAL."""
    weights     = np.full(len(counter_values), NEUTRAL_WEIGHT, dtype=float)
    has_counter = ~np.isnan(counter_values)
    if has_counter.sum() > 1:
        vals         = counter_values[has_counter]
        c_min, c_max = vals.min(), vals.max()
        if c_max > c_min:
            weights[has_counter] = WEIGHT_MIN + (WEIGHT_MAX - WEIGHT_MIN) * (
                (c_max - vals) / (c_max - c_min)
            )
    return weights


def load_sdf_main_train(sdf_path: Path):
    """Load the 4132-compound training SDF; read std_error directly."""
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=True)
    mols, names, targets, std_errors, counters = [], [], [], [], []
    skipped = 0

    for mol in suppl:
        if mol is None:
            skipped += 1
            continue
        name    = mol.GetProp(TAG_NAME) if mol.HasProp(TAG_NAME) else mol.GetProp("_Name")
        target  = float(mol.GetProp(TAG_TARGET))
        se      = float(mol.GetProp(TAG_SE))     if mol.HasProp(TAG_SE)      else 1.0
        counter = float(mol.GetProp(TAG_COUNTER)) if mol.HasProp(TAG_COUNTER) else np.nan

        mols.append(mol)
        names.append(name)
        targets.append(target)
        std_errors.append(se)
        counters.append(counter)

    if skipped:
        print(f"  {skipped} molecules skipped (unreadable)")
    print(f"  Loaded {len(mols)} compounds from main training SDF")
    return (
        mols,
        names,
        np.array(targets,    dtype=float),
        np.array(std_errors, dtype=float),
        np.array(counters,   dtype=float),
    )


def load_phase1_additions(phase1_csv: Path, test_sdf: Path):
    """
    Load the 253 Phase-1 labelled compounds:
      - mol objects from test_sdf (MOE-prepped 3D structures)
      - pEC50 and std_error from phase1_csv (actual measurements)
    Apply CI filter: remove pEC50 <= CI_PECS50_CUTOFF AND CI_upper > CI_UPPER_CUTOFF.
    Returns mol objects, names, targets, std_errors (counter = NaN for all).
    """
    df = pd.read_csv(phase1_csv)

    # CI filter
    ci_mask = (df[COL_TARGET] <= CI_PECS50_CUTOFF) & (df[COL_CI_UPPER] > CI_UPPER_CUTOFF)
    n_removed = ci_mask.sum()
    if n_removed:
        print(f"  CI filter removed {n_removed} ambiguous inactive(s) from Phase-1 set")
    df = df[~ci_mask].reset_index(drop=True)

    # Build name → row lookup
    phase1_lookup = df.set_index(COL_NAME)

    # Index mol objects from test SDF by name
    suppl = Chem.SDMolSupplier(str(test_sdf), removeHs=True)
    mol_by_name = {}
    for mol in suppl:
        if mol is None:
            continue
        n = mol.GetProp("_Name") if not mol.HasProp(TAG_NAME) else mol.GetProp(TAG_NAME)
        mol_by_name[n] = mol

    mols, names, targets, std_errors = [], [], [], []
    missing = 0
    for name, row in phase1_lookup.iterrows():
        mol = mol_by_name.get(name)
        if mol is None:
            missing += 1
            continue
        mols.append(mol)
        names.append(name)
        targets.append(float(row[COL_TARGET]))
        se = float(row[COL_SE]) if not pd.isna(row[COL_SE]) else 1.0
        std_errors.append(max(se, 1e-6))  # guard against zero

    if missing:
        print(f"  {missing} Phase-1 compound(s) not found in test SDF (skipped)")
    print(f"  Loaded {len(mols)} Phase-1 compounds (after CI filter: −{n_removed})")
    counters = np.full(len(mols), np.nan, dtype=float)
    return (
        mols,
        names,
        np.array(targets,    dtype=float),
        np.array(std_errors, dtype=float),
        counters,
    )


def load_sdf_test(test_sdf: Path, phase1_names: set):
    """
    Load the 260 compounds NOT in Phase 1 from the 513-compound test SDF.
    Uses the pEC50 tag already present in the SDF.
    """
    suppl = Chem.SDMolSupplier(str(test_sdf), removeHs=True)
    mols, names, targets = [], [], []
    skipped = 0

    for mol in suppl:
        if mol is None:
            skipped += 1
            continue
        name = mol.GetProp("_Name") if not mol.HasProp(TAG_NAME) else mol.GetProp(TAG_NAME)
        if name in phase1_names:
            continue  # these are now in training
        if not mol.HasProp(TAG_TARGET):
            skipped += 1
            continue
        mols.append(mol)
        names.append(name)
        targets.append(float(mol.GetProp(TAG_TARGET)))

    if skipped:
        print(f"  {skipped} molecules skipped")
    print(f"  Loaded {len(mols)} test compounds (513 − 253 phase-1 = {len(mols)})")
    return mols, names, np.array(targets, dtype=float)


def build_mpnn(
    ffn_hidden_dim, ffn_n_layers, dropout, mp_depth, mp_hidden_dim, target_scaler
) -> models.MPNN:
    feat = featurizers.SimpleMoleculeMolGraphFeaturizer()
    mp   = nn.BondMessagePassing(
               d_v=feat.atom_fdim, d_e=feat.bond_fdim,
               depth=mp_depth, d_h=mp_hidden_dim,
           )
    agg  = nn.MeanAggregation()
    ffn  = nn.RegressionFFN(
        input_dim=mp.output_dim,
        hidden_dim=ffn_hidden_dim,
        n_layers=ffn_n_layers,
        dropout=dropout,
        criterion=nn.metrics.MAE(),
        output_transform=nn.UnscaleTransform.from_standard_scaler(target_scaler),
    )
    return models.MPNN(
        mp, agg, ffn,
        batch_norm=True,
        metrics=[nn.metrics.RMSE(), nn.metrics.MAE()],
        init_lr=INIT_LR, max_lr=MAX_LR, final_lr=FINAL_LR,
    )


def make_datapoints(mols, targets, weights):
    return [
        data.MoleculeDatapoint(mol=mol, y=np.array([t], dtype=float), weight=float(w))
        for mol, t, w in zip(mols, targets, weights)
    ]


def report_metrics(actual: np.ndarray, predicted: np.ndarray, label: str = "") -> dict:
    mae  = float(np.mean(np.abs(actual - predicted)))
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2   = float(1.0 - ss_res / ss_tot)
    rng  = actual.max() - actual.min()
    rae  = float(mae / (rng / 2.0)) if rng > 0 else float("nan")
    rho, _ = stats.spearmanr(actual, predicted)
    tau, _ = stats.kendalltau(actual, predicted)
    tag = f"  [{label}]" if label else ""
    print(
        f"{tag}  MAE={mae:.4f}  RMSE={rmse:.4f}  RAE={rae:.4f}  "
        f"R²={r2:.4f}  Spearman={rho:.4f}  Kendall={tau:.4f}"
    )
    return dict(mae=mae, rmse=rmse, rae=rae, r2=r2, spearman=rho, kendall=tau)


def run_fold(
    fold_idx,
    train_mols, train_targets, train_weights,
    val_mols,   val_targets,   val_weights,
    params, max_epochs, patience,
):
    feat      = featurizers.SimpleMoleculeMolGraphFeaturizer()
    train_dps = make_datapoints(train_mols, train_targets, train_weights)
    val_dps   = make_datapoints(val_mols,   val_targets,   val_weights)

    train_dset = data.MoleculeDataset(train_dps, feat)
    val_dset   = data.MoleculeDataset(val_dps,   feat)

    target_scaler = train_dset.normalize_targets()
    val_dset.normalize_targets(target_scaler)

    train_loader = data.build_dataloader(train_dset, num_workers=NUM_WORKERS)
    val_loader   = data.build_dataloader(val_dset,   num_workers=NUM_WORKERS, shuffle=False)

    mpnn          = build_mpnn(**params, target_scaler=target_scaler)
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
# Step 1 — Load main training SDF (4132 compounds)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nLoading main training SDF:\n  {TRAIN_SDF}")
main_mols, main_names, main_targets, main_se, main_counters = \
    load_sdf_main_train(TRAIN_SDF)

# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — Load 253 Phase-1 compounds (from CSV + test SDF mol objects)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nLoading Phase-1 additions:\n  {PHASE1_CSV}")
p1_mols, p1_names, p1_targets, p1_se, p1_counters = \
    load_phase1_additions(PHASE1_CSV, TEST_SDF)

phase1_name_set = set(p1_names)

# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — Combine training data, compute joint weights
# ══════════════════════════════════════════════════════════════════════════════
train_mols    = main_mols    + p1_mols
train_names   = main_names   + p1_names
train_targets = np.concatenate([main_targets, p1_targets])
all_se        = np.concatenate([main_se,      p1_se])
all_counters  = np.concatenate([main_counters, p1_counters])

# 1/std_error, jointly clipped and normalised
raw_weights   = 1.0 / np.where(all_se > 0, all_se, 1e-6)
w_precision   = normalize_weights(raw_weights)
w_counter     = counter_weights(all_counters)
train_weights = w_precision * w_counter

n_total        = len(train_mols)
n_with_counter = (~np.isnan(all_counters)).sum()

print(f"\nCombined training set:")
print(f"  Main SDF        : {len(main_mols):5d}")
print(f"  Phase-1 added   : {len(p1_mols):5d}")
print(f"  Total           : {n_total:5d}")
print(f"  Counter avail.  : {n_with_counter}/{n_total} "
      f"({100 * n_with_counter / n_total:.1f}%)")
print(f"  pEC50 range     : "
      f"{train_targets.min():.2f} – {train_targets.max():.2f}  "
      f"(mean {train_targets.mean():.2f})")
print(f"  Weight range    : "
      f"{train_weights.min():.3f} – {train_weights.max():.3f}  "
      f"(mean {train_weights.mean():.3f})")

# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — Load test set (260 non-Phase-1 compounds)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nLoading test SDF (260 non-Phase-1 compounds):\n  {TEST_SDF}")
test_mols, test_names, test_targets = load_sdf_test(TEST_SDF, phase1_name_set)
print(f"  pEC50 range : {test_targets.min():.2f} – {test_targets.max():.2f}")

# ══════════════════════════════════════════════════════════════════════════════
# Step 5 — Stratification labels for CV
# ══════════════════════════════════════════════════════════════════════════════
strata = pd.qcut(
    train_targets, q=N_STRATA_BINS, labels=False, duplicates="drop"
).astype(int)

print(f"\nStratification ({N_STRATA_BINS} bins):")
for b in range(strata.max() + 1):
    mask = strata == b
    print(f"  Bin {b}: n={mask.sum():4d}  "
          f"pEC50 {train_targets[mask].min():.2f}–{train_targets[mask].max():.2f}")

# ══════════════════════════════════════════════════════════════════════════════
# Step 6 — 5-fold stratified CV grid search
# ══════════════════════════════════════════════════════════════════════════════
if CV_RESULTS_PATH.exists():
    print(f"\nCV results found at {CV_RESULTS_PATH} — skipping grid search.")
    df_cv = pd.read_csv(CV_RESULTS_PATH).sort_values("mean_val_loss").reset_index(drop=True)
    print(f"\n{df_cv.to_string(index=False)}")
else:
    param_combos = [
        dict(zip(PARAM_GRID.keys(), combo))
        for combo in itertools.product(*PARAM_GRID.values())
    ]

    print(f"\n{'='*70}")
    print(f"5-fold stratified CV — {len(param_combos)} hyperparameter combos")
    print(f"{'='*70}")

    skf        = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    indices    = np.arange(len(train_mols))
    cv_results = []

    for combo_idx, params in enumerate(param_combos):
        print(f"\n[{combo_idx + 1}/{len(param_combos)}] {params}")
        fold_losses, fold_epochs = [], []

        for fold_num, (tr_idx, va_idx) in enumerate(skf.split(indices, strata)):
            set_seed(42)

            val_loss, best_epoch = run_fold(
                fold_num,
                [train_mols[i] for i in tr_idx], train_targets[tr_idx], train_weights[tr_idx],
                [train_mols[i] for i in va_idx], train_targets[va_idx], train_weights[va_idx],
                params, max_epochs=CV_MAX_EPOCHS, patience=CV_PATIENCE,
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
best_row    = df_cv.iloc[0]
best_params = {
    "ffn_hidden_dim": int(best_row["ffn_hidden_dim"]),
    "ffn_n_layers":   int(best_row["ffn_n_layers"]),
    "dropout":        float(best_row["dropout"]),
    "mp_depth":       int(best_row["mp_depth"]),
    "mp_hidden_dim":  int(best_row["mp_hidden_dim"]),
}
final_epochs = max(int(best_row["mean_best_epoch"] * 1.1), 80)

print(f"\nBest hyperparameters : {best_params}")
print(f"Final model epochs   : {final_epochs}")
print(f"Ensemble size        : {len(ENSEMBLE_SEEDS)} seeds")

# ══════════════════════════════════════════════════════════════════════════════
# Step 7 — Build test dataloader
# ══════════════════════════════════════════════════════════════════════════════
feat        = featurizers.SimpleMoleculeMolGraphFeaturizer()
test_dps    = make_datapoints(test_mols, test_targets, np.ones(len(test_mols)))
test_dset   = data.MoleculeDataset(test_dps, feat)
test_loader = data.build_dataloader(test_dset, num_workers=NUM_WORKERS, shuffle=False)

# ══════════════════════════════════════════════════════════════════════════════
# Step 8 — Train ensemble, predict test set
# ══════════════════════════════════════════════════════════════════════════════
ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)
all_test_preds    = []
per_model_metrics = []

for i, seed in enumerate(ENSEMBLE_SEEDS):
    print(f"\n{'='*70}")
    print(f"Ensemble member {i + 1}/{len(ENSEMBLE_SEEDS)}  |  seed={seed}")
    print(f"{'='*70}")

    set_seed(seed)

    train_dps  = make_datapoints(train_mols, train_targets, train_weights)
    train_dset = data.MoleculeDataset(train_dps, feat)
    target_scaler = train_dset.normalize_targets()
    train_loader  = data.build_dataloader(train_dset, num_workers=NUM_WORKERS)

    mpnn = build_mpnn(**best_params, target_scaler=target_scaler)

    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        accelerator="auto",
        devices=1,
        max_epochs=final_epochs,
    )
    trainer.fit(mpnn, train_loader)

    torch.save(mpnn, ENSEMBLE_DIR / f"model_seed{seed}.pt")

    mpnn.eval()
    raw_preds = trainer.predict(mpnn, test_loader)
    preds_i   = torch.cat(raw_preds).numpy().flatten()
    all_test_preds.append(preds_i)

    m = report_metrics(test_targets, preds_i, label=f"seed={seed}")
    m["seed"] = seed
    per_model_metrics.append(m)

# ══════════════════════════════════════════════════════════════════════════════
# Step 9 — Ensemble and output
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Ensemble results")
print(f"{'='*70}")

all_preds_array  = np.stack(all_test_preds, axis=0)
ensemble_preds   = all_preds_array.mean(axis=0)
ensemble_metrics = report_metrics(test_targets, ensemble_preds, label="ENSEMBLE")

pred_cols = {f"pEC50_seed{s}": p for s, p in zip(ENSEMBLE_SEEDS, all_test_preds)}
df_out = pd.DataFrame({
    "Molecule Name":    test_names,
    "pEC50_sdf":        test_targets,
    "pEC50_ensemble":   ensemble_preds,
    "residual":         test_targets - ensemble_preds,
    **pred_cols,
})
df_out.to_csv(OUTPUT_PREDS, index=False)
print(f"\nPredictions saved to {OUTPUT_PREDS}")

df_members = pd.DataFrame(per_model_metrics)
print("\nPer-member test-set metrics:")
print(df_members.to_string(index=False))
print(f"\nMean individual MAE : {df_members['mae'].mean():.4f} ± {df_members['mae'].std():.4f}")
print(f"Ensemble MAE        : {ensemble_metrics['mae']:.4f}")
print(f"Gain vs mean indiv. : {df_members['mae'].mean() - ensemble_metrics['mae']:.4f}")

m = ensemble_metrics
print(
    f"\n[Leaderboard format]  "
    f"MAE={m['mae']:.4f}  RAE={m['rae']:.4f}  "
    f"R²={m['r2']:.4f}  Spearman={m['spearman']:.4f}  Kendall={m['kendall']:.4f}"
)
