"""
ChemProp PXR pEC50 — reads training data from SDF (MOE-prepped 3D structures).

Training data: train_ci_filtered_aug.sdf
  - Active compounds (pEC50 > 3.5): original MOE-prepped 3D mol blocks
  - Inactive compounds (pEC50 <= 3.5, CI upper <= 4.0): original + 3 randomized
    SMILES copies (2D); 8 ambiguous inactives with CI upper > 4.0 removed

Weighting: two independent signals multiplied together.
    1. sample_weight (1/std_error) — measurement precision of the primary assay.
       Clipped at p99 then normalised to [0.5, 2.0].
    2. Counter-screen (pEC50_counter) — inversely mapped to [0.5, 2.0] so
       selective compounds (low counter potency) get higher weight and
       promiscuous hits get lower weight. Defaults to neutral 1.0 when absent
       (23% of inactives, 14–20% of actives lack counter data).
    Final weight = norm(sample_weight) × norm(counter_weight).
    Augmented copies inherit their original compound's weights.

CV leakage prevention:
    Fold split at compound (Name) level.
    Augmented copies only appear in training folds.
    Validation uses the original (is_augmented=0) entry per compound.

Usage:
    conda activate chemprop
    python ~/dockerimages/QSARTuna/chemprop_pxr_pec50_ci_filtered_aug_sdf.py
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
TRAIN_SDF = Path("/home/spal/dockerimages/QSARTuna/PXR/train_ci_filtered_aug.sdf")
TEST_SDF  = Path("/home/spal/dockerimages/QSARTuna/PXR/test_OpenADMET_Data_prepped.sdf")
CV_RESULTS_PATH = Path.home() / "pxr_chemprop_ci_filt_aug_wtd_cv_results.csv"
ENSEMBLE_DIR    = Path.home() / "pxr_ensemble_models_ci_filt_aug_wtd"
OUTPUT_PREDS    = Path.home() / "pxr_chemprop_ci_filt_aug_wtd_test_predictions.csv"

# ── SD tag names ───────────────────────────────────────────────────────────────
TAG_TARGET  = "pEC50"
TAG_WEIGHT  = "sample_weight"    # 1/std_error, pre-computed in SDF
TAG_COUNTER = "pEC50_counter"
TAG_NAME    = "Name"
TAG_AUG     = "is_augmented"

# ── Weight normalisation range (applied to each component separately) ──────────
WEIGHT_MIN     = 0.5
WEIGHT_MAX     = 2.0
NEUTRAL_WEIGHT = 1.0             # counter-screen default when data absent

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
    """
    Inversely map pEC50_counter to [WEIGHT_MIN, WEIGHT_MAX]:
      low counter potency  → WEIGHT_MAX  (selective compound, trust more)
      high counter potency → WEIGHT_MIN  (promiscuous hit, down-weight)
      NaN                  → NEUTRAL_WEIGHT
    """
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


# ── SDF loader (training) ──────────────────────────────────────────────────────
def load_sdf_train(sdf_path: Path):
    """
    Load training SDF. Returns parallel arrays for all rows plus compound-level
    indices for CV stratification.

    Augmented entries (is_augmented=1) are included in training folds only;
    validation always uses the original (is_augmented=0) entry.
    """
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=True)
    mols, names, targets, raw_weights, counters, is_aug = [], [], [], [], [], []
    skipped = 0

    for mol in suppl:
        if mol is None:
            skipped += 1
            continue
        name      = mol.GetProp(TAG_NAME) if mol.HasProp(TAG_NAME) else mol.GetProp("_Name")
        target    = float(mol.GetProp(TAG_TARGET))
        raw_wt    = float(mol.GetProp(TAG_WEIGHT))  if mol.HasProp(TAG_WEIGHT)  else 1.0
        counter   = float(mol.GetProp(TAG_COUNTER)) if mol.HasProp(TAG_COUNTER) else np.nan
        augmented = int(mol.GetProp(TAG_AUG))       if mol.HasProp(TAG_AUG)     else 0

        mols.append(mol)
        names.append(name)
        targets.append(target)
        raw_weights.append(raw_wt)
        counters.append(counter)
        is_aug.append(augmented)

    targets     = np.array(targets,     dtype=float)
    raw_weights = np.array(raw_weights, dtype=float)
    counters    = np.array(counters,    dtype=float)
    is_aug      = np.array(is_aug,      dtype=int)

    # unique_positions: index of the original (non-augmented) entry per compound,
    # used for CV stratification and validation.
    seen, unique_positions = set(), []
    for idx, (name, aug) in enumerate(zip(names, is_aug)):
        if name not in seen and aug == 0:
            seen.add(name)
            unique_positions.append(idx)

    # name → all positions (incl. augmented copies)
    name_to_positions: dict[str, list[int]] = {}
    for pos, name in enumerate(names):
        name_to_positions.setdefault(name, []).append(pos)

    n_aug_entries = int(is_aug.sum())
    print(f"  Loaded {len(mols)} rows  "
          f"({len(unique_positions)} unique compounds, {n_aug_entries} augmented rows)")
    if skipped:
        print(f"  {skipped} molecules skipped (unreadable)")

    return mols, names, targets, raw_weights, counters, is_aug, unique_positions, name_to_positions


# ── SDF loader (test set) ──────────────────────────────────────────────────────
def load_sdf_test(sdf_path: Path):
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=True)
    mols, names, targets = [], [], []
    skipped = 0

    for mol in suppl:
        if mol is None:
            skipped += 1
            continue
        if not mol.HasProp(TAG_TARGET):
            skipped += 1
            continue
        name = mol.GetProp(TAG_NAME) if mol.HasProp(TAG_NAME) else mol.GetProp("_Name")
        mols.append(mol)
        names.append(name)
        targets.append(float(mol.GetProp(TAG_TARGET)))

    if skipped:
        print(f"  {skipped} molecules skipped")
    return mols, names, np.array(targets, dtype=float)


# ── MPNN builder ───────────────────────────────────────────────────────────────
def build_mpnn(
    ffn_hidden_dim, ffn_n_layers, dropout, mp_depth, mp_hidden_dim, target_scaler
) -> models.MPNN:
    feat = featurizers.SimpleMoleculeMolGraphFeaturizer()
    mp   = nn.BondMessagePassing(
               d_v=feat.atom_fdim, d_e=feat.bond_fdim,
               depth=mp_depth, d_h=mp_hidden_dim,
           )
    agg  = nn.MeanAggregation()
    output_transform = nn.UnscaleTransform.from_standard_scaler(target_scaler)
    ffn  = nn.RegressionFFN(
        input_dim=mp.output_dim,
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
# Step 1 — Load training SDF
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nLoading training SDF:\n  {TRAIN_SDF}")
(
    train_mols, train_names, train_targets, raw_weights, counter_values,
    is_aug_flags, unique_positions, name_to_positions,
) = load_sdf_train(TRAIN_SDF)

# Combined weight: precision × selectivity
w_precision  = normalize_weights(raw_weights)
w_counter    = counter_weights(counter_values)
train_weights = w_precision * w_counter
n_unique      = len(unique_positions)

# Unique-compound arrays (used for CV stratification)
uniq_mols    = [train_mols[i]    for i in unique_positions]
uniq_targets = train_targets[unique_positions]
uniq_weights = train_weights[unique_positions]

n_with_counter = (~np.isnan(counter_values)).sum()
print(f"  Total rows            : {len(train_mols)}")
print(f"  Unique compounds      : {n_unique}")
print(f"  Augmented extra rows  : {is_aug_flags.sum()}")
print(f"  Counter screen avail. : {n_with_counter}/{len(train_mols)} "
      f"({100 * n_with_counter / len(train_mols):.1f}%)")
print(f"  pEC50 range           : "
      f"{train_targets.min():.2f} – {train_targets.max():.2f}  "
      f"(mean {train_targets.mean():.2f})")
print(f"  Combined weight range : "
      f"{train_weights.min():.3f} – {train_weights.max():.3f}  "
      f"(mean {train_weights.mean():.3f})")

# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — Load test SDF
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nLoading test SDF:\n  {TEST_SDF}")
test_mols, test_names, test_targets = load_sdf_test(TEST_SDF)
print(f"  Loaded {len(test_mols)} molecules")
print(f"  pEC50 range : {test_targets.min():.2f} – {test_targets.max():.2f}")

# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — Stratification labels for CV
# ══════════════════════════════════════════════════════════════════════════════
strata = pd.qcut(
    uniq_targets, q=N_STRATA_BINS, labels=False, duplicates="drop"
).astype(int)

print(f"\nStratification ({N_STRATA_BINS} bins, unique compounds only):")
for b in range(strata.max() + 1):
    mask = strata == b
    print(f"  Bin {b}: n={mask.sum():4d}  "
          f"pEC50 {uniq_targets[mask].min():.2f}–{uniq_targets[mask].max():.2f}")

# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — 5-fold stratified CV grid search (compound-level, leakage-safe)
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
    print(f"5-fold CV — {len(param_combos)} hyperparameter combos")
    print("Train folds include all augmented copies; val uses original entry only.")
    print(f"{'='*70}")

    skf          = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    uniq_indices = np.arange(n_unique)
    cv_results   = []

    for combo_idx, params in enumerate(param_combos):
        print(f"\n[{combo_idx + 1}/{len(param_combos)}] {params}")
        fold_losses, fold_epochs = [], []

        for fold_num, (tr_uniq_idx, va_uniq_idx) in enumerate(
            skf.split(uniq_indices, strata)
        ):
            set_seed(42)

            # Training: all rows (incl. augmented) for each training compound
            tr_positions = [
                pos
                for i in tr_uniq_idx
                for pos in name_to_positions[train_names[unique_positions[i]]]
            ]
            # Validation: original entry only for each validation compound
            va_positions = [unique_positions[i] for i in va_uniq_idx]

            val_loss, best_epoch = run_fold(
                fold_num,
                [train_mols[p]    for p in tr_positions],
                train_targets[tr_positions],
                train_weights[tr_positions],
                [train_mols[p]    for p in va_positions],
                train_targets[va_positions],
                train_weights[va_positions],
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
# Step 5 — Build test dataloader
# ══════════════════════════════════════════════════════════════════════════════
feat             = featurizers.SimpleMoleculeMolGraphFeaturizer()
test_weights_d   = np.ones(len(test_mols))
test_dps         = make_datapoints(test_mols, test_targets, test_weights_d)
test_dset        = data.MoleculeDataset(test_dps, feat)
test_loader      = data.build_dataloader(test_dset, num_workers=NUM_WORKERS, shuffle=False)

# ══════════════════════════════════════════════════════════════════════════════
# Step 6 — Train ensemble on all augmented training data
# ══════════════════════════════════════════════════════════════════════════════
ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)
all_test_preds    = []
per_model_metrics = []

for i, seed in enumerate(ENSEMBLE_SEEDS):
    print(f"\n{'='*70}")
    print(f"Ensemble member {i + 1}/{len(ENSEMBLE_SEEDS)}  |  seed={seed}")
    print(f"{'='*70}")

    set_seed(seed)

    all_train_dps  = make_datapoints(train_mols, train_targets, train_weights)
    all_train_dset = data.MoleculeDataset(all_train_dps, feat)
    target_scaler  = all_train_dset.normalize_targets()
    train_loader   = data.build_dataloader(all_train_dset, num_workers=NUM_WORKERS)

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

    model_path = ENSEMBLE_DIR / f"model_seed{seed}.pt"
    torch.save(mpnn, model_path)
    print(f"  Saved → {model_path}")

    mpnn.eval()
    raw_preds = trainer.predict(mpnn, test_loader)
    preds_i   = torch.cat(raw_preds).numpy().flatten()
    all_test_preds.append(preds_i)

    m = report_metrics(test_targets, preds_i, label=f"seed={seed}")
    m["seed"] = seed
    per_model_metrics.append(m)

# ══════════════════════════════════════════════════════════════════════════════
# Step 7 — Ensemble prediction
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Ensemble results")
print(f"{'='*70}")

all_preds_array  = np.stack(all_test_preds, axis=0)
ensemble_preds   = all_preds_array.mean(axis=0)
ensemble_metrics = report_metrics(test_targets, ensemble_preds, label="ENSEMBLE")

pred_cols = {f"pEC50_seed{s}": p for s, p in zip(ENSEMBLE_SEEDS, all_test_preds)}
df_out = pd.DataFrame({
    "Molecule Name":  test_names,
    "pEC50_actual":   test_targets,
    "pEC50_ensemble": ensemble_preds,
    "residual":       test_targets - ensemble_preds,
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
