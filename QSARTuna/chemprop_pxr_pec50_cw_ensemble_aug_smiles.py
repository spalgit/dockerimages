"""
ChemProp PXR pEC50 — pure chemprop (no RDKit 2D) + counter-assay weighting
+ multi-seed ensemble + inactive-SMILES augmentation.

Training data: train_cliff_error_augmented.csv (6050 rows / 4124 unique compounds)
  - 963 inactive compounds (pEC50 ≤ 3.5) represented by 3 random SMILES each
  - 3161 active/moderate compounds represented by 1 canonical SMILES
  - Weight column in CSV is placeholder (all 1.0); recomputed from pEC50_counter here

Weighting scheme (same as rdkit2d script):
    pEC50_counter INVERSELY mapped to [MIN_WEIGHT, MAX_WEIGHT]:
        low counter potency  → high weight  (selective, trust more)
        high counter potency → low weight   (promiscuous, down-weight)
        NaN                  → NEUTRAL_WEIGHT

CV leakage prevention:
    Fold split is at the compound (Molecule Name) level.
    Augmented SMILES copies only appear in training folds.
    Validation uses ONE canonical SMILES per compound.

Usage:
    conda activate chemprop
    python ~/dockerimages/QSARTuna/chemprop_pxr_pec50_cw_ensemble_aug_smiles.py
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
TRAIN_CSV = Path(
    "/home/spal/dockerimages/QSARTuna/PXR/train_cliff_error_augmented.csv"
)
TEST_SDF = Path(
    "/home/spal/dockerimages/QSARTuna/PXR/test_OpenADMET_Data_prepped.sdf"
)
CV_RESULTS_PATH = Path.home() / "pxr_chemprop_aug_cv_results.csv"
ENSEMBLE_DIR    = Path.home() / "pxr_ensemble_models_aug"
OUTPUT_PREDS    = Path.home() / "pxr_chemprop_aug_test_predictions.csv"

# ── SD tag / column names ──────────────────────────────────────────────────────
TAG_TARGET  = "pEC50"
TAG_COUNTER = "pEC50_counter"

# ── Counter-screen weighting ───────────────────────────────────────────────────
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


# ── CSV loader ─────────────────────────────────────────────────────────────────
def load_csv(csv_path: Path):
    """
    Load training CSV (including augmented rows).
    Returns parallel arrays for all rows, plus a DataFrame of unique compounds
    (first occurrence per Molecule Name) for CV stratification.

    Weights are recomputed from pEC50_counter (CSV weight column is ignored).
    """
    df = pd.read_csv(csv_path)
    mols, names, smiles_list, targets, counters = [], [], [], [], []
    skipped_idx = set()

    for i, row in df.iterrows():
        mol = Chem.MolFromSmiles(str(row["SMILES"]))
        if mol is None:
            skipped_idx.add(i)
            print(f"  Row {i} skipped (unparseable SMILES): {row['SMILES'][:60]}")
            continue
        mols.append(mol)
        names.append(str(row["Molecule Name"]))
        smiles_list.append(str(row["SMILES"]))  # keep original SMILES string
        targets.append(float(row[TAG_TARGET]))
        counters.append(float(row[TAG_COUNTER]) if pd.notna(row[TAG_COUNTER]) else np.nan)

    targets  = np.array(targets,  dtype=float)
    counters = np.array(counters, dtype=float)

    # Build unique-compound index for CV (first occurrence of each name)
    seen, unique_rows = set(), []
    for idx, name in enumerate(names):
        if name not in seen:
            seen.add(name)
            unique_rows.append(idx)  # position in the parallel arrays above

    # Map name → list of positions in the parallel arrays
    name_to_positions: dict[str, list[int]] = {}
    for pos, name in enumerate(names):
        name_to_positions.setdefault(name, []).append(pos)

    print(f"  Loaded {len(mols)} rows  ({len(unique_rows)} unique compounds)")
    return mols, names, smiles_list, targets, counters, unique_rows, name_to_positions


# ── SDF loader (test set) ──────────────────────────────────────────────────────
def load_sdf(sdf_path: Path, required_tags: list, optional_tags: list = None):
    optional_tags = optional_tags or []
    all_tags = required_tags + optional_tags

    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=True)
    mols, names, smiles_list = [], [], []
    rows = {t: [] for t in all_tags}
    skipped = 0

    for mol in suppl:
        if mol is None:
            skipped += 1
            continue
        missing = [t for t in required_tags if not mol.HasProp(t)]
        if missing:
            name = mol.GetProp("_Name") if mol.HasProp("_Name") else "?"
            print(f"  Skipped {name}: missing required tags {missing}")
            skipped += 1
            continue

        mols.append(mol)
        names.append(mol.GetProp("_Name") if mol.HasProp("_Name") else "")
        smiles_list.append(Chem.MolToSmiles(mol))

        for tag in all_tags:
            if mol.HasProp(tag):
                try:
                    rows[tag].append(float(mol.GetPropsAsDict()[tag]))
                except (ValueError, TypeError):
                    rows[tag].append(np.nan)
            else:
                rows[tag].append(np.nan)

    tag_arrays = {t: np.array(v, dtype=float) for t, v in rows.items()}
    if skipped:
        print(f"  {skipped} molecules skipped")
    return mols, names, smiles_list, tag_arrays


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


# ── MPNN builder (no extra descriptors) ───────────────────────────────────────
def build_mpnn(
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
        data.MoleculeDatapoint(
            mol=mol,
            y=np.array([t], dtype=float),
            weight=float(w),
        )
        for mol, t, w in zip(mols, targets, weights)
    ]


# ── Metrics helper ─────────────────────────────────────────────────────────────
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


# ── One CV fold ────────────────────────────────────────────────────────────────
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
# Step 1 — Load training CSV (augmented)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nLoading training CSV (augmented):\n  {TRAIN_CSV}")
(
    train_mols, train_names, train_smiles,
    train_targets, counter_values,
    unique_positions, name_to_positions,
) = load_csv(TRAIN_CSV)

train_weights = compute_weights(counter_values)

n_unique      = len(unique_positions)
n_augmented   = len(train_mols) - n_unique
n_with_counter = (~np.isnan(counter_values)).sum()

# Unique-compound arrays (used for CV stratification)
uniq_mols    = [train_mols[i]    for i in unique_positions]
uniq_targets = train_targets[unique_positions]
uniq_weights = train_weights[unique_positions]

print(f"  Total rows            : {len(train_mols)}")
print(f"  Unique compounds      : {n_unique}")
print(f"  Augmented extra rows  : {n_augmented}")
print(f"  Counter screen avail. : {n_with_counter}/{len(train_mols)} "
      f"({100 * n_with_counter / len(train_mols):.1f}%)")
print(f"  pEC50 range           : "
      f"{train_targets.min():.2f} – {train_targets.max():.2f}  "
      f"(mean {train_targets.mean():.2f})")
print(f"  Weight range          : "
      f"{train_weights.min():.3f} – {train_weights.max():.3f}  "
      f"(neutral={NEUTRAL_WEIGHT})")

# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — Load test SDF
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nLoading test SDF:\n  {TEST_SDF}")
test_mols, test_names, test_smiles, test_tags = load_sdf(
    TEST_SDF,
    required_tags=[TAG_TARGET],
)
test_targets = test_tags[TAG_TARGET]
print(f"  Loaded {len(test_mols)} molecules")
print(f"  pEC50 range : {test_targets.min():.2f} – {test_targets.max():.2f}  "
      f"(mean {test_targets.mean():.2f})")

# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — Stratification labels for CV (at unique-compound level)
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
# Step 4 — 5-fold stratified CV grid search (group-aware)
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
    print(f"Train folds include all augmented SMILES; val uses canonical SMILES only.")
    print(f"{'='*70}")

    skf             = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    uniq_indices    = np.arange(n_unique)
    cv_results      = []

    for combo_idx, params in enumerate(param_combos):
        print(f"\n[{combo_idx + 1}/{len(param_combos)}] {params}")
        fold_losses, fold_epochs = [], []

        for fold_num, (tr_uniq_idx, va_uniq_idx) in enumerate(
            skf.split(uniq_indices, strata)
        ):
            set_seed(42)

            # Training: all rows (incl. augmented copies) for each training compound
            tr_positions = [
                pos
                for i in tr_uniq_idx
                for pos in name_to_positions[train_names[unique_positions[i]]]
            ]
            # Validation: first/canonical SMILES only for each val compound
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
print(f"Final model epochs   : {final_epochs}  "
      f"(= max(CV mean_best_epoch {int(best_row['mean_best_epoch'])} × 1.1, 80))")
print(f"Ensemble size        : {len(ENSEMBLE_SEEDS)} seeds → {ENSEMBLE_SEEDS}")

# ══════════════════════════════════════════════════════════════════════════════
# Step 5 — Build test dataloader
# ══════════════════════════════════════════════════════════════════════════════
feat = featurizers.SimpleMoleculeMolGraphFeaturizer()

test_weights_dummy = np.ones(len(test_mols))
test_dps    = make_datapoints(test_mols, test_targets, test_weights_dummy)
test_dset   = data.MoleculeDataset(test_dps, feat)
test_loader = data.build_dataloader(test_dset, num_workers=NUM_WORKERS, shuffle=False)

# ══════════════════════════════════════════════════════════════════════════════
# Step 6 — Train N ensemble members on ALL augmented training data
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
# Step 7 — Ensemble prediction (mean across all seeds)
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
    "SMILES":         test_smiles,
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
