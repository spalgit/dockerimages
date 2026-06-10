"""
ChemProp PXR pEC50 — RDKit 2D descriptors + counter-assay weighting +
activity-range weighting + RMSE loss + CV-weighted multi-seed ensemble.

Changes vs chemprop_pxr_pec50_rdkit2d_counter_weight_ensemble_sdf.py:
  1. RMSE criterion (penalises large errors quadratically vs MAE)
  2. Activity-range weighting on top of counter-screen weighting:
       pEC50 < ACTIVITY_LOW_THR  → linearly upweighted (worst overprediction zone)
       pEC50 > ACTIVITY_HIGH_THR → constant boost (worst underprediction zone)
  3. CV-weighted ensemble: a stratified calibration holdout (CALIB_FRACTION of
     training data) is used to score each seed by its test RMSE; seeds are
     combined with inverse-RMSE weights rather than a flat mean.
  4. Protonation-safe SDF loading: molecules are read with removeHs=False so the
     exact protonation state written by MOE is preserved; Chem.RemoveHs() then
     strips only non-essential explicit Hs to produce the heavy-atom graph for
     ChemProp and RDKit descriptors. The input SDF MUST already be protonated
     (e.g. by MOE). No protonation tool is called inside this script.

Usage:
    conda activate chemprop
    python ~/dockerimages/QSARTuna/chemprop_pxr_pec50_rdkit2d_rmse_actweight_cvens_sdf.py
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
TRAIN_SDF = Path(
    "/home/spal/dockerimages/QSARTuna/PXR_For_QSARTuna_Modelling/"
    "processed_Openadmet_REAL_PXR_train_AND_test_main_with_side_info_"
    "AND_counter_screen_weighted_moe_prepped.sdf"
)
TEST_SDF = Path(
    "/home/spal/dockerimages/QSARTuna/PXR/test_OpenADMET_Data_prepped.sdf"
)
CV_RESULTS_PATH  = Path.home() / "pxr_rdkit2d_rmse_actweight_cvens_sdf_cv_results.csv"
ENSEMBLE_DIR     = Path.home() / "pxr_ensemble_models_rmse_actweight_cvens_sdf"
OUTPUT_PREDS     = Path.home() / "pxr_rdkit2d_rmse_actweight_cvens_sdf_test_predictions.csv"
KEPT_DESCS_PATH  = Path.home() / "pxr_rdkit2d_rmse_actweight_cvens_sdf_kept_descriptors.txt"
CALIB_PREDS_PATH = Path.home() / "pxr_rdkit2d_rmse_actweight_cvens_sdf_calib_predictions.csv"

# ── SD tag names ───────────────────────────────────────────────────────────────
TAG_TARGET  = "pEC50"
TAG_COUNTER = "pEC50_counter"

# ── Counter-screen weighting ───────────────────────────────────────────────────
MIN_WEIGHT     = 0.5
MAX_WEIGHT     = 2.0
NEUTRAL_WEIGHT = 1.0

# ── Activity-range weighting ───────────────────────────────────────────────────
# Compounds below LOW_THR get a linearly increasing boost (worst overprediction zone)
# Compounds above HIGH_THR get a constant boost (worst underprediction zone)
ACTIVITY_LOW_THR   = 4.0
ACTIVITY_HIGH_THR  = 6.0
ACTIVITY_TAIL_BOOST = 2.5   # max multiplier at the tails

# ── CV-weighted ensemble calibration split ─────────────────────────────────────
# Fraction of TRAINING data held out to score each ensemble seed.
# Seeds are averaged with weights = softmax(-CALIB_RMSE / WEIGHT_TEMPERATURE).
CALIB_FRACTION      = 0.15
WEIGHT_TEMPERATURE  = 0.05   # lower → more aggressive weight differentiation

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


# ── SDF loader ─────────────────────────────────────────────────────────────────
def load_sdf(sdf_path: Path, required_tags: list, optional_tags: list = None):
    """
    Load a pre-protonated SDF and return (mols, names, tag_dict, smiles_list).

    Molecules are read with removeHs=False to capture the exact protonation
    state written by MOE. Chem.RemoveHs() then strips only non-essential
    explicit Hs (stereo Hs and charged Hs are kept as implicit counts), giving
    the heavy-atom graph expected by ChemProp and RDKit descriptors.

    The caller MUST supply an already-protonated SDF. Protonation is not applied
    here.
    """
    optional_tags = optional_tags or []
    all_tags = required_tags + optional_tags

    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=True)
    mols, names, smiles_list = [], [], []
    rows = {t: [] for t in all_tags}
    skipped = 0

    for mol in suppl:
        if mol is None:
            skipped += 1
            continue

        # Strip only non-essential explicit Hs; preserves formal charges,
        # stereo Hs, and the protonation state encoded in implicit H counts.
        mol = Chem.RemoveHs(mol)
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
        print(f"  {skipped} molecules skipped (None or missing required tags)")
    return mols, names, smiles_list, tag_arrays


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
def compute_counter_weights(counter_values: np.ndarray) -> np.ndarray:
    """Inversely map counter-screen pEC50 → [MIN_WEIGHT, MAX_WEIGHT]."""
    weights     = np.full(len(counter_values), NEUTRAL_WEIGHT, dtype=float)
    has_counter = ~np.isnan(counter_values)
    vals        = counter_values[has_counter]
    c_min, c_max = vals.min(), vals.max()
    if c_max > c_min:
        weights[has_counter] = MIN_WEIGHT + (MAX_WEIGHT - MIN_WEIGHT) * (
            (c_max - vals) / (c_max - c_min)
        )
    return weights


# ── Activity-range weighting ───────────────────────────────────────────────────
def compute_activity_weights(targets: np.ndarray) -> np.ndarray:
    """
    Upweight compounds at the extremes of the activity range.

    pEC50 < ACTIVITY_LOW_THR:
        Linear boost from 1.0 at the threshold up to ACTIVITY_TAIL_BOOST at 0.
    pEC50 > ACTIVITY_HIGH_THR:
        Constant boost of ACTIVITY_TAIL_BOOST.
    """
    w = np.ones(len(targets), dtype=float)

    below = targets < ACTIVITY_LOW_THR
    if below.any():
        # fraction below threshold: 0 at LOW_THR, 1 at 0
        frac = np.clip((ACTIVITY_LOW_THR - targets[below]) / ACTIVITY_LOW_THR,
                       0.0, 1.0)
        w[below] = 1.0 + (ACTIVITY_TAIL_BOOST - 1.0) * frac

    above = targets > ACTIVITY_HIGH_THR
    w[above] = ACTIVITY_TAIL_BOOST

    return w


# ── Combined weights ───────────────────────────────────────────────────────────
def compute_combined_weights(targets: np.ndarray,
                             counter_values: np.ndarray) -> np.ndarray:
    cw = compute_counter_weights(counter_values)
    aw = compute_activity_weights(targets)
    combined = cw * aw
    # Renormalise so the mean weight stays at 1.0 (keeps effective lr stable)
    combined /= combined.mean()
    return combined


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
        criterion=nn.metrics.RMSE(),      # ← RMSE replaces MAE
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


# ── Metrics helper ─────────────────────────────────────────────────────────────
def report_metrics(actual: np.ndarray, predicted: np.ndarray,
                   label: str = "") -> dict:
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
    train_mols, train_targets, train_weights, train_x_d,
    val_mols,   val_targets,   val_weights,   val_x_d,
    n_descriptors, params, max_epochs, patience,
):
    feat      = featurizers.SimpleMoleculeMolGraphFeaturizer()
    train_dps = make_datapoints(train_mols, train_targets, train_weights, train_x_d)
    val_dps   = make_datapoints(val_mols,   val_targets,   val_weights,   val_x_d)

    train_dset = data.MoleculeDataset(train_dps, feat)
    val_dset   = data.MoleculeDataset(val_dps,   feat)

    target_scaler = train_dset.normalize_targets()
    val_dset.normalize_targets(target_scaler)

    train_loader = data.build_dataloader(train_dset, num_workers=NUM_WORKERS)
    val_loader   = data.build_dataloader(val_dset,   num_workers=NUM_WORKERS,
                                         shuffle=False)

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
# Step 1 — Load training SDF  (must be protonated — no protonation done here)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nLoading training SDF:\n  {TRAIN_SDF}")
train_mols, train_names, train_smiles, train_tags = load_sdf(
    TRAIN_SDF,
    required_tags=[TAG_TARGET],
    optional_tags=[TAG_COUNTER],
)
train_targets  = train_tags[TAG_TARGET]
counter_values = train_tags[TAG_COUNTER]
train_weights  = compute_combined_weights(train_targets, counter_values)

n_with_counter = (~np.isnan(counter_values)).sum()
print(f"  Loaded {len(train_mols)} molecules")
print(f"  Counter screen available : {n_with_counter}/{len(train_mols)} "
      f"({100 * n_with_counter / len(train_mols):.1f}%)")
print(f"  pEC50 range              : "
      f"{train_targets.min():.2f} – {train_targets.max():.2f}  "
      f"(mean {train_targets.mean():.2f})")
print(f"  Combined weight range    : "
      f"{train_weights.min():.3f} – {train_weights.max():.3f}  "
      f"(mean {train_weights.mean():.3f})")

# Weight distribution by activity bin for transparency
print("  Activity-bin weight summary:")
for lo, hi, label in [(0,4,'<4'), (4,5,'4-5'), (5,6,'5-6'), (6,99,'>6')]:
    mask = (train_targets >= lo) & (train_targets < hi)
    if mask.sum():
        print(f"    pEC50 {label:>4}: n={mask.sum():4d}  "
              f"mean_w={train_weights[mask].mean():.3f}  "
              f"max_w={train_weights[mask].max():.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — Load test SDF  (must be protonated — no protonation done here)
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
# Step 3 — RDKit 2D descriptors; column selection from training set only
# ══════════════════════════════════════════════════════════════════════════════
print("\nComputing RDKit 2D descriptors for training set...")
x_d_train_all = compute_rdkit_descriptors(train_mols)

col_mask   = select_valid_columns(x_d_train_all)
kept_names = [ALL_DESC_NAMES[i] for i, k in enumerate(col_mask) if k]
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
# Step 4 — Calibration holdout split for CV-weighted ensemble
#
# A stratified CALIB_FRACTION of training data is held out to score each
# ensemble seed independently. Seeds are then combined with inverse-RMSE
# weights rather than a flat mean.
# ══════════════════════════════════════════════════════════════════════════════
strata_all = pd.qcut(
    train_targets, q=N_STRATA_BINS, labels=False, duplicates="drop"
).astype(int)

skf_calib = StratifiedKFold(
    n_splits=int(round(1.0 / CALIB_FRACTION)),
    shuffle=True, random_state=7,
)
# Use only the first fold as the calibration holdout
train_idx_core, calib_idx = next(skf_calib.split(np.arange(len(train_mols)), strata_all))

print(f"\nCalibration split: {len(train_idx_core)} train / {len(calib_idx)} calib "
      f"({100 * len(calib_idx) / len(train_mols):.1f}% held out)")

# Convenience subsets
def subset(arr_or_list, idx):
    if isinstance(arr_or_list, np.ndarray):
        return arr_or_list[idx]
    return [arr_or_list[i] for i in idx]

core_mols    = subset(train_mols,    train_idx_core)
core_targets = subset(train_targets, train_idx_core)
core_weights = subset(train_weights, train_idx_core)
core_xd_raw  = subset(x_d_train_raw, train_idx_core)

calib_mols    = subset(train_mols,    calib_idx)
calib_targets = subset(train_targets, calib_idx)
calib_weights = subset(train_weights, calib_idx)
calib_xd_raw  = subset(x_d_train_raw, calib_idx)

# ══════════════════════════════════════════════════════════════════════════════
# Step 5 — Stratification labels for CV grid search (on core set)
# ══════════════════════════════════════════════════════════════════════════════
strata_core = pd.qcut(
    core_targets, q=N_STRATA_BINS, labels=False, duplicates="drop"
).astype(int)

print(f"\nStratification ({N_STRATA_BINS} bins, core set):")
for b in range(strata_core.max() + 1):
    mask = strata_core == b
    print(f"  Bin {b}: n={mask.sum():4d}  "
          f"pEC50 {core_targets[mask].min():.2f}–{core_targets[mask].max():.2f}")

# ══════════════════════════════════════════════════════════════════════════════
# Step 6 — 5-fold stratified CV grid search (on core set)
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
    print(f"5-fold CV — {len(param_combos)} hyperparameter combos  |  "
          f"n_descriptors={n_descriptors}")
    print(f"{'='*70}")

    skf     = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    indices = np.arange(len(core_mols))
    cv_results = []

    for combo_idx, params in enumerate(param_combos):
        print(f"\n[{combo_idx + 1}/{len(param_combos)}] {params}")
        fold_losses, fold_epochs = [], []

        for fold_num, (tr_idx, va_idx) in enumerate(skf.split(indices, strata_core)):
            set_seed(42)
            xd_scaler = StandardScaler()
            x_d_tr = xd_scaler.fit_transform(core_xd_raw[tr_idx])
            x_d_va = xd_scaler.transform(core_xd_raw[va_idx])

            val_loss, best_epoch = run_fold(
                fold_num,
                [core_mols[i] for i in tr_idx],
                core_targets[tr_idx],
                core_weights[tr_idx],
                x_d_tr,
                [core_mols[i] for i in va_idx],
                core_targets[va_idx],
                core_weights[va_idx],
                x_d_va,
                n_descriptors,
                params,
                max_epochs=CV_MAX_EPOCHS,
                patience=CV_PATIENCE,
            )
            fold_losses.append(val_loss)
            fold_epochs.append(best_epoch)
            print(f"  Fold {fold_num + 1}: val_loss={val_loss:.4f}  "
                  f"best_epoch={best_epoch}")

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
print(f"Ensemble size        : {len(ENSEMBLE_SEEDS)} seeds → {ENSEMBLE_SEEDS}")

# ══════════════════════════════════════════════════════════════════════════════
# Step 7 — Fit x_d scaler on CORE training set; apply to calib + test
# ══════════════════════════════════════════════════════════════════════════════
final_xd_scaler   = StandardScaler()
x_d_core_scaled   = final_xd_scaler.fit_transform(core_xd_raw)
x_d_calib_scaled  = final_xd_scaler.transform(calib_xd_raw)
x_d_test_scaled   = final_xd_scaler.transform(x_d_test_raw)

feat = featurizers.SimpleMoleculeMolGraphFeaturizer()

dummy_w = np.ones(len(calib_mols))
calib_dps  = make_datapoints(calib_mols, calib_targets, dummy_w, x_d_calib_scaled)
calib_dset = data.MoleculeDataset(calib_dps, feat)
calib_loader = data.build_dataloader(calib_dset, num_workers=NUM_WORKERS,
                                     shuffle=False)

dummy_w_test = np.ones(len(test_mols))
test_dps    = make_datapoints(test_mols, test_targets, dummy_w_test, x_d_test_scaled)
test_dset   = data.MoleculeDataset(test_dps, feat)
test_loader = data.build_dataloader(test_dset, num_workers=NUM_WORKERS,
                                    shuffle=False)

# ══════════════════════════════════════════════════════════════════════════════
# Step 8 — Train N ensemble members on CORE training data;
#           evaluate each on the calibration holdout
# ══════════════════════════════════════════════════════════════════════════════
ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)
all_test_preds   = []
all_calib_preds  = []
seed_calib_rmse  = {}
per_model_metrics = []

for i, seed in enumerate(ENSEMBLE_SEEDS):
    print(f"\n{'='*70}")
    print(f"Ensemble member {i + 1}/{len(ENSEMBLE_SEEDS)}  |  seed={seed}")
    print(f"{'='*70}")

    set_seed(seed)

    core_dps   = make_datapoints(core_mols, core_targets, core_weights,
                                 x_d_core_scaled)
    core_dset  = data.MoleculeDataset(core_dps, feat)
    target_scaler = core_dset.normalize_targets()

    # Apply the same scaler to the calibration normalised targets
    # (needed only for loss tracking; we always predict in original units)
    train_loader = data.build_dataloader(core_dset, num_workers=NUM_WORKERS)

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

    mpnn.eval()

    # Calibration holdout predictions → RMSE for ensemble weighting
    raw_calib = trainer.predict(mpnn, calib_loader)
    preds_calib = torch.cat(raw_calib).numpy().flatten()
    all_calib_preds.append(preds_calib)
    calib_rmse = float(np.sqrt(np.mean((calib_targets - preds_calib) ** 2)))
    seed_calib_rmse[seed] = calib_rmse
    print(f"  Calibration RMSE (seed={seed}): {calib_rmse:.4f}")

    # Test predictions
    raw_preds = trainer.predict(mpnn, test_loader)
    preds_i   = torch.cat(raw_preds).numpy().flatten()
    all_test_preds.append(preds_i)

    m = report_metrics(test_targets, preds_i, label=f"seed={seed}")
    m["seed"] = seed
    m["calib_rmse"] = calib_rmse
    per_model_metrics.append(m)

# ══════════════════════════════════════════════════════════════════════════════
# Step 9 — CV-weighted ensemble (softmax of −calib_RMSE / temperature)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Computing CV-weighted ensemble")
print(f"{'='*70}")

rmse_arr = np.array([seed_calib_rmse[s] for s in ENSEMBLE_SEEDS])
log_weights = -rmse_arr / WEIGHT_TEMPERATURE
log_weights -= log_weights.max()          # numerical stability
ens_weights  = np.exp(log_weights)
ens_weights /= ens_weights.sum()

print("\nPer-seed calibration RMSE and ensemble weight:")
for seed, rmse, w in zip(ENSEMBLE_SEEDS, rmse_arr, ens_weights):
    print(f"  seed={seed:6d}  calib_RMSE={rmse:.4f}  weight={w:.4f}")

all_preds_array = np.stack(all_test_preds, axis=0)   # (n_seeds, n_test)
ensemble_preds  = (all_preds_array * ens_weights[:, None]).sum(axis=0)

# Compare flat-mean vs weighted for transparency
flat_mean_preds = all_preds_array.mean(axis=0)
print("\nFlat-mean ensemble:")
report_metrics(test_targets, flat_mean_preds, label="flat-mean")
print("CV-weighted ensemble:")
ensemble_metrics = report_metrics(test_targets, ensemble_preds, label="CV-weighted")

# ══════════════════════════════════════════════════════════════════════════════
# Step 10 — Save outputs
# ══════════════════════════════════════════════════════════════════════════════
pred_cols = {f"pEC50_seed{s}": p for s, p in zip(ENSEMBLE_SEEDS, all_test_preds)}
df_out = pd.DataFrame({
    "Molecule Name":       test_names,
    "SMILES":              test_smiles,
    "pEC50_actual":        test_targets,
    "pEC50_ensemble":      ensemble_preds,
    "pEC50_flatmean":      flat_mean_preds,
    "residual":            test_targets - ensemble_preds,
    **pred_cols,
})
df_out.to_csv(OUTPUT_PREDS, index=False)
print(f"\nPredictions saved to {OUTPUT_PREDS}")

# Calibration predictions per seed
calib_cols = {f"pEC50_seed{s}": p
              for s, p in zip(ENSEMBLE_SEEDS, all_calib_preds)}
df_calib = pd.DataFrame({
    "Molecule Name": [train_names[i] for i in calib_idx],
    "pEC50_actual":  calib_targets,
    **calib_cols,
})
df_calib.to_csv(CALIB_PREDS_PATH, index=False)
print(f"Calibration predictions saved to {CALIB_PREDS_PATH}")

df_members = pd.DataFrame(per_model_metrics)
print("\nPer-member metrics:")
print(df_members.to_string(index=False))

m = ensemble_metrics
print(
    f"\n[Leaderboard format — CV-weighted]  "
    f"MAE={m['mae']:.4f}  RAE={m['rae']:.4f}  "
    f"R²={m['r2']:.4f}  Spearman={m['spearman']:.4f}  Kendall={m['kendall']:.4f}"
)
