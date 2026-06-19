"""
ChemProp PXR pEC50 — RDKit 2D descriptors + counter-assay weighting.
Training set: f1_plus_htchem_train_4827.sdf  (4827 compounds; includes
              HTChem crudes + semi-pure with yield-corrected pEC50)
Test set    : test_phase2_260.sdf            (260 Phase 2 compounds)

Weighting   : counter-assay only (compounds more selective for PXR vs
              counter assay receive higher weight; HTChem compounds that
              have no counter data receive neutral weight = 1.0).

Compare with chemprop_htchem_counter_plus_se_weight.py which additionally
scales weights by inverse measurement variance (1 / pEC50_std.error²).

Usage:
    conda activate chemprop
    python /home/spal/dockerimages/QSARTuna/PXR/June_19/chemprop_htchem_counter_weight.py
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
HERE = Path("/home/spal/dockerimages/QSARTuna/PXR/June_19")

TRAIN_SDF = HERE / "f1_plus_htchem_train_4827.sdf"
TEST_SDF  = HERE / "test_phase2_260.sdf"

CV_RESULTS_PATH = HERE / "htchem_counter_cv_results.csv"
ENSEMBLE_DIR    = HERE / "htchem_counter_ensemble"
OUTPUT_PREDS    = HERE / "htchem_counter_test_predictions.csv"
KEPT_DESCS_PATH = HERE / "htchem_counter_kept_descriptors.txt"

# ── SD tag names ───────────────────────────────────────────────────────────────
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
    "mp_hidden_dim":  [300, 600, 1024],
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


def load_sdf(sdf_path: Path, required_tags: list, optional_tags: list = None):
    optional_tags = optional_tags or []
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=True)
    mols, names, smiles_list = [], [], []
    rows = {t: [] for t in required_tags + optional_tags}
    skipped = 0

    for mol in suppl:
        if mol is None:
            skipped += 1
            continue
        missing = [t for t in required_tags if not mol.HasProp(t)]
        if missing:
            print(f"  Skipped {mol.GetProp('_Name') if mol.HasProp('_Name') else '?'}: "
                  f"missing {missing}")
            skipped += 1
            continue
        mols.append(mol)
        names.append(mol.GetProp("_Name") if mol.HasProp("_Name") else "")
        smiles_list.append(Chem.MolToSmiles(mol))
        for tag in required_tags + optional_tags:
            if mol.HasProp(tag):
                try:
                    rows[tag].append(float(mol.GetPropsAsDict()[tag]))
                except (ValueError, TypeError):
                    rows[tag].append(np.nan)
            else:
                rows[tag].append(np.nan)

    if skipped:
        print(f"  {skipped} molecules skipped")
    return mols, names, smiles_list, {t: np.array(v, dtype=float) for t, v in rows.items()}


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
    return np.all(np.isfinite(arr), axis=0) & (np.var(arr, axis=0) > 0)


def compute_counter_weights(counter_values: np.ndarray) -> np.ndarray:
    """Linear rescaling of counter pEC50 to [MIN_WEIGHT, MAX_WEIGHT].
    Compounds with no counter data get NEUTRAL_WEIGHT."""
    weights     = np.full(len(counter_values), NEUTRAL_WEIGHT, dtype=float)
    has_counter = ~np.isnan(counter_values)
    vals        = counter_values[has_counter]
    c_min, c_max = vals.min(), vals.max()
    if c_max > c_min:
        weights[has_counter] = MIN_WEIGHT + (MAX_WEIGHT - MIN_WEIGHT) * (
            (c_max - vals) / (c_max - c_min)
        )
    return weights


def build_mpnn(n_descriptors, ffn_hidden_dim, ffn_n_layers, dropout,
               mp_depth, mp_hidden_dim, target_scaler) -> models.MPNN:
    feat = featurizers.SimpleMoleculeMolGraphFeaturizer()
    mp   = nn.BondMessagePassing(
               d_v=feat.atom_fdim, d_e=feat.bond_fdim,
               depth=mp_depth, d_h=mp_hidden_dim)
    agg  = nn.MeanAggregation()
    ffn  = nn.RegressionFFN(
        input_dim=mp.output_dim + n_descriptors,
        hidden_dim=ffn_hidden_dim,
        n_layers=ffn_n_layers,
        dropout=dropout,
        criterion=nn.metrics.MAE(),
        output_transform=nn.UnscaleTransform.from_standard_scaler(target_scaler),
    )
    return models.MPNN(mp, agg, ffn, batch_norm=True,
                       metrics=[nn.metrics.RMSE(), nn.metrics.MAE()],
                       init_lr=INIT_LR, max_lr=MAX_LR, final_lr=FINAL_LR)


def make_datapoints(mols, targets, weights, x_d):
    return [
        data.MoleculeDatapoint(mol=mol, y=np.array([t], dtype=float),
                               weight=float(w), x_d=xd.astype(float))
        for mol, t, w, xd in zip(mols, targets, weights, x_d)
    ]


def report_metrics(actual, predicted, label=""):
    mae  = float(np.mean(np.abs(actual - predicted)))
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    r2   = float(1 - np.sum((actual - predicted)**2) / np.sum((actual - np.mean(actual))**2))
    rho, _ = stats.spearmanr(actual, predicted)
    tag = f"  [{label}]" if label else ""
    print(f"{tag}  MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}  Spearman={rho:.4f}")
    return dict(mae=mae, rmse=rmse, r2=r2, spearman=rho)


def run_fold(fold_idx, train_mols, train_targets, train_weights, train_x_d,
             val_mols, val_targets, val_weights, val_x_d,
             n_descriptors, params, max_epochs, patience):
    feat      = featurizers.SimpleMoleculeMolGraphFeaturizer()
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
        logger=False, enable_checkpointing=False, enable_progress_bar=False,
        accelerator="auto", devices=1, max_epochs=max_epochs,
        callbacks=[EarlyStopping(monitor="val_loss", patience=patience, mode="min"),
                   epoch_tracker],
    )
    trainer.fit(mpnn, train_loader, val_loader)
    return epoch_tracker.best_val_loss, epoch_tracker.best_epoch


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — Load data
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("HTChem F1: 4827 compounds (original + HTChem crudes + semi-pure)")
print("Weighting: counter-assay only")
print(f"{'='*70}")

print(f"\nLoading training SDF:\n  {TRAIN_SDF}")
train_mols, train_names, train_smiles, train_tags = load_sdf(
    TRAIN_SDF,
    required_tags=[TAG_TARGET],
    optional_tags=[TAG_COUNTER],
)
train_targets  = train_tags[TAG_TARGET]
counter_values = train_tags[TAG_COUNTER]
train_weights  = compute_counter_weights(counter_values)

n_counter = (~np.isnan(counter_values)).sum()
print(f"  Loaded {len(train_mols)} molecules")
print(f"  Counter screen available : {n_counter}/{len(train_mols)} "
      f"({100*n_counter/len(train_mols):.1f}%)")
print(f"  Weight range             : {train_weights.min():.3f} – {train_weights.max():.3f}")
print(f"  pEC50 range              : {train_targets.min():.2f} – {train_targets.max():.2f} "
      f"(mean {train_targets.mean():.2f})")

print(f"\nLoading test SDF:\n  {TEST_SDF}")
test_mols, test_names, test_smiles, _ = load_sdf(TEST_SDF, required_tags=[])
print(f"  Loaded {len(test_mols)} molecules (prediction only — no labels)")

# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — RDKit 2D descriptors
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

print("Computing RDKit 2D descriptors for test set...")
x_d_test_raw = compute_rdkit_descriptors(test_mols)[:, col_mask]

# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — Stratified CV grid search
# ══════════════════════════════════════════════════════════════════════════════
strata = pd.qcut(train_targets, q=N_STRATA_BINS, labels=False,
                 duplicates="drop").astype(int)

if CV_RESULTS_PATH.exists():
    print(f"\nCV results found at {CV_RESULTS_PATH} — skipping grid search.")
    df_cv = pd.read_csv(CV_RESULTS_PATH).sort_values("mean_val_loss").reset_index(drop=True)
else:
    param_combos = [dict(zip(PARAM_GRID.keys(), c))
                    for c in itertools.product(*PARAM_GRID.values())]
    print(f"\n5-fold CV — {len(param_combos)} combos  |  n_descriptors={n_descriptors}")

    skf     = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    indices = np.arange(len(train_mols))
    cv_results = []

    for ci, params in enumerate(param_combos):
        print(f"\n[{ci+1}/{len(param_combos)}] {params}")
        fold_losses, fold_epochs = [], []
        for fi, (tr_idx, va_idx) in enumerate(skf.split(indices, strata)):
            set_seed(42)
            xd_scaler = StandardScaler()
            x_d_tr = xd_scaler.fit_transform(x_d_train_raw[tr_idx])
            x_d_va = xd_scaler.transform(x_d_train_raw[va_idx])
            vl, be = run_fold(
                fi,
                [train_mols[i] for i in tr_idx], train_targets[tr_idx],
                train_weights[tr_idx], x_d_tr,
                [train_mols[i] for i in va_idx], train_targets[va_idx],
                train_weights[va_idx], x_d_va,
                n_descriptors, params, CV_MAX_EPOCHS, CV_PATIENCE,
            )
            fold_losses.append(vl); fold_epochs.append(be)
            print(f"  Fold {fi+1}: val_loss={vl:.4f}  best_epoch={be}")
        mean_loss = float(np.mean(fold_losses))
        print(f"  → Mean val loss: {mean_loss:.4f} ± {np.std(fold_losses):.4f}")
        cv_results.append({**params, "mean_val_loss": mean_loss,
                           "std_val_loss": float(np.std(fold_losses)),
                           "mean_best_epoch": int(np.mean(fold_epochs))})

    df_cv = pd.DataFrame(cv_results).sort_values("mean_val_loss").reset_index(drop=True)
    df_cv.to_csv(CV_RESULTS_PATH, index=False)
    print(f"\nCV results saved to {CV_RESULTS_PATH}")

print(f"\n{df_cv.to_string(index=False)}")

best_row    = df_cv.iloc[0]
best_params = {k: (int(best_row[k]) if k != "dropout" else float(best_row[k]))
               for k in PARAM_GRID}
final_epochs = max(int(best_row["mean_best_epoch"] * 1.1), 80)
print(f"\nBest params  : {best_params}")
print(f"Final epochs : {final_epochs}")

# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — Scale descriptors on full training set
# ══════════════════════════════════════════════════════════════════════════════
final_xd_scaler  = StandardScaler()
x_d_train_scaled = final_xd_scaler.fit_transform(x_d_train_raw)
x_d_test_scaled  = final_xd_scaler.transform(x_d_test_raw)

feat = featurizers.SimpleMoleculeMolGraphFeaturizer()
test_dps    = make_datapoints(test_mols, np.zeros(len(test_mols)),
                              np.ones(len(test_mols)), x_d_test_scaled)
test_dset   = data.MoleculeDataset(test_dps, feat)
test_loader = data.build_dataloader(test_dset, num_workers=NUM_WORKERS, shuffle=False)

# ══════════════════════════════════════════════════════════════════════════════
# Step 5 — Train ensemble
# ══════════════════════════════════════════════════════════════════════════════
ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)
all_test_preds = []

for i, seed in enumerate(ENSEMBLE_SEEDS):
    print(f"\n{'='*70}")
    print(f"Ensemble member {i+1}/{len(ENSEMBLE_SEEDS)}  seed={seed}")
    print(f"{'='*70}")
    set_seed(seed)

    all_dps   = make_datapoints(train_mols, train_targets, train_weights, x_d_train_scaled)
    all_dset  = data.MoleculeDataset(all_dps, feat)
    t_scaler  = all_dset.normalize_targets()
    tr_loader = data.build_dataloader(all_dset, num_workers=NUM_WORKERS)

    mpnn = build_mpnn(n_descriptors, **best_params, target_scaler=t_scaler)
    trainer = pl.Trainer(
        logger=False, enable_checkpointing=False, enable_progress_bar=True,
        accelerator="auto", devices=1, max_epochs=final_epochs,
    )
    trainer.fit(mpnn, tr_loader)
    torch.save(mpnn, ENSEMBLE_DIR / f"model_seed{seed}.pt")

    mpnn.eval()
    preds_i = torch.cat(trainer.predict(mpnn, test_loader)).numpy().flatten()
    all_test_preds.append(preds_i)
    print(f"  preds: min={preds_i.min():.3f}  max={preds_i.max():.3f}  "
          f"mean={preds_i.mean():.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# Step 6 — Ensemble prediction
# ══════════════════════════════════════════════════════════════════════════════
arr            = np.stack(all_test_preds)
ensemble_preds = arr.mean(axis=0)
ensemble_std   = arr.std(axis=0)

df_out = pd.DataFrame({
    "Molecule Name":  test_names,
    "SMILES":         test_smiles,
    "pEC50_ensemble": ensemble_preds,
    "pEC50_std":      ensemble_std,
    **{f"pEC50_seed{s}": p for s, p in zip(ENSEMBLE_SEEDS, all_test_preds)},
})
df_out.to_csv(OUTPUT_PREDS, index=False)

print(f"\n{'='*70}")
print("Ensemble complete — counter weighting only")
print(f"{'='*70}")
print(f"  n compounds : {len(ensemble_preds)}")
print(f"  pEC50 range : {ensemble_preds.min():.3f} – {ensemble_preds.max():.3f}")
print(f"  Mean std    : {ensemble_std.mean():.3f}  (ensemble uncertainty)")
print(f"  Saved       : {OUTPUT_PREDS}")
