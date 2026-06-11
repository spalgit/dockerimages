"""
ChemProp PXR pEC50 — torsion FP (chirality-aware) + counter-assay weighting
+ multi-seed ensemble.

Changes versus chemprop_pxr_pec50_rdkit2d_counter_weight_ensemble_sdf.py:

  1. RDKit 2D physicochemical descriptors are replaced entirely by a 2000-bit
     topological torsion fingerprint (includeChirality=True).  The torsion FP
     encodes the four-atom bond paths around every rotatable bond and is
     sensitive to stereochemistry at chiral centres, unlike the 2D descriptors
     it replaces.

  2. pH 7.4 protonation via dimorphite_dl is applied to every molecule
     immediately after SDF loading, replacing the MOE-prepared protonation state.

Everything else — counter-screen weighting, 5-fold stratified CV grid search,
8-seed ensemble training, output format — is unchanged from the original.

Feature vector fed to the ChemProp FFN (x_d):
    Topological torsion FP — 2000 bits, includeChirality=True
    (bits with zero variance across the training set are dropped)

Usage:
    conda activate chemprop
    python ~/dockerimages/QSARTuna/chemprop_pxr_pec50_torsion_chirality_ph74_ensemble_sdf.py
"""

import itertools
import random
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import dimorphite_dl
import numpy as np
import pandas as pd
import torch
from lightning import pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
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
CV_RESULTS_PATH  = Path.home() / "pxr_torsion_chiral_ph74_cv_results.csv"
ENSEMBLE_DIR     = Path.home() / "pxr_torsion_chiral_ph74_ensemble_models"
OUTPUT_PREDS     = Path.home() / "pxr_torsion_chiral_ph74_test_predictions.csv"
KEPT_BITS_PATH   = Path.home() / "pxr_torsion_chiral_ph74_kept_bits.txt"

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
    "mp_hidden_dim":  [300, 1024],
}

# ── Torsion fingerprint ────────────────────────────────────────────────────────
TORSION_BITS = 2000
_TORSION_GEN = rdFingerprintGenerator.GetTopologicalTorsionGenerator(
    fpSize=TORSION_BITS, includeChirality=True
)


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


# ── pH 7.4 protonation ─────────────────────────────────────────────────────────
def _protonate_mol(mol: Chem.Mol, ph: float = 7.4) -> Chem.Mol:
    """Protonate a molecule at the given pH using dimorphite_dl.

    Converts to SMILES, protonates, converts back to mol.
    Falls back to the original mol if dimorphite_dl fails or returns an
    unparseable SMILES.
    """
    smi = Chem.MolToSmiles(mol)
    try:
        variants = dimorphite_dl.protonate_smiles(
            smi, ph_min=ph, ph_max=ph, max_variants=1, validate_output=True
        )
        prot_smi = variants[0] if variants else smi
    except Exception:
        prot_smi = smi
    prot_mol = Chem.MolFromSmiles(prot_smi)
    return prot_mol if prot_mol is not None else mol


# ── SDF loader with pH 7.4 protonation ────────────────────────────────────────
def load_sdf(
    sdf_path: Path,
    required_tags: list,
    optional_tags: list = None,
    protonate_ph: float = 7.4,
):
    """Load SDF, protonate each molecule at protonate_ph, return parsed data.

    Returns
    -------
    mols         : list[Mol]  — protonated, explicit-H-removed
    names        : list[str]
    smiles_list  : list[str]  — canonical SMILES of the protonated form
    tag_arrays   : dict[str -> np.ndarray]  (float; NaN for missing tags)
    """
    optional_tags = optional_tags or []
    all_tags = required_tags + optional_tags

    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=True)
    mols, names, smiles_list = [], [], []
    rows    = {t: [] for t in all_tags}
    skipped      = 0
    n_protonated = 0

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

        name = mol.GetProp("_Name") if mol.HasProp("_Name") else ""

        # Protonate at pH 7.4
        prot_mol = _protonate_mol(mol, ph=protonate_ph)
        if Chem.MolToSmiles(prot_mol) != Chem.MolToSmiles(mol):
            n_protonated += 1

        mols.append(prot_mol)
        names.append(name)
        smiles_list.append(Chem.MolToSmiles(prot_mol))

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
    print(f"  Protonation state changed by dimorphite_dl: "
          f"{n_protonated}/{len(mols)} molecules")
    return mols, names, smiles_list, tag_arrays


# ── Torsion fingerprints ───────────────────────────────────────────────────────
def compute_torsion_fps(mols: list) -> np.ndarray:
    """Compute chirality-aware topological torsion fingerprints (TORSION_BITS bits)."""
    rows = [_TORSION_GEN.GetFingerprintAsNumPy(mol).astype(float) for mol in mols]
    return np.array(rows, dtype=float)


def select_valid_columns(arr: np.ndarray) -> np.ndarray:
    """Keep columns that are finite in all rows and have non-zero variance."""
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
# Step 1 — Load and protonate training SDF
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nLoading training SDF (protonating at pH 7.4 with dimorphite_dl):\n  {TRAIN_SDF}")
train_mols, train_names, train_smiles, train_tags = load_sdf(
    TRAIN_SDF,
    required_tags=[TAG_TARGET],
    optional_tags=[TAG_COUNTER],
    protonate_ph=7.4,
)
train_targets  = train_tags[TAG_TARGET]
counter_values = train_tags[TAG_COUNTER]
train_weights  = compute_weights(counter_values)

n_with_counter = (~np.isnan(counter_values)).sum()
print(f"  Loaded {len(train_mols)} molecules")
print(f"  Counter screen available : {n_with_counter}/{len(train_mols)} "
      f"({100 * n_with_counter / len(train_mols):.1f}%)")
print(f"  pEC50_counter range      : "
      f"{np.nanmin(counter_values):.2f} – {np.nanmax(counter_values):.2f}  "
      f"(mean {np.nanmean(counter_values):.2f})")
print(f"  Weight range             : "
      f"{train_weights.min():.3f} – {train_weights.max():.3f}  "
      f"(neutral={NEUTRAL_WEIGHT})")
print(f"  pEC50 range              : "
      f"{train_targets.min():.2f} – {train_targets.max():.2f}  "
      f"(mean {train_targets.mean():.2f})")

# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — Load and protonate test SDF
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nLoading test SDF (protonating at pH 7.4 with dimorphite_dl):\n  {TEST_SDF}")
test_mols, test_names, test_smiles, test_tags = load_sdf(
    TEST_SDF,
    required_tags=[TAG_TARGET],
    protonate_ph=7.4,
)
test_targets = test_tags[TAG_TARGET]
print(f"  Loaded {len(test_mols)} molecules")
print(f"  pEC50 range : {test_targets.min():.2f} – {test_targets.max():.2f}  "
      f"(mean {test_targets.mean():.2f})")

# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — Torsion fingerprints; column selection from training set only
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nComputing torsion fingerprints (chirality-aware, {TORSION_BITS} bits) "
      f"for training set...")
x_d_train_all = compute_torsion_fps(train_mols)

col_mask      = select_valid_columns(x_d_train_all)
kept_indices  = [i for i, k in enumerate(col_mask) if k]
print(f"  Bits kept after variance filter : {col_mask.sum()} / {TORSION_BITS}")

x_d_train_raw = x_d_train_all[:, col_mask]
n_descriptors = x_d_train_raw.shape[1]

with open(KEPT_BITS_PATH, "w") as fh:
    fh.write("\n".join(f"torsion_{i}" for i in kept_indices))
print(f"  Kept bit indices saved to {KEPT_BITS_PATH}")

print(f"\nComputing torsion fingerprints for test set...")
x_d_test_all = compute_torsion_fps(test_mols)
x_d_test_raw = x_d_test_all[:, col_mask]

# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — Stratification labels for CV
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
# Step 5 — 5-fold stratified CV grid search
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
                train_weights[tr_idx],
                x_d_tr,
                [train_mols[i] for i in va_idx],
                train_targets[va_idx],
                train_weights[va_idx],
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
# Step 6 — Fit x_d scaler on all training data
# ══════════════════════════════════════════════════════════════════════════════
final_xd_scaler  = StandardScaler()
x_d_train_scaled = final_xd_scaler.fit_transform(x_d_train_raw)
x_d_test_scaled  = final_xd_scaler.transform(x_d_test_raw)

feat = featurizers.SimpleMoleculeMolGraphFeaturizer()

test_weights_dummy = np.ones(len(test_mols))
test_dps    = make_datapoints(test_mols, test_targets, test_weights_dummy, x_d_test_scaled)
test_dset   = data.MoleculeDataset(test_dps, feat)
test_loader = data.build_dataloader(test_dset, num_workers=NUM_WORKERS, shuffle=False)

# ══════════════════════════════════════════════════════════════════════════════
# Step 7 — Train N ensemble members on ALL training data
# ══════════════════════════════════════════════════════════════════════════════
ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)
all_test_preds    = []
per_model_metrics = []

for i, seed in enumerate(ENSEMBLE_SEEDS):
    print(f"\n{'='*70}")
    print(f"Ensemble member {i + 1}/{len(ENSEMBLE_SEEDS)}  |  seed={seed}")
    print(f"{'='*70}")

    set_seed(seed)

    all_train_dps  = make_datapoints(
        train_mols, train_targets, train_weights, x_d_train_scaled
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

    mpnn.eval()
    raw_preds = trainer.predict(mpnn, test_loader)
    preds_i   = torch.cat(raw_preds).numpy().flatten()
    all_test_preds.append(preds_i)

    m = report_metrics(test_targets, preds_i, label=f"seed={seed}")
    m["seed"] = seed
    per_model_metrics.append(m)

# ══════════════════════════════════════════════════════════════════════════════
# Step 8 — Ensemble prediction (mean across all seeds)
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
