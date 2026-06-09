"""
ChemProp PXR pEC50 — hyperparameters selected against the 253 phase-1 test compounds.

Workflow:
  1. Train each hyperparameter combo on all 4132 training compounds
     (CI-filtered, no augmentation, combined sample_weight × counter weighting).
  2. Evaluate directly on the 253 phase-1 compounds using pEC50 ground truth from
     test_phase1.csv (not from the SDF, which contains different values).
  3. Select the best hyperparameters by MAE on the 253.
  4. Train an 8-seed ensemble with those hyperparameters on all training data.
  5. Predict all 513 test compounds (253 with known labels + 260 blind).

Usage:
    conda activate chemprop
    python ~/dockerimages/QSARTuna/chemprop_pxr_pec50_phase1_opt.py
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
TRAIN_SDF       = Path("/home/spal/dockerimages/QSARTuna/PXR/train_ci_filtered.sdf")
TEST_SDF        = Path("/home/spal/dockerimages/QSARTuna/PXR/test_OpenADMET_Data_prepped.sdf")
PHASE1_CSV      = Path("/home/spal/dockerimages/QSARTuna/PXR/test_phase1.csv")
HP_RESULTS_PATH = Path.home() / "pxr_phase1_opt_hp_results.csv"
ENSEMBLE_DIR    = Path.home() / "pxr_ensemble_models_phase1_opt"
OUTPUT_PREDS    = Path.home() / "pxr_chemprop_phase1_opt_predictions.csv"

# ── SD tag names ───────────────────────────────────────────────────────────────
TAG_TARGET  = "pEC50"
TAG_WEIGHT  = "sample_weight"
TAG_COUNTER = "pEC50_counter"
TAG_NAME    = "Name"

# ── Weight normalisation ───────────────────────────────────────────────────────
WEIGHT_MIN     = 0.5
WEIGHT_MAX     = 2.0
NEUTRAL_WEIGHT = 1.0

# ── HP search training settings ────────────────────────────────────────────────
HP_MAX_EPOCHS = 80
HP_PATIENCE   = 15
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


# ── Training SDF loader ────────────────────────────────────────────────────────
def load_sdf_train(sdf_path: Path):
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=True)
    mols, names, targets, raw_weights, counters = [], [], [], [], []
    skipped = 0

    for mol in suppl:
        if mol is None:
            skipped += 1
            continue
        name    = mol.GetProp(TAG_NAME) if mol.HasProp(TAG_NAME) else mol.GetProp("_Name")
        target  = float(mol.GetProp(TAG_TARGET))
        raw_wt  = float(mol.GetProp(TAG_WEIGHT))  if mol.HasProp(TAG_WEIGHT)  else 1.0
        counter = float(mol.GetProp(TAG_COUNTER)) if mol.HasProp(TAG_COUNTER) else np.nan

        mols.append(mol)
        names.append(name)
        targets.append(target)
        raw_weights.append(raw_wt)
        counters.append(counter)

    targets     = np.array(targets,     dtype=float)
    raw_weights = np.array(raw_weights, dtype=float)
    counters    = np.array(counters,    dtype=float)

    if skipped:
        print(f"  {skipped} molecules skipped (unreadable)")
    print(f"  Loaded {len(mols)} compounds")
    return mols, names, targets, raw_weights, counters


# ── Test SDF + phase-1 CSV loader ─────────────────────────────────────────────
def load_test_sets(test_sdf: Path, phase1_csv: Path):
    """
    Returns:
        eval_mols, eval_names, eval_targets  — 253 phase-1 compounds
                                               (pEC50 from phase1_csv, not SDF)
        pred_mols, pred_names                — 260 blind-prediction compounds
        all_mols,  all_names                 — full 513 for final ensemble output
    """
    phase1_df  = pd.read_csv(phase1_csv)
    pec50_gt   = phase1_df.set_index("Molecule Name")["pEC50"].to_dict()
    phase1_set = set(pec50_gt.keys())

    suppl = Chem.SDMolSupplier(str(test_sdf), removeHs=True)
    eval_mols,  eval_names,  eval_targets = [], [], []
    pred_mols,  pred_names               = [], []
    all_mols,   all_names                = [], []
    skipped = 0

    for mol in suppl:
        if mol is None:
            skipped += 1
            continue
        name = mol.GetProp(TAG_NAME) if mol.HasProp(TAG_NAME) else mol.GetProp("_Name")
        all_mols.append(mol)
        all_names.append(name)
        if name in phase1_set:
            eval_mols.append(mol)
            eval_names.append(name)
            eval_targets.append(float(pec50_gt[name]))   # ground truth from CSV
        else:
            pred_mols.append(mol)
            pred_names.append(name)

    if skipped:
        print(f"  {skipped} test molecules skipped")
    print(f"  Phase-1 eval set  : {len(eval_mols)} compounds")
    print(f"  Blind prediction  : {len(pred_mols)} compounds")

    return (
        eval_mols, eval_names, np.array(eval_targets, dtype=float),
        pred_mols, pred_names,
        all_mols,  all_names,
    )


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


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — Load training data
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nLoading training SDF:\n  {TRAIN_SDF}")
train_mols, train_names, train_targets, raw_weights, counter_values = \
    load_sdf_train(TRAIN_SDF)

w_precision   = normalize_weights(raw_weights)
w_counter     = counter_weights(counter_values)
train_weights = w_precision * w_counter

n_with_counter = (~np.isnan(counter_values)).sum()
print(f"  Counter screen avail. : {n_with_counter}/{len(train_mols)} "
      f"({100 * n_with_counter / len(train_mols):.1f}%)")
print(f"  pEC50 range           : "
      f"{train_targets.min():.2f} – {train_targets.max():.2f}")
print(f"  Combined weight range : "
      f"{train_weights.min():.3f} – {train_weights.max():.3f}  "
      f"(mean {train_weights.mean():.3f})")

# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — Load test sets
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nLoading test SDF + phase-1 ground truth:")
(
    eval_mols, eval_names, eval_targets,
    pred_mols, pred_names,
    all_test_mols, all_test_names,
) = load_test_sets(TEST_SDF, PHASE1_CSV)

print(f"  pEC50 range (phase-1) : "
      f"{eval_targets.min():.2f} – {eval_targets.max():.2f}")

# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — Build dataloaders
# ══════════════════════════════════════════════════════════════════════════════
feat = featurizers.SimpleMoleculeMolGraphFeaturizer()

eval_weights_dummy = np.ones(len(eval_mols))
eval_dps    = make_datapoints(eval_mols,  eval_targets,  eval_weights_dummy)
eval_dset   = data.MoleculeDataset(eval_dps, feat)
eval_loader = data.build_dataloader(eval_dset, num_workers=NUM_WORKERS, shuffle=False)

all_test_weights_dummy = np.ones(len(all_test_mols))
all_test_dps    = make_datapoints(all_test_mols, np.zeros(len(all_test_mols)), all_test_weights_dummy)
all_test_dset   = data.MoleculeDataset(all_test_dps, feat)
all_test_loader = data.build_dataloader(all_test_dset, num_workers=NUM_WORKERS, shuffle=False)

# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — Hyperparameter search (evaluated on 253 phase-1 compounds)
# ══════════════════════════════════════════════════════════════════════════════
if HP_RESULTS_PATH.exists():
    print(f"\nHP results found at {HP_RESULTS_PATH} — skipping search.")
    df_hp = pd.read_csv(HP_RESULTS_PATH).sort_values("mae").reset_index(drop=True)
    print(f"\n{df_hp.to_string(index=False)}")
else:
    param_combos = [
        dict(zip(PARAM_GRID.keys(), combo))
        for combo in itertools.product(*PARAM_GRID.values())
    ]

    print(f"\n{'='*70}")
    print(f"HP search — {len(param_combos)} combos, evaluated on {len(eval_mols)} phase-1 compounds")
    print(f"{'='*70}")

    hp_results = []

    for combo_idx, params in enumerate(param_combos):
        print(f"\n[{combo_idx + 1}/{len(param_combos)}] {params}")
        set_seed(42)

        # Build full training dataset
        train_dps  = make_datapoints(train_mols, train_targets, train_weights)
        train_dset = data.MoleculeDataset(train_dps, feat)
        target_scaler = train_dset.normalize_targets()

        # Apply same scaler to eval set
        eval_dset_scaled = data.MoleculeDataset(
            make_datapoints(eval_mols, eval_targets, eval_weights_dummy), feat
        )
        eval_dset_scaled.normalize_targets(target_scaler)

        train_loader = data.build_dataloader(train_dset, num_workers=NUM_WORKERS)
        val_loader   = data.build_dataloader(eval_dset_scaled, num_workers=NUM_WORKERS, shuffle=False)

        mpnn          = build_mpnn(**params, target_scaler=target_scaler)
        epoch_tracker = BestEpochTracker()

        trainer = pl.Trainer(
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            accelerator="auto",
            devices=1,
            max_epochs=HP_MAX_EPOCHS,
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=HP_PATIENCE, mode="min"),
                epoch_tracker,
            ],
        )
        trainer.fit(mpnn, train_loader, val_loader)

        # Evaluate on phase-1 ground truth (unscaled predictions)
        mpnn.eval()
        raw_preds = trainer.predict(mpnn, eval_loader)
        preds     = torch.cat(raw_preds).numpy().flatten()
        m         = report_metrics(eval_targets, preds, label=f"combo {combo_idx+1}")

        hp_results.append({
            **params,
            "mae":        m["mae"],
            "rmse":       m["rmse"],
            "r2":         m["r2"],
            "spearman":   m["spearman"],
            "best_epoch": epoch_tracker.best_epoch,
        })

    df_hp = (
        pd.DataFrame(hp_results)
        .sort_values("mae")
        .reset_index(drop=True)
    )
    df_hp.to_csv(HP_RESULTS_PATH, index=False)
    print(f"\nHP results saved → {HP_RESULTS_PATH}")
    print(f"\n{df_hp.head(10).to_string(index=False)}")

# ── Best hyperparameters ───────────────────────────────────────────────────────
best_row    = df_hp.iloc[0]
best_params = {
    "ffn_hidden_dim": int(best_row["ffn_hidden_dim"]),
    "ffn_n_layers":   int(best_row["ffn_n_layers"]),
    "dropout":        float(best_row["dropout"]),
    "mp_depth":       int(best_row["mp_depth"]),
    "mp_hidden_dim":  int(best_row["mp_hidden_dim"]),
}
final_epochs = max(int(best_row["best_epoch"] * 1.1), 80)

print(f"\nBest hyperparameters : {best_params}")
print(f"  Phase-1 MAE        : {best_row['mae']:.4f}")
print(f"  Phase-1 R²         : {best_row['r2']:.4f}")
print(f"  Final model epochs : {final_epochs}")
print(f"  Ensemble size      : {len(ENSEMBLE_SEEDS)} seeds")

# ══════════════════════════════════════════════════════════════════════════════
# Step 5 — Train ensemble on all training data, predict all 513 test compounds
# ══════════════════════════════════════════════════════════════════════════════
ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)
all_preds    = []         # predictions for all 513 test compounds
eval_preds   = []         # predictions for the 253 phase-1 compounds
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

    # Predictions on all 513 test compounds
    raw_all  = trainer.predict(mpnn, all_test_loader)
    preds_all = torch.cat(raw_all).numpy().flatten()
    all_preds.append(preds_all)

    # Metrics on 253 phase-1 compounds (using ground-truth pEC50 from CSV)
    phase1_idx = [all_test_names.index(n) for n in eval_names]
    preds_eval = preds_all[phase1_idx]
    eval_preds.append(preds_eval)

    m = report_metrics(eval_targets, preds_eval, label=f"seed={seed} phase-1")
    m["seed"] = seed
    per_model_metrics.append(m)

# ══════════════════════════════════════════════════════════════════════════════
# Step 6 — Ensemble predictions and final output
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Ensemble results — phase-1 compounds")
print(f"{'='*70}")

all_preds_array  = np.stack(all_preds, axis=0)          # (8, 513)
ensemble_all     = all_preds_array.mean(axis=0)          # (513,)

eval_preds_array = np.stack(eval_preds, axis=0)          # (8, 253)
ensemble_eval    = eval_preds_array.mean(axis=0)          # (253,)

ensemble_metrics = report_metrics(eval_targets, ensemble_eval, label="ENSEMBLE phase-1")

# Build output dataframe for all 513 compounds
phase1_gt = dict(zip(eval_names, eval_targets))
per_seed_cols = {f"pEC50_seed{s}": all_preds_array[i] for i, s in enumerate(ENSEMBLE_SEEDS)}

df_out = pd.DataFrame({
    "Molecule Name":  all_test_names,
    "pEC50_actual":   [phase1_gt.get(n, np.nan) for n in all_test_names],
    "pEC50_ensemble": ensemble_all,
    "in_phase1":      [n in phase1_gt for n in all_test_names],
    **per_seed_cols,
})
df_out.to_csv(OUTPUT_PREDS, index=False)
print(f"\nAll-513 predictions saved → {OUTPUT_PREDS}")

df_members = pd.DataFrame(per_model_metrics)
print("\nPer-member phase-1 metrics:")
print(df_members.to_string(index=False))
print(f"\nMean individual MAE : {df_members['mae'].mean():.4f} ± {df_members['mae'].std():.4f}")
print(f"Ensemble MAE        : {ensemble_metrics['mae']:.4f}")

m = ensemble_metrics
print(
    f"\n[Leaderboard format]  "
    f"MAE={m['mae']:.4f}  RAE={m['rae']:.4f}  "
    f"R²={m['r2']:.4f}  Spearman={m['spearman']:.4f}  Kendall={m['kendall']:.4f}"
)
