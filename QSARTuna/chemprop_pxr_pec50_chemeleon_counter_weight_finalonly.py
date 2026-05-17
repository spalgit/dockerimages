"""
ChemProp + CheMeleon PXR pEC50 — Final model on ALL training compounds.

COMPARE WITH: chemprop_pxr_pec50_rdkit2d_counter_weight_scaffold_finalonly.py
─────────────────────────────────────────────────────────────────────────────
This script is a drop-in replacement for scaffold_finalonly.py with one
fundamental upgrade: instead of training a ChemProp MPNN from random weights,
it fine-tunes the CheMeleon foundation model (pre-trained on ~100 M molecules).

Everything that is NOT listed below is intentionally identical:
  ✓ Same training CSV (all 4,140 compounds)
  ✓ Same test CSV (test.csv, 513 molecules)
  ✓ Same counter-screen weighting scheme (linear inverse, 0.5 – 2.0)
  ✓ Same loss criterion (MAE)
  ✓ Same submission output format

What IS different, and why:
  ┌─────────────────────┬──────────────────────────────┬─────────────────────────────────────┐
  │ Aspect              │ scaffold_finalonly.py         │ THIS script                         │
  ├─────────────────────┼──────────────────────────────┼─────────────────────────────────────┤
  │ Backbone            │ Random-init ChemProp MPNN     │ CheMeleon pre-trained backbone      │
  │ RDKit 2D descs      │ Yes (209 descriptors as x_d)  │ No — CheMeleon already encodes them │
  │ Epochs              │ 150                           │ 50 (pre-trained converges faster)   │
  │ LR schedule         │ 3-phase slow (1e-4→2e-4→1e-5) │ Noam with 2 warmup epochs           │
  │ Seeds               │ 1                             │ 3 (ensemble → lower variance)       │
  │ TTA                 │ No                            │ 5 random SMILES per molecule        │
  │ Framework           │ raw chemprop                  │ openadmet (ChemPropModel etc.)      │
  │ Conda env           │ chemprop                      │ openadmet-models                    │
  └─────────────────────┴──────────────────────────────┴─────────────────────────────────────┘

Why CheMeleon gives "larger effective training":
  Pre-training on ~100 M diverse molecules gives the message-passing backbone a
  molecular representation that encodes far more chemical knowledge than 4,140
  labelled PXR compounds alone can provide.  Fine-tuning then adapts this rich
  representation to the PXR task specifically.

Hyperparameters below are reasonable fine-tuning defaults; run
pxr_chemprop_chemeleon_butina.py with --n_trials 25 to optimise them.

Output:
  ~/OpenAdmet/Submission_Chemeleon_CW_FinalOnly.csv  — competition submission
  ~/OpenAdmet/chemeleon_cw_finalonly/seed_*/         — saved model weights

Usage:
    conda activate openadmet-models
    cd ~/OpenAdmet
    python ~/dockerimages/QSARTuna/chemprop_pxr_pec50_chemeleon_counter_weight_finalonly.py
"""

import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rdkit import Chem

# DIFFERENCE: openadmet framework instead of raw chemprop imports
from chemprop import nn as cp_nn
from openadmet.models.registries import *  # noqa: F401 F403
from openadmet.models.architecture.chemprop import ChemPropModel
from openadmet.models.features.chemprop import ChemPropFeaturizer
from openadmet.models.trainer.lightning import LightningTrainer

warnings.filterwarnings("ignore")

# ── Paths ── IDENTICAL to scaffold_finalonly.py ──────────────────────────────────
TRAIN_PATH = Path(
    "/home/spal/dockerimages/QSARTuna/PXR_For_QSARTuna_Modelling/"
    "processed_Openadmet_REAL_PXR_train_AND_test_main_with_side_info_"
    "AND_counter_screen_weighted.csv"
)
TEST_PATH       = Path.home() / "dockerimages/QSARTuna/PXR/test.csv"
SUBMISSION_PATH = Path.home() / "OpenAdmet/Submission_Chemeleon_CW_FinalOnly.csv"
OUT_DIR         = Path.home() / "OpenAdmet/chemeleon_cw_finalonly"

TRAIN_SMILES_COL = "SMILES"
TRAIN_TARGET_COL = "pEC50"
COUNTER_COL      = "pEC50_counter"
TEST_SMILES_COL  = "SMILES"
TEST_NAME_COL    = "Molecule Name"

# ── Counter-screen weighting ── IDENTICAL to scaffold_finalonly.py ───────────────
MIN_WEIGHT     = 0.5
MAX_WEIGHT     = 2.0
NEUTRAL_WEIGHT = 1.0

# ── CheMeleon fine-tuning hyperparameters ── DIFFERENT from scaffold_finalonly.py ─
# scaffold_finalonly uses: ffn_hidden_dim=300, ffn_n_layers=3, dropout=0.2,
#                          mp_depth=4, mp_hidden_dim=1024, epochs=150, 1 seed
# Here we use fine-tuning defaults suited to a pre-trained backbone:
FFN_HIDDEN_DIM = 512     # wider head than scratch (pre-trained features are richer)
FFN_NUM_LAYERS = 2       # shallower head is enough on top of CheMeleon
MAX_LR         = 1e-4   # lower than scratch (2e-4) — pre-trained weights are fragile
DROPOUT        = 0.1    # lighter dropout — backbone is already regularised by pre-training
BATCH_NORM     = True
WEIGHT_DECAY   = 1e-5

# DIFFERENCE: fewer epochs (pre-trained converges in ~50 vs 150 from scratch)
FINAL_EPOCHS  = 200
WARMUP_EPOCHS = 2       # noam warmup — not needed for 3-phase LR in scratch script

# DIFFERENCE: ensemble + TTA — not present in scaffold_finalonly.py
FINAL_SEEDS = [0, 1, 2]  # train 3 models and average → lower prediction variance
N_TTA       = 5           # random SMILES augmentations per molecule at inference


# ── Helpers ──────────────────────────────────────────────────────────────────────

def compute_weights(counter_values: np.ndarray) -> np.ndarray:
    """Inverse linear map of pEC50_counter → [MIN_WEIGHT, MAX_WEIGHT].
    IDENTICAL to scaffold_finalonly.py — same weighting scheme."""
    weights     = np.full(len(counter_values), NEUTRAL_WEIGHT, dtype=float)
    has_counter = ~np.isnan(counter_values)
    vals        = counter_values[has_counter]
    c_min, c_max = vals.min(), vals.max()
    weights[has_counter] = MIN_WEIGHT + (MAX_WEIGHT - MIN_WEIGHT) * (
        (c_max - vals) / (c_max - c_min)
    )
    return weights


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def randomize_smiles(smi: str, n: int = N_TTA, seed: int = 0) -> list[str]:
    """Return n SMILES for the same molecule (canonical + atom-renumbered variants).
    DIFFERENCE: not present in scaffold_finalonly.py — TTA reduces prediction noise."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return [smi] * n
    rng = random.Random(seed)
    result = [Chem.MolToSmiles(mol)]
    for _ in range(n - 1):
        order = list(range(mol.GetNumAtoms()))
        rng.shuffle(order)
        result.append(Chem.MolToSmiles(Chem.RenumberAtoms(mol, order), canonical=False))
    return result


def build_and_train(
    X: pd.Series,
    y: pd.DataFrame,
    weights: np.ndarray,
    seed: int,
    out_dir: Path,
) -> ChemPropModel:
    """Fine-tune CheMeleon on all training compounds for one seed.
    DIFFERENCE: uses ChemPropModel(from_chemeleon=True) + LightningTrainer
    instead of build_mpnn() + pl.Trainer in scaffold_finalonly.py."""
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seeds(seed)

    # DIFFERENCE: ChemPropFeaturizer replaces compute_rdkit_descriptors() +
    # StandardScaler + make_datapoints(); no x_d needed — CheMeleon handles
    # molecular encoding internally.
    feat = ChemPropFeaturizer()
    train_dl, _, scaler, _ = feat.featurize(X, y, weights=weights)

    # DIFFERENCE: from_chemeleon=True loads the pre-trained backbone weights.
    # All other ChemPropModel kwargs map 1-to-1 to the build_mpnn() params
    # in scaffold_finalonly.py.
    model = ChemPropModel(
        n_tasks=1,
        from_chemeleon=True,        # ← the key upgrade
        ffn_hidden_dim=FFN_HIDDEN_DIM,
        ffn_num_layers=FFN_NUM_LAYERS,
        max_lr=MAX_LR,
        dropout=DROPOUT,
        batch_norm=BATCH_NORM,
        weight_decay=WEIGHT_DECAY,
        metric_list=["mae", "rmse"],
        monitor_metric="val_loss",
        scheduler="noam",           # DIFFERENCE: noam vs 3-phase in scaffold_finalonly
        warmup_epochs=WARMUP_EPOCHS,
    )
    model.build(scaler=scaler)

    # Align criterion to MAE — IDENTICAL intent to scaffold_finalonly.py
    # (criterion=nn.metrics.MAE() in build_mpnn), just set after build() here.
    model.estimator.predictor.criterion = cp_nn.metrics.MAE()

    # DIFFERENCE: LightningTrainer replaces pl.Trainer; early_stopping=False
    # because 100% of training data is used (same reason as scaffold_finalonly).
    trainer = LightningTrainer(
        max_epochs=FINAL_EPOCHS,
        accelerator="gpu",
        early_stopping=False,
        output_dir=out_dir,
        use_wandb=False,
    )
    trainer.model = model

    try:
        trainer.build(no_val=True)
        model = trainer.train(train_dl)
    except TypeError:
        # Older openadmet-models API: pass train as both train and val loader
        trainer.build(no_val=False)
        model = trainer.train(train_dl, train_dl)

    return model


def predict_with_tta(
    models: list[ChemPropModel],
    smiles_list: list[str],
) -> np.ndarray:
    """Average predictions across seeds and SMILES augmentations.
    DIFFERENCE: scaffold_finalonly.py uses a single forward pass with no TTA."""
    n_mols     = len(smiles_list)
    seed_preds = []

    for model in models:
        aug_smiles, mol_map = [], []
        for i, smi in enumerate(smiles_list):
            variants = randomize_smiles(smi, n=N_TTA, seed=i)
            aug_smiles.extend(variants)
            mol_map.extend([i] * N_TTA)

        feat       = ChemPropFeaturizer()
        aug_series = pd.Series(aug_smiles).reset_index(drop=True)
        dl, _, _, _ = feat.featurize(aug_series)
        raw        = model.predict(dl, accelerator="gpu").flatten()

        mol_map  = np.array(mol_map)
        mol_pred = np.array([raw[mol_map == i].mean() for i in range(n_mols)])
        seed_preds.append(mol_pred)

    return np.mean(seed_preds, axis=0)


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — Load training data and compute counter-screen weights
# IDENTICAL to scaffold_finalonly.py Step 1
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nLoading training data:\n  {TRAIN_PATH}")
df_train = pd.read_csv(TRAIN_PATH)
print(f"  {len(df_train)} rows")

df_train = (df_train[[TRAIN_SMILES_COL, TRAIN_TARGET_COL, COUNTER_COL]]
            .dropna(subset=[TRAIN_SMILES_COL, TRAIN_TARGET_COL])
            .reset_index(drop=True))

weights_all    = compute_weights(df_train[COUNTER_COL].values)
n_with_counter = (~np.isnan(df_train[COUNTER_COL].values)).sum()
print(f"  Counter screen available : {n_with_counter}/{len(df_train)} "
      f"({100*n_with_counter/len(df_train):.1f}%)")
print(f"  Weight range             : {weights_all.min():.3f} – {weights_all.max():.3f}")

X_train = df_train[TRAIN_SMILES_COL]
y_train = df_train[[TRAIN_TARGET_COL]]
print(f"  Usable training molecules: {len(X_train)}")

# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — Load test set
# IDENTICAL to scaffold_finalonly.py Step 2
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nLoading test set:\n  {TEST_PATH}")
df_test     = pd.read_csv(TEST_PATH)
test_names  = df_test[TEST_NAME_COL].tolist()
test_smiles = df_test[TEST_SMILES_COL].tolist()
print(f"  Test molecules: {len(test_smiles)}")

# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — Fine-tune CheMeleon on ALL training compounds (3 seeds)
# DIFFERENCE vs scaffold_finalonly.py Step 5:
#   • no RDKit descriptor computation or StandardScaler
#   • 3 seeds instead of 1
#   • ChemPropFeaturizer + LightningTrainer instead of pl.Trainer
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*62}")
print(f"Fine-tuning CheMeleon on ALL {len(X_train)} compounds")
print(f"  ffn_hidden_dim={FFN_HIDDEN_DIM}  ffn_num_layers={FFN_NUM_LAYERS}")
print(f"  max_lr={MAX_LR}  dropout={DROPOUT}  weight_decay={WEIGHT_DECAY}")
print(f"  epochs={FINAL_EPOCHS}  warmup={WARMUP_EPOCHS}  batch_norm={BATCH_NORM}")
print(f"  seeds={FINAL_SEEDS}  TTA={N_TTA}× per molecule at inference")
print(f"{'='*62}")

final_models = []
for seed in FINAL_SEEDS:
    print(f"\n  Seed {seed} ...")
    model = build_and_train(
        X_train, y_train, weights_all,
        seed=seed,
        out_dir=OUT_DIR / f"seed_{seed}",
    )
    final_models.append(model)
    print(f"  Seed {seed} done.")

# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — Predict on test set with TTA and write submission
# DIFFERENCE vs scaffold_finalonly.py Step 7:
#   • 3-seed ensemble instead of 1 model
#   • 5× SMILES augmentation per molecule (TTA)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nPredicting {len(test_smiles)} test molecules "
      f"({len(FINAL_SEEDS)} seeds × {N_TTA} TTA augmentations)...")

preds = predict_with_tta(final_models, test_smiles)

df_submission = pd.DataFrame({
    "Molecule Name": test_names,
    "SMILES":        test_smiles,
    "pEC50":         preds,
})
SUBMISSION_PATH.parent.mkdir(parents=True, exist_ok=True)
df_submission.to_csv(SUBMISSION_PATH, index=False)

print(f"\nSubmission saved to {SUBMISSION_PATH}")
print(f"\nFirst 5 predictions:")
print(df_submission.head().to_string(index=False))
print(f"\npEC50 summary:")
print(df_submission["pEC50"].describe().to_string())
