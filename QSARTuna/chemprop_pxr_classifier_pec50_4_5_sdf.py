"""
ChemProp PXR — Binary classifier at pEC50 = 4.5 cut-off.

RDKit 2D descriptors + counter-assay weighting + multi-seed ensemble.
Mirrors chemprop_pxr_pec50_rdkit2d_counter_weight_ensemble_sdf.py but
trains a BinaryClassificationFFN to predict P(active) where:
    active   (1) = pEC50 >= 4.5
    inactive (0) = pEC50 <  4.5

Threshold rationale (from test-set analysis):
    - 4.5 gives the best MCC (0.708) and most balanced classes (69/31%)
    - Matches the existing train_binary_classifier.csv labelling convention
    - 214 test compounds fall within ±0.5 of threshold — largest rescue zone
    - AUC from regression already 0.963 here; a dedicated classifier should improve

Rescue workflow (combine with regression ensemble):
    regression >= 4.5  AND  proba_active >= 0.5  ->  confident active
    regression <  4.5  AND  proba_active <  0.5  ->  confident inactive
    regression >= 4.5  BUT  proba_active <  0.5  ->  flag: regression may over-predict
    regression <  4.5  BUT  proba_active >= 0.5  ->  flag: possible missed active
    High proba_std + borderline regression -> priority for experimental follow-up

Output CSV:
    Molecule Name, SMILES, pEC50_actual, label_actual, label_pred,
    proba_active, proba_std, proba_seed<N> (one column per ensemble member)

Usage:
    conda activate chemprop
    python ~/dockerimages/QSARTuna/chemprop_pxr_classifier_pec50_4_5_sdf.py
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
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)
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
CV_RESULTS_PATH = Path.home() / "pxr_classifier_4_5_cv_results.csv"
ENSEMBLE_DIR    = Path.home() / "pxr_classifier_4_5_ensemble_models"
OUTPUT_PREDS    = Path.home() / "pxr_classifier_4_5_test_predictions.csv"
KEPT_DESCS_PATH = Path.home() / "pxr_classifier_4_5_kept_descriptors.txt"

# ── SD tag names ───────────────────────────────────────────────────────────────
TAG_TARGET  = "pEC50"
TAG_COUNTER = "pEC50_counter"

# ── Classification threshold ───────────────────────────────────────────────────
THRESHOLD = 4.5  # pEC50 >= THRESHOLD -> active (label 1)

# ── Counter-screen weighting ───────────────────────────────────────────────────
MIN_WEIGHT     = 0.5
MAX_WEIGHT     = 2.0
NEUTRAL_WEIGHT = 1.0

# ── CV settings ────────────────────────────────────────────────────────────────
N_FOLDS       = 5
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
    Load an SDF file and return (mols, names, smiles_list, tag_arrays).
    Explicit Hs are removed after loading so the heavy-atom graph matches
    what MolFromSmiles would produce while preserving the protonation state.
    """
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
def compute_weights(counter_values: np.ndarray) -> np.ndarray:
    weights      = np.full(len(counter_values), NEUTRAL_WEIGHT, dtype=float)
    has_counter  = ~np.isnan(counter_values)
    vals         = counter_values[has_counter]
    c_min, c_max = vals.min(), vals.max()
    weights[has_counter] = MIN_WEIGHT + (MAX_WEIGHT - MIN_WEIGHT) * (
        (c_max - vals) / (c_max - c_min)
    )
    return weights


# ── MPNN builder — binary classification ──────────────────────────────────────
def build_mpnn_classifier(
    n_descriptors: int,
    ffn_hidden_dim: int,
    ffn_n_layers: int,
    dropout: float,
    mp_depth: int,
    mp_hidden_dim: int,
) -> models.MPNN:
    feat = featurizers.SimpleMoleculeMolGraphFeaturizer()
    mp   = nn.BondMessagePassing(
               d_v=feat.atom_fdim, d_e=feat.bond_fdim,
               depth=mp_depth, d_h=mp_hidden_dim,
           )
    agg  = nn.MeanAggregation()
    ffn  = nn.BinaryClassificationFFN(
        input_dim=mp.output_dim + n_descriptors,
        hidden_dim=ffn_hidden_dim,
        n_layers=ffn_n_layers,
        dropout=dropout,
    )
    return models.MPNN(
        mp, agg, ffn,
        batch_norm=True,
        metrics=[nn.metrics.BinaryAUROC(), nn.metrics.BinaryAccuracy()],
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


# ── Classification metrics ─────────────────────────────────────────────────────
def report_clf_metrics(
    actual: np.ndarray, proba: np.ndarray, label: str = ""
) -> dict:
    pred = (proba >= 0.5).astype(int)
    auc  = float(roc_auc_score(actual, proba))
    mcc  = float(matthews_corrcoef(actual, pred))
    acc  = float(accuracy_score(actual, pred))
    prec = float(precision_score(actual, pred, zero_division=0))
    rec  = float(recall_score(actual, pred, zero_division=0))
    f1   = float(f1_score(actual, pred, zero_division=0))
    tn, fp, fn, tp = confusion_matrix(actual, pred).ravel()
    tag = f"  [{label}]" if label else ""
    print(
        f"{tag}  AUC={auc:.4f}  MCC={mcc:.4f}  Acc={acc:.4f}  "
        f"Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}  "
        f"TP={tp}  TN={tn}  FP={fp}  FN={fn}"
    )
    return dict(
        auc=auc, mcc=mcc, accuracy=acc, precision=prec,
        recall=rec, f1=f1, tp=int(tp), tn=int(tn), fp=int(fp), fn=int(fn),
    )


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

    # No target normalisation for binary classification
    train_loader = data.build_dataloader(train_dset, num_workers=NUM_WORKERS)
    val_loader   = data.build_dataloader(val_dset,   num_workers=NUM_WORKERS, shuffle=False)

    mpnn          = build_mpnn_classifier(n_descriptors, **params)
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
train_mols, train_names, train_smiles, train_tags = load_sdf(
    TRAIN_SDF,
    required_tags=[TAG_TARGET],
    optional_tags=[TAG_COUNTER],
)
train_pec50    = train_tags[TAG_TARGET]
counter_values = train_tags[TAG_COUNTER]
train_weights  = compute_weights(counter_values)

train_targets = (train_pec50 >= THRESHOLD).astype(float)
n_active      = int(train_targets.sum())
n_inactive    = int((train_targets == 0).sum())

print(f"  Loaded {len(train_mols)} molecules")
print(f"  Active   (pEC50 >= {THRESHOLD}) : {n_active} ({100*n_active/len(train_mols):.1f}%)")
print(f"  Inactive (pEC50 <  {THRESHOLD}) : {n_inactive} ({100*n_inactive/len(train_mols):.1f}%)")
n_with_counter = (~np.isnan(counter_values)).sum()
print(f"  Counter screen available : {n_with_counter}/{len(train_mols)} "
      f"({100*n_with_counter/len(train_mols):.1f}%)")

# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — Load test SDF
# ══════════════════════════════════════════════════════════════════════════════
print(f"\nLoading test SDF:\n  {TEST_SDF}")
test_mols, test_names, test_smiles, test_tags = load_sdf(
    TEST_SDF,
    required_tags=[TAG_TARGET],
)
test_pec50    = test_tags[TAG_TARGET]
test_targets  = (test_pec50 >= THRESHOLD).astype(float)
n_test_active = int(test_targets.sum())
print(f"  Loaded {len(test_mols)} molecules")
print(f"  Active   : {n_test_active} ({100*n_test_active/len(test_mols):.1f}%)")
print(f"  Inactive : {len(test_mols)-n_test_active} "
      f"({100*(len(test_mols)-n_test_active)/len(test_mols):.1f}%)")

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
# Step 4 — Stratified CV grid search (stratify on binary class label)
# ══════════════════════════════════════════════════════════════════════════════
strata = train_targets.astype(int)

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
    print(f"5-fold CV — {len(param_combos)} combos  |  n_descriptors={n_descriptors}")
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
        print(f"  → Mean val loss: {mean_loss:.4f} ± {std_loss:.4f}  Mean epoch: {mean_epoch}")

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
# Step 5 — Fit x_d scaler on all training data
# ══════════════════════════════════════════════════════════════════════════════
final_xd_scaler  = StandardScaler()
x_d_train_scaled = final_xd_scaler.fit_transform(x_d_train_raw)
x_d_test_scaled  = final_xd_scaler.transform(x_d_test_raw)

feat               = featurizers.SimpleMoleculeMolGraphFeaturizer()
test_weights_dummy = np.ones(len(test_mols))
test_dps           = make_datapoints(test_mols, test_targets, test_weights_dummy, x_d_test_scaled)
test_dset          = data.MoleculeDataset(test_dps, feat)
test_loader        = data.build_dataloader(test_dset, num_workers=NUM_WORKERS, shuffle=False)

# ══════════════════════════════════════════════════════════════════════════════
# Step 6 — Train N ensemble members on ALL training data
# ══════════════════════════════════════════════════════════════════════════════
ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)
all_test_probas   = []
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
    train_loader   = data.build_dataloader(all_train_dset, num_workers=NUM_WORKERS)

    mpnn = build_mpnn_classifier(n_descriptors, **best_params)

    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
        accelerator="auto",
        devices=1,
        max_epochs=final_epochs,
    )
    trainer.fit(mpnn, train_loader)

    model_path = ENSEMBLE_DIR / f"classifier_seed{seed}.pt"
    torch.save(mpnn, model_path)
    print(f"  Saved → {model_path}")

    mpnn.eval()
    raw_preds = trainer.predict(mpnn, test_loader)
    # BinaryClassificationFFN outputs sigmoid probabilities directly
    proba_i = torch.cat(raw_preds).numpy().flatten()
    all_test_probas.append(proba_i)

    m = report_clf_metrics(test_targets, proba_i, label=f"seed={seed}")
    m["seed"] = seed
    per_model_metrics.append(m)

# ══════════════════════════════════════════════════════════════════════════════
# Step 7 — Ensemble prediction (mean probability across all seeds)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Ensemble results")
print(f"{'='*70}")

all_probas_array  = np.stack(all_test_probas, axis=0)
ensemble_proba    = all_probas_array.mean(axis=0)
ensemble_std      = all_probas_array.std(axis=0)
ensemble_pred_cls = (ensemble_proba >= 0.5).astype(int)
ensemble_metrics  = report_clf_metrics(test_targets, ensemble_proba, label="ENSEMBLE")

proba_cols = {f"proba_seed{s}": np.round(p, 4) for s, p in zip(ENSEMBLE_SEEDS, all_test_probas)}
df_out = pd.DataFrame({
    "Molecule Name": test_names,
    "SMILES":        test_smiles,
    "pEC50_actual":  test_pec50,
    "label_actual":  test_targets.astype(int),
    "label_pred":    ensemble_pred_cls,
    "proba_active":  np.round(ensemble_proba, 4),
    "proba_std":     np.round(ensemble_std, 4),
    **proba_cols,
})
df_out.to_csv(OUTPUT_PREDS, index=False)
print(f"\nPredictions saved to {OUTPUT_PREDS}")

df_members = pd.DataFrame(per_model_metrics)
print("\nPer-member test-set metrics:")
print(df_members.to_string(index=False))
print(f"\nMean individual AUC : {df_members['auc'].mean():.4f} ± {df_members['auc'].std():.4f}")
print(f"Ensemble AUC        : {ensemble_metrics['auc']:.4f}")
print(f"Ensemble MCC        : {ensemble_metrics['mcc']:.4f}")
print(f"Ensemble F1         : {ensemble_metrics['f1']:.4f}")
