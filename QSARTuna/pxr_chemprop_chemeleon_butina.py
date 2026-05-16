#!/usr/bin/env python3
"""
pxr_chemprop_chemeleon_butina.py
─────────────────────────────────────────────────────────────────
ChemProp + CheMeleon PXR pEC50 — Butina CV, MAE loss, 100% final training.

Design (matches Jeremy's benchmark, adapted to your preferences):
  • 3 × 5-fold Butina clustering CV for OOF scoring — scaffold-diverse splits
    that simulate blind-test difficulty (fixes the random-split HPO over-optimism)
  • HPO via Optuna using the first --n_hpo_folds Butina folds per trial
    (fast but scaffold-aware; default 2 folds)
  • MAE as both training criterion AND val_loss — no MSE/MAE mismatch
  • Counter-screen exponential down-weighting, zero data removed
  • Final model: 100% training data, 3 seeds averaged, fixed epoch count (no
    early-stopping on full data — avoids holding out any compound)
  • SMILES test-time augmentation (5 random SMILES per molecule, averaged)

Usage:
    conda activate openadmet
    cd ~/OpenAdmet
    python ~/dockerimages/QSARTuna/pxr_chemprop_chemeleon_butina.py

    # Skip HPO (use --fixed_params_json) to go straight to OOF + final model:
    python ~/dockerimages/QSARTuna/pxr_chemprop_chemeleon_butina.py \\
        --skip_hpo \\
        --fixed_params_json '{"ffn_hidden_dim":512,"ffn_num_layers":2,"max_lr":0.001,"dropout":0.1,"batch_norm":true,"weight_decay":1e-5}'

CheMeleon citation: DOI 10.48550/arXiv.2506.15792
"""

import argparse
import json
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.ML.Cluster import Butina

from chemprop import nn as cp_nn

from openadmet.models.registries import *  # noqa: F401 F403
from openadmet.models.architecture.chemprop import ChemPropModel
from openadmet.models.features.chemprop import ChemPropFeaturizer
from openadmet.models.trainer.lightning import LightningTrainer

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_CSV = (
    "/home/spal/dockerimages/QSARTuna/PXR_For_QSARTuna_Modelling/"
    "processed_Openadmet_REAL_PXR_train_AND_test_main_with_side_info_AND_counter_screen.csv"
)
EXTERNAL_TEST_CSV = "/home/spal/dockerimages/QSARTuna/PXR/test.csv"

SMILES_COL  = "SMILES"
TARGET_COL  = "pEC50"
COUNTER_COL = "pEC50_counter"
ID_COL      = "ID"            # training set ID column
EXT_ID_COL  = "Molecule Name" # test set ID column

# ── CV / training budget ───────────────────────────────────────────────────────
N_FOLDS       = 5
N_REPEATS     = 3
CV_SEEDS      = [0, 1, 2]
BUTINA_CUTOFF = 0.4   # Tanimoto distance threshold (matches Jeremy)

HPO_MAX_EPOCHS  = 50
HPO_ES_PATIENCE = 12

OOF_MAX_EPOCHS  = 70  # longer to ensure convergence; early stopping finds best epoch
OOF_ES_PATIENCE = 15

FINAL_EPOCHS = 50     # fixed epoch count for 100% final training (Jeremy's Sub2 value)
FINAL_SEEDS  = [0, 1, 2]
N_TTA        = 5      # random SMILES per molecule at inference

# ── HPO search space ───────────────────────────────────────────────────────────
FFN_HIDDEN_DIM_OPTS  = [128, 256, 300, 512, 1024]
FFN_NUM_LAYERS_RANGE = (1, 4)
MAX_LR_RANGE         = (5e-5, 5e-3)
DROPOUT_RANGE        = (0.0, 0.4)
WEIGHT_DECAY_RANGE   = (1e-6, 1e-2)
STUDY_NAME           = "pxr_chemeleon_butina_hpo"


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def compute_sample_weights(df: pd.DataFrame, scale: float) -> np.ndarray:
    """Exponential down-weight for compounds active in the counter screen.

    weight = exp(-scale * max(0, pEC50_counter - pEC50))
    Compounds where counter screen pEC50 > primary pEC50 are suspect aggregators/
    non-selective binders; we reduce their influence without discarding them.
    """
    delta = (df[COUNTER_COL] - df[TARGET_COL]).clip(lower=0.0).fillna(0.0)
    return np.exp(-scale * delta.to_numpy(dtype=float))


def load_data(csv_path: str, counter_weight_scale: float):
    df = pd.read_csv(csv_path)
    df = (df[[ID_COL, SMILES_COL, TARGET_COL, COUNTER_COL]]
          .dropna(subset=[SMILES_COL, TARGET_COL])
          .reset_index(drop=True))
    weights = compute_sample_weights(df, counter_weight_scale)
    n_down = int((weights < 1.0).sum())
    print(f"  {len(df)} compounds loaded | {n_down} down-weighted by counter screen "
          f"(scale={counter_weight_scale})")
    return df[SMILES_COL], df[[TARGET_COL]], df[ID_COL], weights


# ─────────────────────────────────────────────────────────────────────────────
# Butina clustering CV
# ─────────────────────────────────────────────────────────────────────────────

def _morgan_fps(smiles_list: list[str]):
    fps, orig_idx = [], []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048))
            orig_idx.append(i)
    return fps, orig_idx


def butina_kfold_splits(
    smiles_list: list[str], k: int = 5, seed: int = 0, cutoff: float = 0.4
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return k (train_idx, val_idx) pairs via Butina cluster stratification.

    Compounds in the same cluster are always assigned to the same fold,
    so train and val sets are maximally scaffold-diverse — matching the
    diversity of the blind test set.
    """
    n = len(smiles_list)
    fps, orig_idx = _morgan_fps(smiles_list)
    n_valid = len(fps)

    # O(n²) pairwise Tanimoto distances
    dists = []
    for i in range(1, n_valid):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend([1.0 - s for s in sims])

    clusters = list(Butina.ClusterData(dists, n_valid, cutoff, isDistData=True))

    rng = np.random.RandomState(seed)
    order = np.arange(len(clusters))
    rng.shuffle(order)
    clusters = [clusters[i] for i in order]

    # Assign local (valid-molecule) indices → fold number
    local_fold = np.zeros(n_valid, dtype=int)
    for fold_i, cluster in enumerate(clusters):
        for local_i in cluster:
            local_fold[local_i] = fold_i % k

    # Map back to original indices
    fold_of = np.zeros(n, dtype=int)
    for local_i, orig_i in enumerate(orig_idx):
        fold_of[orig_i] = local_fold[local_i]

    splits = []
    for fold in range(k):
        val_idx   = np.where(fold_of == fold)[0]
        train_idx = np.where(fold_of != fold)[0]
        splits.append((train_idx, val_idx))
    return splits


def repeated_butina_splits(
    smiles_list: list[str],
    k: int = N_FOLDS,
    seeds: list[int] = CV_SEEDS,
    cutoff: float = BUTINA_CUTOFF,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """N_REPEATS × k Butina folds."""
    all_splits = []
    for seed in seeds:
        all_splits.extend(butina_kfold_splits(smiles_list, k=k, seed=seed, cutoff=cutoff))
    return all_splits


# ─────────────────────────────────────────────────────────────────────────────
# SMILES augmentation (TTA)
# ─────────────────────────────────────────────────────────────────────────────

def randomize_smiles(smi: str, n: int = 5, seed: int = 0) -> list[str]:
    """Return n SMILES for the same molecule: canonical + (n-1) atom-renumbered."""
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


# ─────────────────────────────────────────────────────────────────────────────
# Core train + evaluate (one fold)
# ─────────────────────────────────────────────────────────────────────────────

def _set_seeds(seed: int) -> None:
    import torch
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_and_eval(
    X_tr: pd.Series,
    y_tr: pd.DataFrame,
    X_val: pd.Series,
    y_val: pd.DataFrame,
    *,
    weights_tr: np.ndarray | None,
    ffn_hidden_dim: int,
    ffn_num_layers: int,
    max_lr: float,
    dropout: float,
    batch_norm: bool,
    weight_decay: float,
    max_epochs: int,
    es_patience: int,
    output_dir: Path,
    seed: int = 42,
) -> tuple[float, float, np.ndarray, ChemPropModel]:
    """Train one CV fold. Returns (val_MAE, val_Spearman, val_preds, model)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _set_seeds(seed)

    feat = ChemPropFeaturizer()
    train_dl, _, scaler, _ = feat.featurize(X_tr, y_tr, weights=weights_tr)
    val_dl, _, _, _        = feat.featurize(X_val, y_val)

    model = ChemPropModel(
        n_tasks=1,
        from_chemeleon=True,
        ffn_hidden_dim=ffn_hidden_dim,
        ffn_num_layers=ffn_num_layers,
        max_lr=max_lr,
        dropout=dropout,
        batch_norm=batch_norm,
        weight_decay=weight_decay,
        metric_list=["mae", "rmse"],
        monitor_metric="val_loss",
        scheduler="noam",
        warmup_epochs=2,
    )
    model.build(scaler=scaler)

    # Align training criterion and val_loss to MAE.
    # Patch must happen after build() but before trainer.build() so that
    # val_loss (used for early stopping) also uses MAE.
    model.estimator.predictor.criterion = cp_nn.metrics.MAE()

    trainer = LightningTrainer(
        max_epochs=max_epochs,
        accelerator="gpu",
        early_stopping=(es_patience > 0),
        early_stopping_patience=es_patience,
        early_stopping_mode="min",
        early_stopping_min_delta=0.0005,
        output_dir=output_dir,
        use_wandb=False,
    )
    trainer.model = model
    trainer.build(no_val=False)
    model = trainer.train(train_dl, val_dl)

    y_true = y_val[TARGET_COL].to_numpy()
    preds  = model.predict(val_dl, accelerator="gpu").flatten()
    mae    = float(mean_absolute_error(y_true, preds))
    rho, _ = spearmanr(y_true, preds)

    return mae, float(rho), preds, model


# ─────────────────────────────────────────────────────────────────────────────
# Optuna HPO (scaffold-aware: uses first n_hpo_folds Butina folds)
# ─────────────────────────────────────────────────────────────────────────────

def make_objective(
    X: pd.Series,
    y: pd.DataFrame,
    weights: np.ndarray,
    hpo_splits: list[tuple[np.ndarray, np.ndarray]],
    out_base: Path,
):
    def objective(trial: optuna.Trial) -> float:
        ffn_hidden_dim = trial.suggest_categorical("ffn_hidden_dim", FFN_HIDDEN_DIM_OPTS)
        ffn_num_layers = trial.suggest_int("ffn_num_layers", *FFN_NUM_LAYERS_RANGE)
        max_lr         = trial.suggest_float("max_lr", *MAX_LR_RANGE, log=True)
        dropout        = trial.suggest_float("dropout", *DROPOUT_RANGE)
        batch_norm     = trial.suggest_categorical("batch_norm", [True, False])
        weight_decay   = trial.suggest_float("weight_decay", *WEIGHT_DECAY_RANGE, log=True)

        fold_maes = []
        for fold_i, (tr_idx, val_idx) in enumerate(hpo_splits):
            X_tr  = X.iloc[tr_idx].reset_index(drop=True)
            y_tr  = y.iloc[tr_idx].reset_index(drop=True)
            X_val = X.iloc[val_idx].reset_index(drop=True)
            y_val = y.iloc[val_idx].reset_index(drop=True)
            w_tr  = weights[tr_idx]

            fold_dir = out_base / f"trial_{trial.number:03d}" / f"fold_{fold_i}"
            try:
                mae, _, _, _ = train_and_eval(
                    X_tr, y_tr, X_val, y_val,
                    weights_tr=w_tr,
                    ffn_hidden_dim=ffn_hidden_dim,
                    ffn_num_layers=ffn_num_layers,
                    max_lr=max_lr,
                    dropout=dropout,
                    batch_norm=batch_norm,
                    weight_decay=weight_decay,
                    max_epochs=HPO_MAX_EPOCHS,
                    es_patience=HPO_ES_PATIENCE,
                    output_dir=fold_dir,
                    seed=trial.number * 10 + fold_i,
                )
                fold_maes.append(mae)
            except Exception as e:
                print(f"  [Trial {trial.number} fold {fold_i}] FAILED: {e}")
                raise optuna.exceptions.TrialPruned()

        mean_mae = float(np.mean(fold_maes))
        print(
            f"  [Trial {trial.number:03d}] MAE={mean_mae:.4f} "
            f"({[round(m,4) for m in fold_maes]}) | "
            + " | ".join(f"{k}={v}" for k, v in trial.params.items())
        )
        return mean_mae

    return objective


# ─────────────────────────────────────────────────────────────────────────────
# OOF evaluation (full 3 × 5-fold Butina CV)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_oof(
    X: pd.Series,
    y: pd.DataFrame,
    weights: np.ndarray,
    all_splits: list[tuple[np.ndarray, np.ndarray]],
    best_params: dict,
    out_base: Path,
) -> tuple[np.ndarray, float, float]:
    """
    Run 3×5-fold Butina CV with best_params. OOF predictions are averaged over
    the three repeats for each compound.
    Returns (oof_preds, oof_MAE, oof_Spearman).
    """
    n = len(X)
    oof_sum = np.zeros(n)
    oof_cnt = np.zeros(n, dtype=int)

    n_splits = len(all_splits)
    n_rep    = n_splits // N_FOLDS

    for split_i, (tr_idx, val_idx) in enumerate(all_splits):
        rep  = split_i // N_FOLDS
        fold = split_i %  N_FOLDS
        print(f"  OOF repeat {rep+1}/{n_rep}, fold {fold+1}/{N_FOLDS} "
              f"| train={len(tr_idx)}, val={len(val_idx)}")

        X_tr  = X.iloc[tr_idx].reset_index(drop=True)
        y_tr  = y.iloc[tr_idx].reset_index(drop=True)
        X_val = X.iloc[val_idx].reset_index(drop=True)
        y_val = y.iloc[val_idx].reset_index(drop=True)
        w_tr  = weights[tr_idx]

        fold_dir = out_base / "oof" / f"rep{rep}_fold{fold}"
        mae, rho, preds, _ = train_and_eval(
            X_tr, y_tr, X_val, y_val,
            weights_tr=w_tr,
            **best_params,
            max_epochs=OOF_MAX_EPOCHS,
            es_patience=OOF_ES_PATIENCE,
            output_dir=fold_dir,
            seed=split_i,
        )
        print(f"    → MAE={mae:.4f}  Spearman={rho:.4f}")

        oof_sum[val_idx] += preds
        oof_cnt[val_idx] += 1

    oof_preds = oof_sum / np.maximum(oof_cnt, 1)
    y_arr = y[TARGET_COL].to_numpy()
    oof_mae = float(mean_absolute_error(y_arr, oof_preds))
    oof_rho, _ = spearmanr(y_arr, oof_preds)
    return oof_preds, oof_mae, float(oof_rho)


# ─────────────────────────────────────────────────────────────────────────────
# Final model — 100% training data, multi-seed, fixed epochs, no early stopping
# ─────────────────────────────────────────────────────────────────────────────

def train_final_models(
    X: pd.Series,
    y: pd.DataFrame,
    weights: np.ndarray,
    best_params: dict,
    final_epochs: int,
    seeds: list[int],
    out_base: Path,
) -> list[ChemPropModel]:
    """Train one model per seed on 100% of data. Returns list of trained models."""
    models = []

    for seed in seeds:
        print(f"\n  Seed {seed} | {final_epochs} epochs | {len(X)} compounds (100% of data)")
        seed_dir = out_base / "final_models" / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        _set_seeds(seed)

        feat = ChemPropFeaturizer()
        train_dl, _, scaler, _ = feat.featurize(X, y, weights=weights)

        model = ChemPropModel(
            n_tasks=1,
            from_chemeleon=True,
            ffn_hidden_dim=best_params["ffn_hidden_dim"],
            ffn_num_layers=best_params["ffn_num_layers"],
            max_lr=best_params["max_lr"],
            dropout=best_params["dropout"],
            batch_norm=best_params["batch_norm"],
            weight_decay=best_params["weight_decay"],
            metric_list=["mae", "rmse"],
            monitor_metric="val_loss",
            scheduler="noam",
            warmup_epochs=2,
        )
        model.build(scaler=scaler)
        model.estimator.predictor.criterion = cp_nn.metrics.MAE()

        trainer = LightningTrainer(
            max_epochs=final_epochs,
            accelerator="gpu",
            early_stopping=False,  # 100% data — no held-out val for stopping signal
            output_dir=seed_dir,
            use_wandb=False,
        )
        trainer.model = model

        # Try no_val=True (cleanest path). Fall back to passing train as val
        # with early_stopping=False if the API doesn't support no_val.
        try:
            trainer.build(no_val=True)
            model = trainer.train(train_dl)
        except TypeError:
            trainer.build(no_val=False)
            model = trainer.train(train_dl, train_dl)

        print(f"  Seed {seed} done.")
        models.append(model)

    return models


# ─────────────────────────────────────────────────────────────────────────────
# TTA inference
# ─────────────────────────────────────────────────────────────────────────────

def predict_with_tta(
    models: list[ChemPropModel],
    smiles_series: pd.Series,
    n_aug: int = N_TTA,
    accelerator: str = "gpu",
) -> np.ndarray:
    """Average predictions across seeds and SMILES augmentations."""
    n_mols = len(smiles_series)
    seed_preds = []

    for model in models:
        aug_smiles, mol_map = [], []
        for i, smi in enumerate(smiles_series):
            variants = randomize_smiles(smi, n=n_aug, seed=i)
            aug_smiles.extend(variants)
            mol_map.extend([i] * n_aug)

        aug_series = pd.Series(aug_smiles).reset_index(drop=True)
        feat = ChemPropFeaturizer()
        dl, _, _, _ = feat.featurize(aug_series)
        raw = model.predict(dl, accelerator=accelerator).flatten()

        mol_map = np.array(mol_map)
        mol_preds = np.array([raw[mol_map == i].mean() for i in range(n_mols)])
        seed_preds.append(mol_preds)

    return np.mean(seed_preds, axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "ChemProp + CheMeleon PXR — Butina CV, MAE loss, 100% final training"
        )
    )
    parser.add_argument("--n_trials",    type=int,   default=25,
                        help="Optuna HPO trials (default 25)")
    parser.add_argument("--n_hpo_folds", type=int,   default=2,
                        help="Butina folds used per HPO trial (default 2; "
                             "increase for more robust HPO at the cost of runtime)")
    parser.add_argument("--out_dir",                 default="results_butina_chemeleon")
    parser.add_argument("--data_csv",                default=DATA_CSV)
    parser.add_argument("--external_test_csv",       default=EXTERNAL_TEST_CSV)
    parser.add_argument("--counter_weight_scale",
                        type=float, default=1.0,
                        help="Exponential penalty scale for counter-screen hits. "
                             "0 disables weighting (all weights=1).")
    parser.add_argument("--final_epochs", type=int,  default=FINAL_EPOCHS,
                        help=f"Epochs for 100%% final model training (default {FINAL_EPOCHS})")
    parser.add_argument("--skip_hpo", action="store_true",
                        help="Skip HPO; requires --fixed_params_json")
    parser.add_argument("--fixed_params_json", default=None,
                        help='JSON hyperparameter dict, e.g. \'{"ffn_hidden_dim":512,...}\'')
    parser.add_argument("--skip_oof", action="store_true",
                        help="Skip OOF evaluation (saves time; goes straight to final training)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Master random seed for Butina shuffling and Optuna sampler")
    args = parser.parse_args()

    out_base = Path(args.out_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    print("\n=== Loading data ===")
    X, y, ids, weights = load_data(args.data_csv, args.counter_weight_scale)

    # ── Pre-compute Butina splits (O(n²), done once and reused) ───────────────
    print(f"\n=== Computing Butina splits on {len(X)} compounds (O(n²), ~30s) ===")
    smiles_list = X.tolist()

    # Seed-0 splits are used for HPO; first n_hpo_folds folds only
    hpo_base = butina_kfold_splits(smiles_list, k=N_FOLDS, seed=args.seed, cutoff=BUTINA_CUTOFF)
    hpo_splits = hpo_base[: args.n_hpo_folds]
    print(f"  HPO using {args.n_hpo_folds}/{N_FOLDS} folds from seed-{args.seed} split "
          f"| val sizes: {[len(v) for _, v in hpo_splits]}")

    # Full 3×5 splits for OOF (computed lazily only if needed)
    oof_splits = None
    if not args.skip_oof:
        print(f"  Computing {N_REPEATS}×{N_FOLDS} OOF splits (seeds {CV_SEEDS})...")
        oof_splits = repeated_butina_splits(smiles_list)
        print(f"  {len(oof_splits)} OOF folds ready.")

    # ── HPO ───────────────────────────────────────────────────────────────────
    if args.skip_hpo:
        if not args.fixed_params_json:
            parser.error("--skip_hpo requires --fixed_params_json")
        best_params = json.loads(args.fixed_params_json)
        print(f"\n=== HPO skipped — using fixed params ===")
        for k, v in best_params.items():
            print(f"  {k:25s} = {v}")
    else:
        print(f"\n=== HPO: {args.n_trials} trials × {args.n_hpo_folds} Butina folds ===")
        study = optuna.create_study(
            direction="minimize",
            study_name=STUDY_NAME,
            storage=f"sqlite:///{out_base}/optuna.db",
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=args.seed),
        )
        study.optimize(
            make_objective(X, y, weights, hpo_splits, out_base),
            n_trials=args.n_trials,
            gc_after_trial=True,
        )
        best = study.best_trial
        best_params = {k: best.params[k] for k in [
            "ffn_hidden_dim", "ffn_num_layers", "max_lr",
            "dropout", "batch_norm", "weight_decay",
        ]}
        print(f"\nBest trial #{best.number}  HPO val_MAE={best.value:.4f}")
        for k, v in best_params.items():
            print(f"  {k:25s} = {v}")

    # ── OOF evaluation ────────────────────────────────────────────────────────
    oof_mae = oof_rho = None
    if not args.skip_oof:
        print(f"\n=== OOF evaluation: {N_REPEATS}×{N_FOLDS}-fold Butina CV ===")
        oof_preds, oof_mae, oof_rho = evaluate_oof(
            X, y, weights, oof_splits, best_params, out_base
        )
        print(f"\n  OOF MAE      : {oof_mae:.4f}")
        print(f"  OOF Spearman : {oof_rho:.4f}")

        oof_df = pd.DataFrame({
            ID_COL:      ids.values,
            SMILES_COL:  X.values,
            TARGET_COL:  y[TARGET_COL].values,
            "oof_pred":  oof_preds,
            "residual":  y[TARGET_COL].values - oof_preds,
        })
        oof_df.to_csv(out_base / "oof_predictions.csv", index=False)
        print(f"  OOF predictions → {out_base}/oof_predictions.csv")
    else:
        print("\n=== OOF evaluation skipped ===")

    # ── Final model: 100% of training data ────────────────────────────────────
    print(f"\n=== Final model: {len(FINAL_SEEDS)} seeds × {args.final_epochs} epochs "
          f"on 100% of {len(X)} compounds ===")
    final_models = train_final_models(
        X, y, weights, best_params,
        final_epochs=args.final_epochs,
        seeds=FINAL_SEEDS,
        out_base=out_base,
    )

    # ── External test predictions with TTA ────────────────────────────────────
    ext_path = Path(args.external_test_csv)
    if ext_path.exists():
        print(f"\n=== Predicting external test set with TTA (n_aug={N_TTA}, "
              f"{len(FINAL_SEEDS)} seeds) ===")
        ext_df     = pd.read_csv(ext_path)
        ext_smiles = ext_df["SMILES"].reset_index(drop=True)
        ext_preds  = predict_with_tta(final_models, ext_smiles, n_aug=N_TTA)

        pred_df = pd.DataFrame({
            EXT_ID_COL: ext_df[EXT_ID_COL].values,
            "SMILES":   ext_smiles.values,
            "pEC50":    ext_preds,
        })
        pred_path = out_base / "final_predictions.csv"
        pred_df.to_csv(pred_path, index=False)
        print(f"  {len(pred_df)} predictions → {pred_path}")
        print(f"  pEC50 range: {ext_preds.min():.3f} – {ext_preds.max():.3f}  "
              f"mean: {ext_preds.mean():.3f}")
    else:
        print(f"\nWarning: external test CSV not found: {ext_path}")

    # ── Summary JSON ──────────────────────────────────────────────────────────
    summary = {
        "n_train":              int(len(X)),
        "best_params":          best_params,
        "final_epochs":         args.final_epochs,
        "final_seeds":          FINAL_SEEDS,
        "n_tta":                N_TTA,
        "butina_cutoff":        BUTINA_CUTOFF,
        "counter_weight_scale": args.counter_weight_scale,
        "hpo_n_trials":         args.n_trials,
        "hpo_n_folds":          args.n_hpo_folds,
    }
    if oof_mae is not None:
        summary["oof_mae"]      = round(oof_mae, 5)
        summary["oof_spearman"] = round(oof_rho, 5)

    with open(out_base / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Done.  All artefacts in {out_base}/")
    print(f"  summary.json           — hyperparameters + OOF metrics")
    print(f"  oof_predictions.csv    — per-compound OOF pEC50 predictions")
    print(f"  final_predictions.csv  — blind test predictions (TTA + multi-seed)")
    print(f"  optuna.db              — resumable HPO study")
    print(f"  final_models/seed_*/   — saved model weights")


if __name__ == "__main__":
    main()
