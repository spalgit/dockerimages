#!/usr/bin/env python3
"""
pxr_chemprop_chemeleon_hpo.py
─────────────────────────────
Optuna HPO for PXR pEC50 regression using ChemProp with the CheMeleon
pretrained molecular foundation model (https://zenodo.org/records/15460715).

Key differences from pxr_chemprop_hpo.py
-----------------------------------------
* from_chemeleon=True  — the message-passing encoder is loaded from the
  pretrained CheMeleon weights.  depth, message_hidden_dim and messages
  are therefore FIXED by the pretrained weights and are removed from the
  search space.
* Remaining tunable parameters: ffn_hidden_dim, ffn_num_layers, max_lr,
  dropout, batch_norm, weight_decay.
* Training criterion is patched to MAE so that the model is optimised
  directly on the metric we care about.
* Study name and default output directory are distinct from previous runs
  so there are no Optuna database conflicts.

Usage (openadmet conda env):
    cd ~/OpenAdmet
    python pxr_chemprop_chemeleon_hpo.py [--n_trials 40] [--out_dir hpo_chemeleon_pxr]

CheMeleon citation (please include in publications):
    DOI: 10.48550/arXiv.2506.15792
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
import yaml
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from chemprop import nn as cp_nn  # for MAE criterion patch

# Registers all OpenAdmet classes into their registries.
from openadmet.models.registries import *  # noqa: F401 F403
from openadmet.models.architecture.chemprop import ChemPropModel
from openadmet.models.features.chemprop import ChemPropFeaturizer
from openadmet.models.trainer.lightning import LightningTrainer
from openadmet.models.split.scaffold import ScaffoldSplitter

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Dataset ───────────────────────────────────────────────────────────────────
DATA_CSV = (
    "/home/spal/dockerimages/QSARTuna/PXR_For_QSARTuna_Modelling/"
    "processed_Openadmet_REAL_PXR_train_AND_test_main_with_side_info_AND_counter_screen.csv"
)
SMILES_COL = "SMILES"
TARGET_COL = "pEC50"

# ── HPO search space ──────────────────────────────────────────────────────────
# depth, message_hidden_dim and messages are fixed by CheMeleon weights.
FFN_HIDDEN_DIM_OPTS  = [128, 256, 300, 512, 1024]
FFN_NUM_LAYERS_RANGE = (1, 4)
MAX_LR_RANGE         = (1e-4, 5e-3)   # log-uniform
DROPOUT_RANGE        = (0.0, 0.4)
WEIGHT_DECAY_RANGE   = (1e-6, 1e-2)   # log-uniform

# ── Training budget ───────────────────────────────────────────────────────────
TRIAL_MAX_EPOCHS  = 100
TRIAL_ES_PATIENCE = 20
FINAL_MAX_EPOCHS  = 150
FINAL_ES_PATIENCE = 25

STUDY_NAME = "pxr_chemprop_chemeleon_hpo"


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_data(csv_path: str) -> tuple[pd.Series, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    df = df[[SMILES_COL, TARGET_COL]].dropna().reset_index(drop=True)
    return df[SMILES_COL], df[[TARGET_COL]]


def scaffold_train_test_split(
    X: pd.Series, y: pd.DataFrame, test_size: float = 0.1, random_state: int = 42
) -> tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    splitter = ScaffoldSplitter(
        test_size=test_size,
        train_size=1.0 - test_size,
        val_size=0.0,
        random_state=random_state,
    )
    X_train, _, X_test, y_train, _, y_test, _ = splitter.split(X, y)
    return (
        X_train.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Core train + evaluate function
# ─────────────────────────────────────────────────────────────────────────────

def train_and_eval(
    X_tr: pd.Series,
    y_tr: pd.DataFrame,
    X_val: pd.Series,
    y_val: pd.DataFrame,
    *,
    ffn_hidden_dim: int,
    ffn_num_layers: int,
    max_lr: float,
    dropout: float,
    batch_norm: bool,
    weight_decay: float,
    max_epochs: int,
    es_patience: int,
    output_dir: Path,
) -> tuple[float, ChemPropModel]:
    """
    Build, train, and evaluate one CheMeleon-ChemProp configuration.
    Returns (val_MAE_unscaled, trained_model).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feat = ChemPropFeaturizer()
    train_dl, _, train_scaler, _ = feat.featurize(X_tr, y_tr)
    val_dl,   _, _,            _ = feat.featurize(X_val, y_val)

    model = ChemPropModel(
        n_tasks=1,
        from_chemeleon=True,      # load pretrained CheMeleon encoder
        ffn_hidden_dim=ffn_hidden_dim,
        ffn_num_layers=ffn_num_layers,
        max_lr=max_lr,
        dropout=dropout,
        batch_norm=batch_norm,
        weight_decay=weight_decay,
        metric_list=["mae", "mse", "rmse"],
        monitor_metric="val_loss",
        scheduler="noam",
        warmup_epochs=2,
    )
    model.build(scaler=train_scaler)

    # Patch training criterion to MAE so the model is optimised on the
    # same metric we evaluate — aligns training loss with the objective.
    model.estimator.predictor.criterion = cp_nn.metrics.MAE()

    trainer = LightningTrainer(
        max_epochs=max_epochs,
        accelerator="gpu",
        early_stopping=(es_patience > 0),
        early_stopping_patience=es_patience,
        early_stopping_mode="min",
        early_stopping_min_delta=0.001,
        output_dir=output_dir,
        use_wandb=False,
    )
    trainer.model = model
    trainer.build(no_val=False)
    model = trainer.train(train_dl, val_dl)

    y_val_true = y_val[TARGET_COL].to_numpy()
    val_preds  = model.predict(val_dl, accelerator="gpu").flatten()
    val_mae    = float(mean_absolute_error(y_val_true, val_preds))

    return val_mae, model


# ─────────────────────────────────────────────────────────────────────────────
# Optuna objective
# ─────────────────────────────────────────────────────────────────────────────

def make_objective(X_train: pd.Series, y_train: pd.DataFrame, out_base: Path):
    def objective(trial: optuna.Trial) -> float:
        # ── Sample hyperparameters (depth/message_hidden_dim fixed by CheMeleon)
        ffn_hidden_dim  = trial.suggest_categorical("ffn_hidden_dim",  FFN_HIDDEN_DIM_OPTS)
        ffn_num_layers  = trial.suggest_int("ffn_num_layers",          *FFN_NUM_LAYERS_RANGE)
        max_lr          = trial.suggest_float("max_lr",                *MAX_LR_RANGE, log=True)
        dropout         = trial.suggest_float("dropout",               *DROPOUT_RANGE)
        batch_norm      = trial.suggest_categorical("batch_norm",      [True, False])
        weight_decay    = trial.suggest_float("weight_decay",          *WEIGHT_DECAY_RANGE, log=True)

        # ── Inner HPO train / val split (random 80/20) ──────────────────────
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=trial.number
        )
        X_tr  = X_tr.reset_index(drop=True)
        X_val = X_val.reset_index(drop=True)
        y_tr  = y_tr.reset_index(drop=True)
        y_val = y_val.reset_index(drop=True)

        trial_dir = out_base / f"trial_{trial.number:03d}"
        try:
            val_mae, _ = train_and_eval(
                X_tr, y_tr, X_val, y_val,
                ffn_hidden_dim=ffn_hidden_dim,
                ffn_num_layers=ffn_num_layers,
                max_lr=max_lr,
                dropout=dropout,
                batch_norm=batch_norm,
                weight_decay=weight_decay,
                max_epochs=TRIAL_MAX_EPOCHS,
                es_patience=TRIAL_ES_PATIENCE,
                output_dir=trial_dir,
            )
        except Exception as e:
            print(f"  [Trial {trial.number}] FAILED: {e}")
            raise optuna.exceptions.TrialPruned()

        print(
            f"  [Trial {trial.number:03d}] val_MAE={val_mae:.4f} | "
            + " | ".join(f"{k}={v}" for k, v in trial.params.items())
        )
        return val_mae

    return objective


# ─────────────────────────────────────────────────────────────────────────────
# YAML writer for best config
# ─────────────────────────────────────────────────────────────────────────────

def write_best_yaml(best_params: dict, out_dir: Path, data_csv: str) -> Path:
    """Write a ready-to-run OpenAdmet YAML for the best CheMeleon configuration."""
    config = {
        "data": {
            "anvil_dir": "file:///home/spal/OpenAdmet",
            "cat_entry": None,
            "dropna": True,
            "input_col": SMILES_COL,
            "resource": data_csv,
            "target_cols": [TARGET_COL],
            "test_resource": None,
            "train_resource": None,
            "type": "intake",
            "val_resource": None,
        },
        "metadata": {
            "authors": "Sandeep Pal",
            "biotargets": ["PXR"],
            "build_number": 0,
            "description": "ChemProp + CheMeleon HPO best config — minimised validation MAE",
            "driver": "pytorch",
            "email": "sandeepbii@yahoo.com",
            "name": "chemprop_chemeleon_hpo_best",
            "tag": "chemprop_chemeleon",
            "tags": ["openadmet", "pxr", "hpo", "chemeleon"],
            "version": "v1",
        },
        "procedure": {
            "ensemble": None,
            "feat": {"params": {}, "type": "ChemPropFeaturizer"},
            "model": {
                "freeze_weights": None,
                "param_path": None,
                "params": {
                    "from_chemeleon":  True,
                    "ffn_hidden_dim":  best_params["ffn_hidden_dim"],
                    "ffn_num_layers":  best_params["ffn_num_layers"],
                    "max_lr":          round(float(best_params["max_lr"]), 7),
                    "dropout":         round(float(best_params["dropout"]), 4),
                    "batch_norm":      best_params["batch_norm"],
                    "weight_decay":    round(float(best_params["weight_decay"]), 8),
                    "n_tasks":         1,
                    "scheduler":       "noam",
                    "warmup_epochs":   2,
                    "metric_list":     ["mae", "mse", "rmse"],
                },
                "serial_path": None,
                "type": "ChemPropModel",
            },
            "split": {
                "params": {
                    "random_state": 42,
                    "test_size":    0.1,
                    "train_size":   0.9,
                    "val_size":     0.0,
                },
                "type": "ScaffoldSplitter",
            },
            "train": {
                "params": {
                    "accelerator":              "gpu",
                    "early_stopping":           True,
                    "early_stopping_min_delta": 0.001,
                    "early_stopping_mode":      "min",
                    "early_stopping_patience":  FINAL_ES_PATIENCE,
                    "max_epochs":               FINAL_MAX_EPOCHS,
                    "monitor_metric":           "val_loss",
                    "use_wandb":                False,
                    "wandb_project":            "pxr_chemeleon_hpo",
                },
                "type": "LightningTrainer",
            },
            "transform": None,
        },
        "report": {
            "eval": [
                {"params": {}, "type": "RegressionMetrics"},
                {
                    "params": {
                        "axes_labels":  ["True pEC50", "Predicted pEC50"],
                        "n_repeats":    3,
                        "n_splits":     3,
                        "pXC50":        True,
                        "random_state": 42,
                        "title":        "True vs Predicted pEC50 — CheMeleon HPO Best Model",
                    },
                    "type": "PytorchLightningRepeatedKFoldCrossValidation",
                },
            ]
        },
    }

    yaml_path = out_dir / "pxr_chemprop_chemeleon_hpo_best.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    return yaml_path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Optuna HPO for ChemProp + CheMeleon PXR pEC50 — minimises MAE"
    )
    parser.add_argument("--n_trials", type=int, default=40,               help="Number of Optuna trials")
    parser.add_argument("--out_dir",  default="hpo_chemeleon_pxr",        help="Output directory")
    parser.add_argument("--data_csv", default=DATA_CSV,                   help="Dataset CSV path")
    parser.add_argument("--seed",     type=int, default=42)
    args = parser.parse_args()

    out_base = Path(args.out_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    # ── Load data & scaffold split ────────────────────────────────────────────
    print("Loading data...")
    X, y = load_data(args.data_csv)
    print(f"  {len(X)} compounds")

    print("Applying scaffold train/test split (90% / 10%)...")
    X_train, X_test, y_train, y_test = scaffold_train_test_split(
        X, y, test_size=0.1, random_state=args.seed
    )
    print(f"  Train: {len(X_train)}   Test: {len(X_test)}")

    # ── Optuna study ──────────────────────────────────────────────────────────
    print(f"\nStarting CheMeleon HPO — {args.n_trials} trials...\n")
    study = optuna.create_study(
        direction="minimize",
        study_name=STUDY_NAME,
        storage=f"sqlite:///{out_base}/optuna.db",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=args.seed),
    )
    study.optimize(
        make_objective(X_train, y_train, out_base),
        n_trials=args.n_trials,
        gc_after_trial=True,
    )

    # ── Best trial summary ────────────────────────────────────────────────────
    best = study.best_trial
    print("\n" + "=" * 60)
    print(f"BEST TRIAL  #{best.number}   val_MAE = {best.value:.4f}")
    print("Hyperparameters:")
    for k, v in best.params.items():
        print(f"  {k:25s} = {v}")

    # ── Retrain on full training set with 10% val for early stopping ──────────
    print("\nRetraining best config on full training set...")
    X_tr_final, X_val_es, y_tr_final, y_val_es = train_test_split(
        X_train, y_train, test_size=0.1, random_state=args.seed
    )
    X_tr_final = X_tr_final.reset_index(drop=True)
    X_val_es   = X_val_es.reset_index(drop=True)
    y_tr_final = y_tr_final.reset_index(drop=True)
    y_val_es   = y_val_es.reset_index(drop=True)

    _, final_model = train_and_eval(
        X_tr_final, y_tr_final, X_val_es, y_val_es,
        ffn_hidden_dim=best.params["ffn_hidden_dim"],
        ffn_num_layers=best.params["ffn_num_layers"],
        max_lr=best.params["max_lr"],
        dropout=best.params["dropout"],
        batch_norm=best.params["batch_norm"],
        weight_decay=best.params["weight_decay"],
        max_epochs=FINAL_MAX_EPOCHS,
        es_patience=FINAL_ES_PATIENCE,
        output_dir=out_base / "best_model",
    )

    # ── Test-set evaluation ───────────────────────────────────────────────────
    print("Evaluating on held-out test set...")
    feat_test = ChemPropFeaturizer()
    test_dl, _, _, _ = feat_test.featurize(X_test)
    test_preds = final_model.predict(test_dl, accelerator="gpu").flatten()
    y_test_arr = y_test[TARGET_COL].to_numpy()

    test_mae  = float(mean_absolute_error(y_test_arr, test_preds))
    test_rmse = float(np.sqrt(np.mean((y_test_arr - test_preds) ** 2)))

    print(f"\n  Test MAE  : {test_mae:.4f}")
    print(f"  Test RMSE : {test_rmse:.4f}")

    # ── Save artefacts ────────────────────────────────────────────────────────
    results = {
        "model":          "ChemProp + CheMeleon",
        "best_trial":     best.number,
        "best_val_mae":   round(best.value,  5),
        "test_mae":       round(test_mae,    5),
        "test_rmse":      round(test_rmse,   5),
        "best_params":    best.params,
        "n_train":        int(len(X_train)),
        "n_test":         int(len(X_test)),
    }
    results_path = out_base / "hpo_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    yaml_path = write_best_yaml(best.params, out_base, args.data_csv)

    print(f"\nArtefacts saved to {out_base}/")
    print(f"  hpo_results.json                    — trial summary + test metrics")
    print(f"  optuna.db                            — full Optuna study (resumable)")
    print(f"  pxr_chemprop_chemeleon_hpo_best.yaml — ready-to-run OpenAdmet YAML")
    print(f"\nTo train the final model via openadmet CLI:")
    print(f"  conda activate openadmet && cd ~/OpenAdmet")
    print(f"  openadmet train --config {yaml_path}")
    print(f"\nNote: please cite DOI 10.48550/arXiv.2506.15792 when using CheMeleon.")


if __name__ == "__main__":
    main()
