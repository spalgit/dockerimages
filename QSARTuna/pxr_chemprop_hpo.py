#!/usr/bin/env python3
"""
Optuna hyperparameter optimisation for ChemProp PXR pEC50 regression.

OpenAdmet does not expose HPO for Lightning/ChemProp models through its YAML
format (only sklearn models support that via SKLearnGridSearchTrainer).
This script uses the OpenAdmet Python API directly and drives Optuna to
minimise the validation MAE, then reports the held-out test-set MAE and
writes a ready-to-run OpenAdmet YAML for the best configuration.

Search space
------------
  depth              : int in [2, 5]
  ffn_hidden_dim     : categorical {128, 256, 300, 512}
  ffn_num_layers     : int in [1, 4]
  message_hidden_dim : categorical {128, 256, 300, 512}
  max_lr             : log-uniform in [1e-4, 5e-3]
  dropout            : uniform in [0.0, 0.4]

NOTE: the original anvil_recipe.yaml used the key ``ffn_hidden_num_layers``
which is not a recognised ChemPropModel field — it was silently ignored and
the default ffn_num_layers=1 was used. This script uses the correct key.

Usage (openadmet conda env):
    cd ~/OpenAdmet
    python pxr_chemprop_hpo.py [--n_trials 40] [--out_dir hpo_chemprop_pxr]
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

# ── HPO search space bounds ───────────────────────────────────────────────────
DEPTH_RANGE          = (2, 5)
FFN_HIDDEN_DIM_OPTS  = [128, 256, 300, 512]
FFN_NUM_LAYERS_RANGE = (1, 4)
MSG_HIDDEN_DIM_OPTS  = [128, 256, 300, 512]
MAX_LR_RANGE         = (1e-4, 5e-3)   # sampled on log scale
DROPOUT_RANGE        = (0.0, 0.4)

# ── Training budget ───────────────────────────────────────────────────────────
TRIAL_MAX_EPOCHS  = 60
TRIAL_ES_PATIENCE = 10
FINAL_MAX_EPOCHS  = 100
FINAL_ES_PATIENCE = 15


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
    depth: int,
    ffn_hidden_dim: int,
    ffn_num_layers: int,
    message_hidden_dim: int,
    max_lr: float,
    dropout: float,
    batch_norm: bool,
    max_epochs: int,
    es_patience: int,
    output_dir: Path,
) -> tuple[float, ChemPropModel]:
    """
    Build, train, and evaluate one ChemProp configuration.
    Returns (val_MAE_unscaled, trained_model).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Featurize (training normalises targets; val featurised for early-stopping
    # monitoring only — model.predict() always returns unscaled values).
    feat = ChemPropFeaturizer()
    train_dl, _, train_scaler, _ = feat.featurize(X_tr, y_tr)
    val_dl,   _, _,            _ = feat.featurize(X_val, y_val)

    # Build model with train-set target scaler baked into the output transform.
    model = ChemPropModel(
        n_tasks=1,
        depth=depth,
        ffn_hidden_dim=ffn_hidden_dim,
        ffn_num_layers=ffn_num_layers,
        message_hidden_dim=message_hidden_dim,
        max_lr=max_lr,
        dropout=dropout,
        batch_norm=batch_norm,
        metric_list=["mse", "mae", "rmse"],
        monitor_metric="val_loss",
        scheduler="noam",
        warmup_epochs=2,
    )
    model.build(scaler=train_scaler)

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

    # Predict on val — model output is unscaled (UnscaleTransform applied).
    y_val_true = y_val[TARGET_COL].to_numpy()
    val_preds  = model.predict(val_dl, accelerator="gpu").flatten()
    val_mae    = float(mean_absolute_error(y_val_true, val_preds))

    return val_mae, model


# ─────────────────────────────────────────────────────────────────────────────
# Optuna objective
# ─────────────────────────────────────────────────────────────────────────────

def make_objective(X_train: pd.Series, y_train: pd.DataFrame, out_base: Path):
    def objective(trial: optuna.Trial) -> float:
        # ── Sample hyperparameters ──────────────────────────────────────────
        depth              = trial.suggest_int("depth",            *DEPTH_RANGE)
        ffn_hidden_dim     = trial.suggest_categorical("ffn_hidden_dim",      FFN_HIDDEN_DIM_OPTS)
        ffn_num_layers     = trial.suggest_int("ffn_num_layers",   *FFN_NUM_LAYERS_RANGE)
        message_hidden_dim = trial.suggest_categorical("message_hidden_dim",  MSG_HIDDEN_DIM_OPTS)
        max_lr             = trial.suggest_float("max_lr",         *MAX_LR_RANGE, log=True)
        dropout            = trial.suggest_float("dropout",        *DROPOUT_RANGE)
        batch_norm         = trial.suggest_categorical("batch_norm", [True, False])

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
                depth=depth,
                ffn_hidden_dim=ffn_hidden_dim,
                ffn_num_layers=ffn_num_layers,
                message_hidden_dim=message_hidden_dim,
                max_lr=max_lr,
                dropout=dropout,
                batch_norm=batch_norm,
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
    """Write a ready-to-run OpenAdmet YAML for the best HPO configuration."""
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
            "description": "ChemProp HPO best config — minimised validation MAE",
            "driver": "pytorch",
            "email": "sandeepbii@yahoo.com",
            "name": "chemprop_hpo_best",
            "tag": "chemprop",
            "tags": ["openadmet", "pxr", "hpo"],
            "version": "v1",
        },
        "procedure": {
            "ensemble": None,
            "feat": {"params": {}, "type": "ChemPropFeaturizer"},
            "model": {
                "freeze_weights": None,
                "param_path": None,
                "params": {
                    "batch_norm":        best_params["batch_norm"],
                    "depth":             best_params["depth"],
                    "dropout":           round(float(best_params["dropout"]), 4),
                    "ffn_hidden_dim":    best_params["ffn_hidden_dim"],
                    "ffn_num_layers":    best_params["ffn_num_layers"],
                    "from_chemeleon":    False,
                    "max_lr":            round(float(best_params["max_lr"]), 7),
                    "message_hidden_dim": best_params["message_hidden_dim"],
                    "messages":          "bond",
                    "n_tasks":           1,
                    "scheduler":         "noam",
                    "warmup_epochs":     2,
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
                    "accelerator":             "gpu",
                    "early_stopping":          True,
                    "early_stopping_min_delta": 0.001,
                    "early_stopping_mode":     "min",
                    "early_stopping_patience": FINAL_ES_PATIENCE,
                    "max_epochs":              FINAL_MAX_EPOCHS,
                    "monitor_metric":          "val_loss",
                    "use_wandb":               False,
                    "wandb_project":           "pxr_hpo",
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
                        "title":        "True vs Predicted pEC50 — HPO Best Model",
                    },
                    "type": "PytorchLightningRepeatedKFoldCrossValidation",
                },
            ]
        },
    }

    yaml_path = out_dir / "pxr_chemprop_hpo_best.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    return yaml_path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Optuna HPO for ChemProp PXR pEC50 — minimises test MAE"
    )
    parser.add_argument("--n_trials",  type=int,   default=40,               help="Number of Optuna trials")
    parser.add_argument("--out_dir",   default="hpo_chemprop_pxr",           help="Output directory")
    parser.add_argument("--data_csv",  default=DATA_CSV,                     help="Dataset CSV path")
    parser.add_argument("--seed",      type=int,   default=42)
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
    print(f"\nStarting Optuna HPO — {args.n_trials} trials...\n")
    study = optuna.create_study(
        direction="minimize",
        study_name="pxr_chemprop_hpo",
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
    X_tr_final  = X_tr_final.reset_index(drop=True)
    X_val_es    = X_val_es.reset_index(drop=True)
    y_tr_final  = y_tr_final.reset_index(drop=True)
    y_val_es    = y_val_es.reset_index(drop=True)

    _, final_model = train_and_eval(
        X_tr_final, y_tr_final, X_val_es, y_val_es,
        depth=best.params["depth"],
        ffn_hidden_dim=best.params["ffn_hidden_dim"],
        ffn_num_layers=best.params["ffn_num_layers"],
        message_hidden_dim=best.params["message_hidden_dim"],
        max_lr=best.params["max_lr"],
        dropout=best.params["dropout"],
        batch_norm=best.params["batch_norm"],
        max_epochs=FINAL_MAX_EPOCHS,
        es_patience=FINAL_ES_PATIENCE,
        output_dir=out_base / "best_model",
    )

    # ── Test-set evaluation ───────────────────────────────────────────────────
    print("Evaluating on held-out test set...")
    feat_test = ChemPropFeaturizer()
    test_dl, _, _, _ = feat_test.featurize(X_test)   # no y → no normalisation
    test_preds = final_model.predict(test_dl, accelerator="gpu").flatten()
    y_test_arr = y_test[TARGET_COL].to_numpy()

    test_mae  = float(mean_absolute_error(y_test_arr, test_preds))
    test_rmse = float(np.sqrt(np.mean((y_test_arr - test_preds) ** 2)))

    print(f"\n  Test MAE  : {test_mae:.4f}")
    print(f"  Test RMSE : {test_rmse:.4f}")

    # ── Save artefacts ────────────────────────────────────────────────────────
    results = {
        "best_trial":     best.number,
        "best_val_mae":   round(best.value, 5),
        "test_mae":       round(test_mae,   5),
        "test_rmse":      round(test_rmse,  5),
        "best_params":    best.params,
        "n_train":        int(len(X_train)),
        "n_test":         int(len(X_test)),
    }
    results_path = out_base / "hpo_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    yaml_path = write_best_yaml(best.params, out_base, args.data_csv)

    print(f"\nArtefacts saved to {out_base}/")
    print(f"  hpo_results.json            — trial summary + test metrics")
    print(f"  optuna.db                   — full Optuna study (resumable)")
    print(f"  pxr_chemprop_hpo_best.yaml  — ready-to-run OpenAdmet YAML")
    print(f"\nTo train the final model via openadmet CLI:")
    print(f"  conda activate openadmet && cd ~/OpenAdmet")
    print(f"  openadmet train --config {yaml_path}")


if __name__ == "__main__":
    main()
