import sys
import apischema
import sklearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import dill
import warnings
import logging
from pathlib import Path
import tqdm
from functools import partialmethod, partial

# QSARTuna imports (de-duplicated)
from optunaz.three_step_opt_build_merge import (
    optimize,
    buildconfig_best,
    build_best,
    build_merged,
)
from optunaz.config import ModelMode, OptimizationDirection
from optunaz.config.optconfig import (
    OptimizationConfig,
    ChemPropRegressor,
    RandomForestRegressor,
    SVR,
    Ridge,
    Lasso,
    PLSRegression,
    KNeighborsRegressor,
    XGBRegressor
)
from optunaz.datareader import Dataset
from optunaz.descriptors import SmilesAndSideInfoFromFile
from optunaz.utils.preprocessing.splitter import Stratified
from optunaz.utils.preprocessing.deduplicator import KeepMedian
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error)

# Setup directories and logging
dirname = "ChemProp_Plots"
os.makedirs(dirname, exist_ok=True)
print(f"Directory '{dirname}' {'created' if not os.path.exists(dirname) else 'already exists'}.")

# Suppress warnings and logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("train").disabled = True
np.seterr(divide="ignore")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

# Disable ChemProp verbose output
def warn(*args, **kwargs):
    pass
warnings.warn = warn

# ✅ FIXED CONFIGURATION
config = OptimizationConfig(
    data=Dataset(
        input_column="SMILES",
        response_column="KSOL",
        training_dataset_file="~/dockerimages/QSARTuna/processed_Openadmet_train_main.csv",
        test_dataset_file="~/dockerimages/QSARTuna/processed_Openadmet_test.csv"
    ),
    descriptors=[
        SmilesAndSideInfoFromFile.new(
            file="~/dockerimages/QSARTuna/processed_Openadmet_train_side_info.csv",
            input_column='SMILES',
            y_aux_weight_pc={"low": 0, "high": 100, "step": 10},
            y_aux_column= "ylabels"
        )
    ],
    algorithms=[
        ChemPropRegressor.new(
            aggregation_norm={"low": 100, "high": 100},
            batch_size=[64],
            ensemble_size=1,
            epochs=10,
            depth={"low": 2, "high": 6},
            ffn_hidden_dim={"low": 300, "high": 2400},
            ffn_num_layers={"low": 1, "high": 4},
            final_lr_ratio={"low": 1e-2, "high": 1e-2},
            message_hidden_dim={"low": 300, "high": 2400},
            init_lr_ratio={"low": 1e-2, "high": 1e-2},
            max_lr={"low": 1e-2, "high": 1e-2},
            warmup_epochs_ratio={"low": 0.1, "high": 0.1},
            activation=["RELU"],  # ✅ Fixed: string list
            aggregation=["norm"],  # ✅ Fixed: string list
            loss_function=["mve"],  # ✅ Fixed: string list
            undirected=[True],  # ✅ Fixed: boolean True (capital T)
            message_bias=[True],  # ✅ Fixed: boolean True
            batch_norm=[True, False],  # ✅ Fixed: boolean list
            molecule_featurizers=["morgan_binary", "morgan_count"]  # ✅ Fixed: string list
        )
    ],
    settings=OptimizationConfig.Settings(
        mode=ModelMode.REGRESSION,
        n_splits=2,
        n_replicates=2,
        n_trials=20,  # Increase to 100+ for production
        n_startup_trials=2,
        random_seed=42,
        scoring="r2",
        n_jobs=4,
        direction=OptimizationDirection.MAXIMIZATION,
    ),
)

# Run optimization
print("🚀 Starting optimization...")
study = optimize(config, study_name="chemprop_study")

# Plot optimization progress
plt.figure(figsize=(10, 6))
sns.set_theme(style="darkgrid")
default_reg_scoring = config.settings.scoring
ax = sns.scatterplot(data=study.trials_dataframe(), x="number", y="value")
ax.set(xlabel="Trial number", ylabel=f"Objective value\n({default_reg_scoring})")
plt.savefig(f"{dirname}/optimization_progress.png", dpi=300, bbox_inches="tight")
plt.close()

# CV plot
cv_test = study.trials_dataframe()["user_attrs_test_scores"].map(lambda d: d[default_reg_scoring])
x, y, fold = [], [], []
for i, vs in cv_test.items():
    for idx, v in enumerate(vs):
        x.append(i)
        y.append(v)
        fold.append(idx)

plt.figure(figsize=(10, 6))
ax = sns.scatterplot(x=x, y=y, hue=fold, style=fold, palette='Set1')
ax.set(xlabel="Trial number", ylabel=f"Objective value\n({default_reg_scoring})")
plt.savefig(f"{dirname}/CV_optimization_progress.png", dpi=300, bbox_inches="tight")
plt.close()

# Build best model
print("🏗️ Building best model...")
buildconfig = buildconfig_best(study)
buildconfig_as_dict = apischema.serialize(buildconfig)

# Save config
with open(os.path.join(dirname, "best_Model_config.json"), "w") as f:
    json.dump(buildconfig_as_dict, f, indent=2)

# Build and save best model
best_build = build_best(buildconfig, f"{dirname}/best.pkl")

# Test predictions (✅ FIXED for ChemProp)
print("🔮 Making test predictions...")
with open(f"{dirname}/best.pkl", "rb") as f:
    model = dill.load(f)


df = pd.read_csv(config.data.test_dataset_file)  # Load test data.

expected = df[config.data.response_column]
predicted = model.predict_from_smiles(df[config.data.input_column])

from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error)
import numpy as np

# R2
r2 = r2_score(y_true=expected, y_pred=predicted)

# RMSE. sklearn 0.24 added squared=False to get RMSE, here we use np.sqrt().
rmse = np.sqrt(mean_squared_error(y_true=expected, y_pred=predicted))

# MAE
mae = mean_absolute_error(y_true=expected, y_pred=predicted)


plt.figure()
plt.scatter(expected, predicted)

# Get the current axes (this is what you should call .plot on)
ax = plt.gca()

# Diagonal line
lims = [expected.min(), expected.max()]
ax.plot(lims, lims, color="black", linestyle="--", linewidth=1)

ax.set_xlabel(f"Expected {config.data.response_column}")
ax.set_ylabel(f"Predicted {config.data.response_column}")

# Add R², MSE, MAE in upper left
textstr = f"$R^2$ = {r2:.3f}\nMSE = {rmse:.3f}\nMAE = {mae:.3f}"
ax.text(
    0.02, 0.98, textstr,
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
)

plt.savefig(f"{dirname}/Test_set_Regression_corr.png", dpi=300, bbox_inches="tight")
plt.close()


build_merged(buildconfig, f"{dirname}/merged.pkl")
