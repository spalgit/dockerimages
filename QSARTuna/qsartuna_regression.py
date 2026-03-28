import sys
import apischema

# Start with the imports.
import sklearn
from optunaz.three_step_opt_build_merge import (
    optimize,
    buildconfig_best,
    build_best,
    build_merged,
)
from optunaz.config import ModelMode, OptimizationDirection
from optunaz.config.optconfig import (
    OptimizationConfig,
    SVR,
    RandomForestRegressor,
    Ridge,
    Lasso,
    PLSRegression,
    KNeighborsRegressor
)
from optunaz.datareader import Dataset
from optunaz.descriptors import ECFP, MACCS_keys, ECFP_counts, PathFP
from optunaz.config.optconfig import ChemPropRegressor
from optunaz.descriptors import SmilesBasedDescriptor, SmilesFromFile
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error)
import numpy as np
from optunaz.utils.preprocessing.splitter import Stratified
from optunaz.utils.preprocessing.deduplicator import KeepMedian
from optunaz.config.optconfig import AnyAlgorithm
from optunaz.config.optconfig import PRFClassifier
from optunaz.config.optconfig import ChemPropRegressor
from optunaz.descriptors import SmilesBasedDescriptor, SmilesFromFile
from optunaz.config.optconfig import ChemPropClassifier, RandomForestClassifier
import matplotlib.pyplot as plt
import os
import seaborn as sns
import dill
import pandas as pd
import matplotlib.pyplot as plt

import sklearn
from optunaz.three_step_opt_build_merge import (
    optimize,
    buildconfig_best,
    build_best,
    build_merged,
)
from optunaz.config import ModelMode, OptimizationDirection
from optunaz.config.optconfig import (
    OptimizationConfig,
    SVR,
    RandomForestRegressor,
    Ridge,
    Lasso,
    PLSRegression,
    KNeighborsRegressor,
    XGBRegressor,
#    ChemPropRegressor,
#    ChemPropHyperoptRegressor,
#    CustomRegressionModel
)
from optunaz.datareader import Dataset
from optunaz.descriptors import ECFP, MACCS_keys, ECFP_counts, PathFP

dirname = "Regression_Plots"
if not os.path.exists(dirname):
    os.mkdir(dirname)
    print(f"Directory '{dirname}' created.")
else:
    print(f"Directory '{dirname}' already exists.")


config = OptimizationConfig(
    data=Dataset(
        input_column="Smiles",  # Typical names are "SMILES" and "smiles".
        response_column="pIC50",  # Often a specific name (like here), or just "activity".
        training_dataset_file="processed_CYP3A4_inhibition_train.csv",
        test_dataset_file="processed_CYP3A4_inhibition_test.csv"  # Hidden during optimization.
    ),
    descriptors=[
        ECFP.new(),
        ECFP_counts.new(),
        MACCS_keys.new(),
        PathFP.new()
    ],
    algorithms=[
        SVR.new(C={"low": 1e-10,
            "high": 100.0},
            gamma={"low":0.0001,
                "high": 100.0}),
        RandomForestRegressor.new(n_estimators={"low": 10, "high": 250},
            max_depth={"low": 2, "high": 32},
            max_features=["auto"]),
        Ridge.new(alpha={"low":1e-6, "high":2}),
        Lasso.new(alpha={"low":1e-6, "high":2}),
        PLSRegression.new(n_components={"low": 2, "high": 3}),
        XGBRegressor.new(max_depth={"low": 2, "high": 32},
            n_estimators={"low": 3, "high": 100},
            learning_rate={"low": 0.1, "high": 0.1}),
        KNeighborsRegressor.new()
    ],
    settings=OptimizationConfig.Settings(
        mode=ModelMode.REGRESSION,
        n_splits=3,
        n_trials=100,  # Total number of trials. Make sure to change to 100 or so for a production run
        n_startup_trials=50,  # Number of startup ("random") trials.
        random_seed=42, # Seed for reproducability
        scoring="r2",
        direction=OptimizationDirection.MAXIMIZATION,
    ),
)

# Setup basic logging.
import logging
from importlib import reload
reload(logging)
logging.basicConfig(level=logging.INFO)
logging.getLogger("train").disabled = True # Prevent ChemProp from logging
import numpy as np
np.seterr(divide="ignore")
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import tqdm
from functools import partialmethod, partial
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True) # Prevent tqdm in ChemProp from flooding log

# Avoid decpreciated warnings from packages etc
import warnings
warnings.simplefilter("ignore")
def warn(*args, **kwargs):
    pass
warnings.warn = warn


study = optimize(config, study_name="my_study")

plt.figure()
sns.set_theme(style="darkgrid")
default_reg_scoring= config.settings.scoring
ax = sns.scatterplot(data=study.trials_dataframe(), x="number", y="value");
ax.set(xlabel="Trial number", ylabel=f"Ojbective value\n({default_reg_scoring})");
plt.savefig(f"{dirname}/optimization_progress.png", dpi=300, bbox_inches="tight")
plt.close()


cv_test = study.trials_dataframe()["user_attrs_test_scores"].map(lambda d: d[default_reg_scoring])
x = []
y = []
fold = []
for i, vs in cv_test.items():
    for idx, v in enumerate(vs):
        x.append(i)
        y.append(v)
        fold.append(idx)

plt.figure()
ax = sns.scatterplot(x=x, y=y, hue=fold, style=fold, palette='Set1')
ax.set(xlabel="Trial number", ylabel=f"Ojbective value\n({default_reg_scoring})");
plt.savefig(f"{dirname}/CV_optimization_progress.png", dpi=300, bbox_inches="tight")
plt.close()


buildconfig = buildconfig_best(study)
import apischema
buildconfig_as_dict = apischema.serialize(buildconfig)

import json
filepath = os.path.join(dirname, "best_Model_config.json")

with open(filepath, "w") as f:
    json.dump(buildconfig_as_dict, f, indent=2)

best_build = build_best(buildconfig, f"{dirname}/best.pkl")

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
