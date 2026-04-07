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

dirname = "PRF_Regression_Plots"
if not os.path.exists(dirname):
    os.mkdir(dirname)
    print(f"Directory '{dirname}' created.")
else:
    print(f"Directory '{dirname}' already exists.")


config = OptimizationConfig(
    data=Dataset(
        input_column="SMILES",  # Typical names are "SMILES" and "smiles".
        response_column="pEC50",  # Often a specific name (like here), or just "activity".
        training_dataset_file="~/dockerimages/QSARTuna/PXR_For_QSARTuna_Modelling/processed_Openadmet_ChemBL_PXR_train_main.csv",
        test_dataset_file="~/dockerimages/QSARTuna/PXR_For_QSARTuna_Modelling/processed_Openadmet_ChemBL_PXR_test.csv",  # Hidden during optimization.
        probabilistic_threshold_representation=True, # This enables PTR
        probabilistic_threshold_representation_threshold=5.0, # This defines the activity threshold, Use 5.0 as it is
        probabilistic_threshold_representation_std=0.6,
    ),
    descriptors=[
        ECFP.new(),
        ECFP_counts.new(),
        MACCS_keys.new(),
        PathFP.new()
    ],
    algorithms=[
        PRFClassifier.new(
            max_depth={
                "low": 2,
                "high": 30
                },
            n_estimators={"low": 10, "high": 250},
            min_py_sum_leaf={"low": 1, "high": 10},
            max_features=["auto"]
            )
    ],
    settings=OptimizationConfig.Settings(
        mode=ModelMode.REGRESSION,
        n_splits=3,
        n_trials=100,  # Total number of trials. Make sure to change to 100 or so for a production run
        n_startup_trials=20,  # Number of startup ("random") trials.
        random_seed=42, # Seed for reproducability
        n_replicates=5,
        n_jobs=4,
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

expected_raw = df[config.data.response_column]
#expected = config.data.get_sets()[1]
predicted = model.predict_from_smiles(df[config.data.input_column])

def ptr_transform(y_raw, threshold=5.0, std=0.6):
    """
    Transform raw pEC50 values to PTR representation (0-1 scale)
    Matches QSARtuna's probabilistic threshold representation
    """
    z = (threshold - y_raw) / std  # Standardize distance from threshold
    ptr = 1.0 / (1.0 + np.exp(z))  # Logistic/sigmoid transformation
    return np.clip(ptr, 0.0, 1.0)

expected_ptr = ptr_transform(expected_raw,
                           threshold=config.data.probabilistic_threshold_representation_threshold,
                           std=config.data.probabilistic_threshold_representation_std)

# Calculate metrics on PTR scale (appropriate for classification/probability)
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error, roc_auc_score)

r2_ptr = r2_score(expected_ptr, predicted)
rmse_ptr = np.sqrt(mean_squared_error(expected_ptr, predicted))
mae_ptr = mean_absolute_error(expected_ptr, predicted)

# Also calculate classification metrics if you want binary performance
binary_expected = (expected_raw >= 5.0).astype(int)
auc = roc_auc_score(binary_expected, predicted)

print(f"PTR Scale Metrics:")
print(f"R² (PTR): {r2_ptr:.3f}")
print(f"RMSE (PTR): {rmse_ptr:.3f}")
print(f"MAE (PTR): {mae_ptr:.3f}")
print(f"ROC-AUC (binary): {auc:.3f}")

# Plot PTR expected vs predicted (following notebook style)
plt.figure(figsize=(8, 6))
ax = plt.scatter(expected_ptr, predicted, alpha=0.6, s=20)
lims = [0, 1]  # Both axes are 0-1 scale
plt.plot(lims, lims, color="black", linestyle="--", linewidth=2, label="Perfect prediction")
plt.xlabel(f"Expected {config.data.response_column} (PTR)")
plt.ylabel(f"Predicted {config.data.response_column} (PTR)")
plt.xlim(lims)
plt.ylim(lims)
plt.grid(True, alpha=0.3)

# Add metrics text
textstr = f"$R^2$ = {r2_ptr:.3f}\nRMSE = {rmse_ptr:.3f}\nMAE = {mae_ptr:.3f}\nAUC = {auc:.3f}"
plt.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
         verticalalignment="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

plt.legend()
plt.tight_layout()
plt.savefig(f"{dirname}/Test_set_PTR_scatter.png", dpi=300, bbox_inches="tight")
plt.close()

# Optional: Raw pEC50 vs Predicted probability plot (for interpretation)
plt.figure(figsize=(8, 6))
plt.scatter(expected_raw, predicted, alpha=0.6, s=20, c=binary_expected, cmap='RdYlBu_r')
plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Decision threshold')
plt.axvline(x=5.0, color='gray', linestyle='--', alpha=0.7, label='Activity threshold')
plt.xlabel(f"True pEC50")
plt.ylabel(f"Predicted Probability (PTR)")
plt.colorbar(label="Active (1) / Inactive (0)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{dirname}/Test_set_raw_vs_predicted.png", dpi=300, bbox_inches="tight")
plt.close()
