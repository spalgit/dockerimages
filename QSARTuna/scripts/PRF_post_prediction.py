import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

dirname = "PRF_Regression_Plots_5"  # Match your existing directory

# Load your data
#df_test = pd.read_csv(config.data.test_dataset_file.expanduser())
df_test = pd.read_csv("PRF_Regression_Plots_5/prediction_train.csv")
predicted_df = pd.read_csv("PRF_Regression_Plots_5/prediction_train.csv")

# Get expected raw values from test data
expected_raw = df_test["pEC50"]
predicted = predicted_df['Prediction'] # Adjust column name if different

# Transform expected to PTR scale (matching your config: threshold=5.0, std=0.6)
def ptr_transform(y_raw, threshold=5.0, std=0.6):
    """Transform raw pEC50 to PTR 0-1 scale"""
    z = (threshold - y_raw) / std
    ptr = 1.0 / (1.0 + np.exp(z))  # Sigmoid transformation
    return np.clip(ptr, 0.0, 1.0)

expected_ptr = ptr_transform(expected_raw)

# Calculate metrics
from sklearn.metrics import (r2_score, mean_squared_error, mean_absolute_error, roc_auc_score)

r2_ptr = r2_score(expected_ptr, predicted)
rmse_ptr = np.sqrt(mean_squared_error(expected_ptr, predicted))
mae_ptr = mean_absolute_error(expected_ptr, predicted)
binary_expected = (expected_raw >= 5.0).astype(int)
auc = roc_auc_score(binary_expected, predicted)

print(f"PTR Metrics: R²={r2_ptr:.3f}, RMSE={rmse_ptr:.3f}, MAE={mae_ptr:.3f}, AUC={auc:.3f}")

# PTR Parity Plot (like the notebook)
plt.figure(figsize=(8, 6))
plt.scatter(expected_ptr, predicted, alpha=0.6, s=20)
lims = [0, 1]
plt.plot(lims, lims, color="black", linestyle="--", linewidth=2, label="Perfect")
plt.xlabel(f"Expected")
plt.ylabel(f"Predicted")
plt.xlim(lims)
plt.ylim(lims)
plt.grid(True, alpha=0.3)

# Add metrics
textstr = f"$R^2$={r2_ptr:.3f}\nRMSE={rmse_ptr:.3f}\nMAE={mae_ptr:.3f}\nAUC={auc:.3f}"
plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes,
         fontsize=11, verticalalignment="top",
         bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
plt.legend()
plt.tight_layout()
plt.savefig(f"{dirname}/PTR_parity_plot.png", dpi=300, bbox_inches="tight")
plt.close()

# Raw pEC50 vs Predicted (biological interpretation)
plt.figure(figsize=(8, 6))
scatter = plt.scatter(expected_raw, predicted, alpha=0.6, s=20,
                     c=binary_expected, cmap='RdYlBu_r')
plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7, label='Pred threshold')
plt.axvline(x=5.0, color='gray', linestyle='--', alpha=0.7, label='Activity threshold')
plt.xlabel("True pEC50")
plt.ylabel("Predicted Probability (PTR)")
plt.colorbar(scatter, label="Active (1) / Inactive (0)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{dirname}/raw_vs_predicted_train.png", dpi=300, bbox_inches="tight")
plt.close()

print("Plots saved to:", dirname)
