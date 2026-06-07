"""
Compute per-compound PTR (Probabilistic Threshold Representation) labels
for the PXR multitask ChemProp model.

Forward transform (training):
    effective_std_i = max(std_error_i, STD_FLOOR)
    PTR(y_i) = norm.cdf(y_i, threshold, effective_std_i)

Reverse transform (inference):
    effective_std_i = max(std_error_test_i, STD_FLOOR)
    pEC50_pred = norm.ppf(ptr_pred_i, threshold, effective_std_i)

STD_FLOOR prevents small per-compound std values from collapsing deeply
inactive compounds (pEC50 << threshold) to PTR≈0, which is unrecoverable.

Inputs:
  - processed_Openadmet_REAL_PXR_train_AND_test_main_with_side_info_AND_counter_screen.csv
    (training labels + per-compound std_error)
  - train_multitask_weighted.csv
    (provides the weight column and ensures same compound set)
  - test_phase1.csv
    (253 scored test compounds with per-compound std_error for reverse transform)

Output:
  - train_ptr_percompound_std.csv  — drop-in replacement for train_multitask_weighted.csv
  - test_phase1_ptr_reverse.csv    — lookup table: SMILES → std_error for reverse transform
"""

import pandas as pd
import numpy as np
from scipy.stats import norm

# ── Parameters ────────────────────────────────────────────────────────────────
THRESHOLD = 5.0        # pEC50 decision boundary (10 µM); becomes PTR = 0.5
PTR_CLIP  = 1e-9       # clip PTR away from exact 0/1 for numerical stability
STD_FLOOR = 0.40       # minimum effective std; prevents collapse for pEC50 << threshold
                       # floor of 0.4 keeps the recoverable range ≈ pEC50 2.5–7.5
SRC_DIR   = "/home/spal/dockerimages/QSARTuna/PXR_For_QSARTuna_Modelling"
OUT_DIR   = "/home/spal/dockerimages/QSARTuna/PXR"

# ── Load data ─────────────────────────────────────────────────────────────────
src = pd.read_csv(
    f"{SRC_DIR}/processed_Openadmet_REAL_PXR_train_AND_test_main_with_side_info_AND_counter_screen.csv"
)
base = pd.read_csv(f"{OUT_DIR}/train_multitask_weighted.csv")
test = pd.read_csv(f"{OUT_DIR}/test_phase1.csv")

# ── Merge src std_error onto base (join on Molecule Name / ID) ────────────────
src = src.rename(columns={"ID": "Molecule Name"})
merged = base.merge(
    src[["Molecule Name", "std_error", "std_error_counter"]],
    on="Molecule Name",
    how="left",
)

missing_se = merged["std_error"].isna().sum()
if missing_se > 0:
    median_se = src["std_error"].median()
    print(f"WARNING: {missing_se} compounds missing std_error — imputing with median {median_se:.3f}")
    merged["std_error"] = merged["std_error"].fillna(median_se)

# Impute missing counter std_error with column median
median_se_counter = src["std_error_counter"].median()
missing_sec = merged["std_error_counter"].isna().sum()
print(f"Counter std_error: {missing_sec} nulls imputed with median {median_se_counter:.3f}")
merged["std_error_counter"] = merged["std_error_counter"].fillna(median_se_counter)

# ── Apply per-compound PTR transform ──────────────────────────────────────────
def effective_std(std, floor=STD_FLOOR):
    """Apply std floor: preserves per-compound uncertainty while preventing collapse."""
    return np.maximum(std, floor)

def ptr_transform(y, std, threshold=THRESHOLD, clip=PTR_CLIP):
    """Apply PTR forward transform; NaN-safe. Uses effective_std internally."""
    result = np.full(len(y), np.nan)
    eff = effective_std(std)
    mask = np.isfinite(y) & np.isfinite(eff)
    result[mask] = norm.cdf(y[mask], threshold, eff[mask]).clip(clip, 1 - clip)
    return result

merged["pEC50_ptr"]         = ptr_transform(merged["pEC50"].values,         merged["std_error"].values)
merged["pEC50_counter_ptr"] = ptr_transform(merged["pEC50_counter"].values,  merged["std_error_counter"].values)

# ── Build output training CSV (same structure as train_multitask_weighted.csv) ─
train_out = merged[["Molecule Name", "SMILES", "pEC50_ptr", "pEC50_counter_ptr", "weight"]].copy()
train_out = train_out.rename(columns={"pEC50_ptr": "pEC50", "pEC50_counter_ptr": "pEC50_counter"})
train_out.to_csv(f"{OUT_DIR}/train_ptr_percompound_std.csv", index=False)
print(f"\nSaved: {OUT_DIR}/train_ptr_percompound_std.csv  ({len(train_out)} rows)")

# ── Build test reverse-transform lookup ───────────────────────────────────────
# Maps each test SMILES → std_error so that at inference:
#   pEC50_pred = norm.ppf(ptr_pred, THRESHOLD, std_error_test)
test_se_col = "pEC50_std.error (-log10(molarity))"
test_out = test[["Molecule Name", "SMILES", "pEC50", test_se_col]].copy()
test_out = test_out.rename(columns={test_se_col: "std_error"})
test_out["effective_std"] = effective_std(test_out["std_error"].values)
test_out["ptr_label"] = ptr_transform(test_out["pEC50"].values, test_out["std_error"].values)
test_out.to_csv(f"{OUT_DIR}/test_phase1_ptr_reverse.csv", index=False)
print(f"Saved: {OUT_DIR}/test_phase1_ptr_reverse.csv  ({len(test_out)} rows)")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\nPTR parameters: threshold={THRESHOLD}, per-compound std_error")
print(f"\nTraining PTR label (pEC50) distribution:")
print(train_out["pEC50"].describe().round(3).to_string())
print(f"\nTraining PTR label (pEC50_counter) distribution:")
print(train_out["pEC50_counter"].describe().round(3).to_string())

# Sanity-check round-trip on training data (should recover original pEC50 closely)
ptr_vals = merged["pEC50_ptr"].values
se_vals  = effective_std(merged["std_error"].values)
recovered = norm.ppf(ptr_vals.clip(PTR_CLIP, 1 - PTR_CLIP), THRESHOLD, se_vals)
finite    = np.isfinite(recovered) & np.isfinite(merged["pEC50"].values)
rt_mae    = np.mean(np.abs(merged["pEC50"].values[finite] - recovered[finite]))
print(f"\nRound-trip MAE (train, PTR → reverse → pEC50): {rt_mae:.4f} log units")

print(f"\nReverse-transform std_error (test, 253 compounds):")
print(test_out["std_error"].describe().round(3).to_string())
print(f"\nDone. Use 'train_ptr_percompound_std.csv' in pxr_chemprop_multitask_cw.yaml")
print(f"After prediction, reverse with: norm.ppf(ptr_pred, {THRESHOLD}, std_error_test_i)")
