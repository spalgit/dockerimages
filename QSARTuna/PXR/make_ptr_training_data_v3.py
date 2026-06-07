"""
PTR label generation v3 — threshold=5.0, std=0.55

Rationale vs previous versions:
  v1: threshold=5.0, std=0.40 — active zone (5.5-7.0) only 0.106 PTR spread → compressed
  v2: threshold=3.5, std=1.00 — active zone pushed into flat tail   → MAE=1.34 for actives
  v3: threshold=5.0, std=0.55 — active zone spread = 0.182 (72% wider than v1)
                                  pEC50=1.7 recovers to 1.701 (err=0.001)
                                  full range [1.7-7.0] recoverable with near-zero round-trip error

Forward transform (training labels):
    PTR(y_i) = norm.cdf(y_i, 5.0, 0.55)

Reverse transform (inference):
    pEC50_pred = norm.ppf(ptr_pred, 5.0, 0.55)

PTR coverage:
    pEC50=1.7  -> PTR=0.000  (clips to 1e-9, recovered=1.701)
    pEC50=3.0  -> PTR=0.000
    pEC50=4.5  -> PTR=0.182
    pEC50=5.0  -> PTR=0.500  (threshold midpoint)
    pEC50=5.5  -> PTR=0.818
    pEC50=6.0  -> PTR=0.965
    pEC50=6.5  -> PTR=0.997
    pEC50=7.0  -> PTR=0.9999

Inputs:
  - processed_Openadmet_REAL_PXR_train_AND_test_main_with_side_info_AND_counter_screen.csv
  - train_multitask_weighted.csv  (weight column + compound set)

Outputs:
  - train_ptr_v3.csv  (drop-in replacement for train_multitask_weighted.csv)
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

# ── Parameters ────────────────────────────────────────────────────────────────
THRESHOLD = 5.0
STD       = 0.55
PTR_CLIP  = 1e-9

SRC_FILE  = ("/home/spal/dockerimages/QSARTuna/PXR_For_QSARTuna_Modelling"
             "/processed_Openadmet_REAL_PXR_train_AND_test_main_with_side_info"
             "_AND_counter_screen.csv")
BASE_FILE = "/home/spal/dockerimages/QSARTuna/PXR/train_multitask_weighted.csv"
OUT_FILE  = "/home/spal/dockerimages/QSARTuna/PXR/train_ptr_v3.csv"


def ptr_forward(y):
    result = np.full(len(y), np.nan)
    mask = np.isfinite(y)
    result[mask] = norm.cdf(y[mask], THRESHOLD, STD).clip(PTR_CLIP, 1 - PTR_CLIP)
    return result


def ptr_reverse(p):
    return norm.ppf(np.clip(p, PTR_CLIP, 1 - PTR_CLIP), THRESHOLD, STD)


# ── Load data ─────────────────────────────────────────────────────────────────
src  = pd.read_csv(SRC_FILE).rename(columns={"ID": "Molecule Name"})
base = pd.read_csv(BASE_FILE)

merged = base.merge(
    src[["Molecule Name", "pEC50_counter"]],
    on="Molecule Name",
    how="left",
    suffixes=("", "_src"),
)
if "pEC50_counter_src" in merged.columns:
    merged["pEC50_counter"] = merged["pEC50_counter"].combine_first(merged["pEC50_counter_src"])
    merged = merged.drop(columns=["pEC50_counter_src"])

# ── Apply PTR forward transform ───────────────────────────────────────────────
merged["pEC50_ptr"]         = ptr_forward(merged["pEC50"].values)
merged["pEC50_counter_ptr"] = ptr_forward(merged["pEC50_counter"].values)

# ── Build output CSV ──────────────────────────────────────────────────────────
out = merged[["Molecule Name", "SMILES", "pEC50_ptr", "pEC50_counter_ptr", "weight"]].copy()
out = out.rename(columns={"pEC50_ptr": "pEC50", "pEC50_counter_ptr": "pEC50_counter"})
out.to_csv(OUT_FILE, index=False)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"PTR v3 parameters: threshold={THRESHOLD}, std={STD}")
print(f"Saved: {OUT_FILE}  ({len(out)} rows)")
print()
print("PTR label (pEC50) distribution:")
print(out["pEC50"].describe().round(4).to_string())
print()
print("PTR label (pEC50_counter) distribution:")
print(out["pEC50_counter"].describe().round(4).to_string())
print()

# Round-trip sanity check
y_orig   = merged["pEC50"].values
ptr_vals = merged["pEC50_ptr"].values
recovered = ptr_reverse(ptr_vals)
finite   = np.isfinite(y_orig) & np.isfinite(recovered)
rt_mae   = np.mean(np.abs(y_orig[finite] - recovered[finite]))
print(f"Round-trip MAE (PTR -> reverse -> pEC50): {rt_mae:.6f} log units")
print()
print("PTR coverage spot-check:")
for pec in [1.7, 3.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]:
    p = norm.cdf(pec, THRESHOLD, STD)
    r = ptr_reverse(p)
    print(f"  pEC50={pec:.1f}  PTR={p:.5f}  recovered={r:.4f}")
print()
print(f"Use 'train_ptr_v3.csv' in pxr_chemprop_multitask_cw_ptr_v3.yaml")
print(f"Reverse transform: norm.ppf(ptr_pred, {THRESHOLD}, {STD})")
