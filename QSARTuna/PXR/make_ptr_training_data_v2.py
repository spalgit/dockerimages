"""
PTR label generation v2 — threshold=3.5, std=1.00 (fixed, global)

Key differences from v1:
  - threshold=3.5  (aligns with active/inactive boundary used by LGBM gate)
  - std=1.00       (wide enough to keep full pEC50 range [1.7-7.5] recoverable)
  - No per-compound or per-class std; single fixed value for all compounds
  - No STD_FLOOR needed (std=1.00 already spans the full range)

Forward transform (training labels):
    PTR(y_i) = norm.cdf(y_i, 3.5, 1.00)

Reverse transform (inference):
    pEC50_pred = norm.ppf(ptr_pred, 3.5, 1.00)

PTR coverage check:
    pEC50=1.7  -> PTR=0.036   (recoverable)
    pEC50=3.5  -> PTR=0.500   (threshold midpoint)
    pEC50=5.0  -> PTR=0.933
    pEC50=6.5  -> PTR=0.999
    pEC50=7.5  -> PTR=1.000   (clips to 1-1e-9)

Inputs:
  - processed_Openadmet_REAL_PXR_train_AND_test_main_with_side_info_AND_counter_screen.csv
  - train_multitask_weighted.csv  (provides weight column and compound set)

Outputs:
  - train_ptr_v2.csv  (drop-in replacement for train_multitask_weighted.csv)
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

# ── Parameters ────────────────────────────────────────────────────────────────
THRESHOLD = 3.5
STD       = 1.00
PTR_CLIP  = 1e-9

SRC_FILE  = ("/home/spal/dockerimages/QSARTuna/PXR_For_QSARTuna_Modelling"
             "/processed_Openadmet_REAL_PXR_train_AND_test_main_with_side_info"
             "_AND_counter_screen.csv")
BASE_FILE = "/home/spal/dockerimages/QSARTuna/PXR/train_multitask_weighted.csv"
OUT_FILE  = "/home/spal/dockerimages/QSARTuna/PXR/train_ptr_v2.csv"


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

# Merge counter-screen values from src onto base (base has primary pEC50 already)
merged = base.merge(
    src[["Molecule Name", "pEC50_counter"]],
    on="Molecule Name",
    how="left",
    suffixes=("", "_src"),
)
# Use counter from src if base counter is NaN
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
print(f"PTR v2 parameters: threshold={THRESHOLD}, std={STD}")
print(f"Saved: {OUT_FILE}  ({len(out)} rows)")
print()
print("PTR label (pEC50) distribution:")
print(out["pEC50"].describe().round(4).to_string())
print()
print("PTR label (pEC50_counter) distribution:")
print(out["pEC50_counter"].describe().round(4).to_string())
print()

# Round-trip sanity check
y_orig = merged["pEC50"].values
ptr_vals = merged["pEC50_ptr"].values
recovered = ptr_reverse(ptr_vals)
finite = np.isfinite(y_orig) & np.isfinite(recovered)
rt_mae = np.mean(np.abs(y_orig[finite] - recovered[finite]))
print(f"Round-trip MAE (PTR -> reverse -> pEC50): {rt_mae:.6f} log units")
print()
print(f"PTR coverage spot-check:")
for pec in [1.7, 2.5, 3.5, 4.5, 5.0, 6.0, 7.0]:
    p = norm.cdf(pec, THRESHOLD, STD)
    r = ptr_reverse(p)
    print(f"  pEC50={pec:.1f} -> PTR={p:.4f} -> recovered={r:.4f}")
print()
print(f"Use 'train_ptr_v2.csv' in pxr_chemprop_multitask_cw_ptr_v2.yaml")
print(f"Reverse transform: norm.ppf(ptr_pred, {THRESHOLD}, {STD})")
