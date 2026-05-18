"""
Add counter-screen selectivity weights to the PXR training CSV.

Formula: inverse linear map of pEC50_counter → [MIN_WEIGHT, MAX_WEIGHT]
  - High pEC50_counter (strong counter-screen hit) → low weight (0.5)
  - Low pEC50_counter (weak counter-screen hit)   → high weight (2.0)
  - No counter data (NaN)                          → neutral weight (1.0)

Matches exactly the compute_weights() function in:
  chemprop_pxr_pec50_chemeleon_counter_weight_finalonly.py
"""

from pathlib import Path
import numpy as np
import pandas as pd

IN_PATH  = Path(
    "/home/spal/dockerimages/QSARTuna/PXR_For_QSARTuna_Modelling/"
    "processed_Openadmet_REAL_PXR_train_AND_test_main_with_side_info_"
    "AND_counter_screen_weighted.csv"
)
OUT_PATH = Path(
    "/home/spal/dockerimages/QSARTuna/PXR_For_QSARTuna_Modelling/"
    "processed_Openadmet_REAL_PXR_train_AND_test_main_with_side_info_"
    "AND_counter_screen_CW_weighted.csv"
)

MIN_WEIGHT     = 0.5
MAX_WEIGHT     = 2.0
NEUTRAL_WEIGHT = 1.0
COUNTER_COL    = "pEC50_counter"
WEIGHT_COL     = "weight_counter_screen"

df = pd.read_csv(IN_PATH)
print(f"Loaded {len(df)} rows from:\n  {IN_PATH}")

counter_values = df[COUNTER_COL].values
weights        = np.full(len(counter_values), NEUTRAL_WEIGHT, dtype=float)
has_counter    = ~np.isnan(counter_values)
vals           = counter_values[has_counter]
c_min, c_max   = vals.min(), vals.max()
weights[has_counter] = MIN_WEIGHT + (MAX_WEIGHT - MIN_WEIGHT) * (
    (c_max - vals) / (c_max - c_min)
)

df[WEIGHT_COL] = weights

n_with = has_counter.sum()
print(f"\nCounter screen available : {n_with}/{len(df)} ({100*n_with/len(df):.1f}%)")
print(f"Weight range (with data) : {weights[has_counter].min():.4f} – {weights[has_counter].max():.4f}")
print(f"Weight for NaN rows      : {NEUTRAL_WEIGHT}")

df.to_csv(OUT_PATH, index=False)
print(f"\nSaved to:\n  {OUT_PATH}")
print(f"Columns: {df.columns.tolist()}")
