"""
Activity-cliff median smoothing for PXR training data.

Strategy:
  1. Load the 203 activity-cliff pairs (similarity >= 0.97, |ΔpEC50| > 0.5).
  2. Build connected components (series) using union-find.
     A series = a group of compounds all highly similar to each other.
  3. For each series, replace every member's pEC50 with the median pEC50
     of the whole series.  This removes the contradictory training signal
     produced by near-identical molecules with very different labels.
  4. Recompute sample weights on the smoothed pEC50 values
     (inactives <3.5 → weight=2.5; all others → weight=1.0).
  5. Produce two output CSVs:
       train_cliff_median.csv         — raw smoothed pEC50 (same format as
                                        train_multitask_weighted.csv)
       train_cliff_median_ptr_v3.csv  — PTR v3 transform applied on top
                                        (threshold=5.0, std=0.55), ready for
                                        pxr_chemprop_multitask_cw_ptr_v3.yaml

Inputs:
  - Compounds_with_activity_cliff.csv
  - train_multitask_weighted.csv  (source of pEC50_counter + weights)

Outputs (written to /home/spal/dockerimages/QSARTuna/PXR/):
  - train_cliff_median.csv
  - train_cliff_median_ptr_v3.csv
"""

from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import norm

# ── Paths ─────────────────────────────────────────────────────────────────────
CLIFF_CSV  = "/home/spal/QSARtuna/notebooks/Compounds_with_activity_cliff.csv"
BASE_CSV   = "/home/spal/dockerimages/QSARTuna/PXR/train_multitask_weighted.csv"
OUT_RAW    = "/home/spal/dockerimages/QSARTuna/PXR/train_cliff_median.csv"
OUT_PTR    = "/home/spal/dockerimages/QSARTuna/PXR/train_cliff_median_ptr_v3.csv"

# PTR v3 parameters (must match pxr_chemprop_multitask_cw_ptr_v3.yaml)
PTR_THRESHOLD = 5.0
PTR_STD       = 0.55
PTR_CLIP      = 1e-9

# Weight rule
INACTIVE_THRESHOLD = 3.5
INACTIVE_WEIGHT    = 2.5


# ── Union-Find helpers ────────────────────────────────────────────────────────
def make_uf(ids):
    return {c: c for c in ids}

def find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

def union(parent, a, b):
    parent[find(parent, a)] = find(parent, b)


# ── PTR helpers ───────────────────────────────────────────────────────────────
def ptr_forward(y):
    result = np.full(len(y), np.nan)
    mask = np.isfinite(y)
    result[mask] = norm.cdf(y[mask], PTR_THRESHOLD, PTR_STD).clip(PTR_CLIP, 1 - PTR_CLIP)
    return result


# ── Load data ─────────────────────────────────────────────────────────────────
cliff = pd.read_csv(CLIFF_CSV)
base  = pd.read_csv(BASE_CSV)

# ── Build series via union-find ───────────────────────────────────────────────
all_ids = set(cliff["Molecule Name Ref"]) | set(cliff["Molecule Name Query"])
parent  = make_uf(all_ids)

for _, row in cliff.iterrows():
    union(parent, row["Molecule Name Ref"], row["Molecule Name Query"])

series = defaultdict(list)
for cid in all_ids:
    series[find(parent, cid)].append(cid)

print(f"Cliff pairs     : {len(cliff)}")
print(f"Unique compounds: {len(all_ids)}")
print(f"Series (components): {len(series)}")
sizes = sorted([len(v) for v in series.values()], reverse=True)
print(f"Series sizes    : {sizes}\n")

# ── Compute per-series median pEC50 ──────────────────────────────────────────
pec50_lookup = base.set_index("Molecule Name")["pEC50"]

series_median = {}   # compound_id → median pEC50 of its series
changed_log   = []

for root, members in series.items():
    pec50s = [pec50_lookup[m] for m in members if m in pec50_lookup]
    if not pec50s:
        continue
    median = float(np.median(pec50s))
    for m in members:
        if m in pec50_lookup:
            orig = pec50_lookup[m]
            series_median[m] = median
            if abs(orig - median) > 0.001:
                changed_log.append({
                    "Molecule Name": m,
                    "pEC50_original": round(orig, 4),
                    "pEC50_smoothed": round(median, 4),
                    "series_size": len(members),
                    "series_pEC50_range": f"{min(pec50s):.3f}–{max(pec50s):.3f}",
                })

print(f"Compounds with changed pEC50: {len(changed_log)}")
print(f"Compounds kept at their series median (no change): "
      f"{len(all_ids) - len(changed_log)}\n")

# ── Apply smoothing to training data ─────────────────────────────────────────
smoothed = base.copy()
smoothed["pEC50"] = smoothed.apply(
    lambda row: series_median.get(row["Molecule Name"], row["pEC50"]),
    axis=1,
)

# Recompute weights based on smoothed pEC50
smoothed["weight"] = np.where(smoothed["pEC50"] < INACTIVE_THRESHOLD,
                               INACTIVE_WEIGHT, 1.0)

assert len(smoothed) == len(base), "Row count changed — something went wrong"

# ── Save raw smoothed CSV ─────────────────────────────────────────────────────
smoothed[["Molecule Name", "SMILES", "pEC50", "pEC50_counter", "weight"]].to_csv(
    OUT_RAW, index=False
)
print(f"Saved: {OUT_RAW}  ({len(smoothed)} rows)")

# ── Apply PTR v3 transform ────────────────────────────────────────────────────
ptr = smoothed.copy()
ptr["pEC50"]         = ptr_forward(ptr["pEC50"].values)
ptr["pEC50_counter"] = ptr_forward(ptr["pEC50_counter"].values)

ptr[["Molecule Name", "SMILES", "pEC50", "pEC50_counter", "weight"]].to_csv(
    OUT_PTR, index=False
)
print(f"Saved: {OUT_PTR}  ({len(ptr)} rows)")

# ── Summary of changes ────────────────────────────────────────────────────────
df_log = pd.DataFrame(changed_log).sort_values("pEC50_original")
print(f"\nTop changes (largest pEC50 shift):")
df_log["delta"] = (df_log["pEC50_smoothed"] - df_log["pEC50_original"]).abs()
print(df_log.sort_values("delta", ascending=False).head(20).to_string(index=False))

print(f"\npEC50 distribution — original vs smoothed:")
print(pd.DataFrame({
    "original": base["pEC50"],
    "smoothed": smoothed["pEC50"],
}).describe().round(3).to_string())

print(f"\nWeight distribution after smoothing:")
print(smoothed["weight"].value_counts().to_string())
print(f"\nUse 'train_cliff_median_ptr_v3.csv' in pxr_chemprop_multitask_cw_ptr_v3.yaml")
print(f"Or  'train_cliff_median.csv' for direct pEC50 regression.")
