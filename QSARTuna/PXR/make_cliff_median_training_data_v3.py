"""
Activity-cliff median smoothing v3.

Uses the expanded cliff file (Compounds_with_activity_cliff_2.csv) built
with a lower similarity threshold (0.95 vs original 0.97). Applies
inactive-only series smoothing: only series where ALL members have
pEC50 < 3.5 are replaced with the series median. Mixed series (any
member pEC50 >= 3.5) are left completely untouched.

Cliff file stats (vs v2 / 0.97 threshold):
  Pairs     : 305  (vs 203)
  Compounds : 313  (vs 184)
  Series    : 100  (vs 61)
  Inactive series : 24  (106 compounds)  -> 93 compounds changed
  Mixed series    : 76  (207 compounds)  -> untouched

Output:
  train_cliff_median_v3_inactive_only.csv
"""

from collections import defaultdict

import numpy as np
import pandas as pd

CLIFF_CSV  = "/home/spal/QSARtuna/notebooks/Compounds_with_activity_cliff_2.csv"
BASE_CSV   = "/home/spal/dockerimages/QSARTuna/PXR/train_multitask_weighted.csv"
OUT_RAW    = "/home/spal/dockerimages/QSARTuna/PXR/train_cliff_median_v3_inactive_only.csv"

INACTIVE_THRESHOLD = 3.5
INACTIVE_WEIGHT    = 2.5


def find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

def union(parent, a, b):
    parent[find(parent, a)] = find(parent, b)


cliff = pd.read_csv(CLIFF_CSV)
base  = pd.read_csv(BASE_CSV)
pec50_lookup = base.set_index("Molecule Name")["pEC50"]

# ── Build connected components ────────────────────────────────────────────────
all_ids = set(cliff["Molecule Name Ref"]) | set(cliff["Molecule Name Query"])
parent  = {c: c for c in all_ids}
for _, row in cliff.iterrows():
    union(parent, row["Molecule Name Ref"], row["Molecule Name Query"])

series = defaultdict(list)
for cid in all_ids:
    series[find(parent, cid)].append(cid)

# ── Classify series ───────────────────────────────────────────────────────────
all_inactive_series = {}
mixed_series        = {}

for root, members in series.items():
    pec50s = [pec50_lookup[m] for m in members if m in pec50_lookup]
    if all(p < INACTIVE_THRESHOLD for p in pec50s):
        all_inactive_series[root] = members
    else:
        mixed_series[root] = members

sizes_inactive = sorted([len(v) for v in all_inactive_series.values()], reverse=True)
print(f"Cliff pairs             : {len(cliff)}")
print(f"Unique compounds        : {len(all_ids)}")
print(f"Total series            : {len(series)}")
print(f"All-inactive series     : {len(all_inactive_series)}  "
      f"({sum(len(v) for v in all_inactive_series.values())} compounds)")
print(f"  Sizes                 : {sizes_inactive}")
print(f"Mixed series (>=3.5)    : {len(mixed_series)}  "
      f"({sum(len(v) for v in mixed_series.values())} compounds) — untouched")

# ── Compute series medians and replacement map ────────────────────────────────
series_median = {}
changed_log   = []

for root, members in all_inactive_series.items():
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
                    "Molecule Name"     : m,
                    "pEC50_original"    : round(float(orig), 4),
                    "pEC50_smoothed"    : round(median, 4),
                    "series_size"       : len(members),
                    "series_range"      : f"{min(pec50s):.3f}–{max(pec50s):.3f}",
                })

print(f"\nCompounds with changed pEC50 : {len(changed_log)}")
print(f"Compounds unchanged          : {len(all_ids) - len(changed_log)}\n")

# ── Apply smoothing ───────────────────────────────────────────────────────────
smoothed = base.copy()
smoothed["pEC50"] = smoothed.apply(
    lambda row: series_median.get(row["Molecule Name"], row["pEC50"]),
    axis=1,
)
smoothed["weight"] = np.where(
    smoothed["pEC50"] < INACTIVE_THRESHOLD, INACTIVE_WEIGHT, 1.0
)
assert len(smoothed) == len(base)

smoothed[["Molecule Name", "SMILES", "pEC50", "pEC50_counter", "weight"]].to_csv(
    OUT_RAW, index=False
)
print(f"Saved: {OUT_RAW}  ({len(smoothed)} rows)")

# ── Summary ───────────────────────────────────────────────────────────────────
df_log = pd.DataFrame(changed_log)
df_log["delta"] = (df_log["pEC50_smoothed"] - df_log["pEC50_original"]).abs()
print(f"\nTop 15 largest pEC50 changes:")
print(df_log.sort_values("delta", ascending=False).head(15).to_string(index=False))

print(f"\npEC50 distribution — original vs smoothed:")
print(pd.DataFrame({
    "original": base["pEC50"],
    "smoothed": smoothed["pEC50"],
}).describe().round(3).to_string())

print(f"\nWeight distribution after smoothing:")
print(smoothed["weight"].value_counts().to_string())
