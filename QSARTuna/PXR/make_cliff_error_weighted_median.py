"""
Activity-cliff error-weighted median smoothing.

Uses Compounds_with_activity_cliff.csv (0.97 similarity threshold, 184 compounds).
Applies error-aware smoothing based on pEC50_std.error from train.csv:

  % error = pEC50_std.error / pEC50 * 100

Rules:
  1. Series (3+ members): compounds with % error > 5% are excluded from the
     median calculation. All series members receive the resulting median.
     If ALL members exceed 5% error, the median of all is used as fallback.

  2. Pairs (2 members): if exactly one member has % error > 5%, the high-error
     member is removed from the training set entirely. The low-error member
     keeps its own pEC50 unchanged.

  3. Otherwise (pairs where both are low-error, or both are high-error):
     both members receive the mean of their pEC50 values.

Weight assignment (matching v3 convention):
  weight = 2.5 if pEC50 < 3.5 else 1.0

Output:
  train_cliff_error_weighted_median.csv
"""

from collections import defaultdict

import numpy as np
import pandas as pd

CLIFF_CSV       = "/home/spal/QSARtuna/notebooks/Compounds_with_activity_cliff.csv"
TRAIN_CSV       = "/home/spal/dockerimages/QSARTuna/PXR/train.csv"
BASE_CSV        = "/home/spal/dockerimages/QSARTuna/PXR/train_multitask_weighted.csv"
OUT_CSV         = "/home/spal/dockerimages/QSARTuna/PXR/train_cliff_error_weighted_median.csv"

ERROR_THRESHOLD    = 5.0   # % error threshold
INACTIVE_THRESHOLD = 3.5
INACTIVE_WEIGHT    = 2.5


# ── Load data ─────────────────────────────────────────────────────────────────
cliff = pd.read_csv(CLIFF_CSV)
train = pd.read_csv(TRAIN_CSV)
base  = pd.read_csv(BASE_CSV)

# Build per-compound lookups from train
train_idx       = train.set_index("Molecule Name")
pec50_lookup    = train_idx["pEC50"]
error_lookup    = train_idx["pEC50_std.error (-log10(molarity))"]
pct_err_lookup  = (error_lookup / pec50_lookup * 100)   # % error per compound


# ── Build connected components (union-find) ───────────────────────────────────
def find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

def union(parent, a, b):
    parent[find(parent, a)] = find(parent, b)

all_ids = set(cliff["Molecule Name Ref"]) | set(cliff["Molecule Name Query"])
parent  = {c: c for c in all_ids}
for _, row in cliff.iterrows():
    union(parent, row["Molecule Name Ref"], row["Molecule Name Query"])

series = defaultdict(list)
for cid in all_ids:
    series[find(parent, cid)].append(cid)

print(f"Cliff pairs          : {len(cliff)}")
print(f"Unique compounds     : {len(all_ids)}")
print(f"Total components     : {len(series)}")
sizes = sorted([len(v) for v in series.values()])
from collections import Counter
print(f"Component sizes      : {Counter(sizes)}\n")


# ── Apply error-weighted smoothing ────────────────────────────────────────────
replacement = {}   # Molecule Name -> new pEC50
to_remove   = set()  # Molecule Names to drop from output
changed_log = []

for root, members in series.items():
    pec50s   = {m: float(pec50_lookup[m])   for m in members if m in pec50_lookup}
    pct_errs = {m: float(pct_err_lookup[m]) for m in members if m in pct_err_lookup}
    if not pec50s:
        continue

    n = len(members)

    if n >= 3:
        # --- Rule 1: series -------------------------------------------------------
        low_err = [m for m in members if pct_errs.get(m, 0) <= ERROR_THRESHOLD]
        if low_err:
            median_val = float(np.median([pec50s[m] for m in low_err]))
            excluded   = [m for m in members if pct_errs.get(m, 0) > ERROR_THRESHOLD]
        else:
            # Fallback: all exceed threshold — use median of all
            median_val = float(np.median(list(pec50s.values())))
            excluded   = []
        for m in members:
            if m in pec50s:
                orig = pec50s[m]
                replacement[m] = median_val
                if abs(orig - median_val) > 0.001:
                    changed_log.append({
                        "Molecule Name"  : m,
                        "pEC50_original" : round(orig, 4),
                        "pEC50_new"      : round(median_val, 4),
                        "pct_error"      : round(pct_errs.get(m, float("nan")), 2),
                        "excluded_from_median": m in excluded,
                        "series_size"    : n,
                        "rule"           : "series_median",
                    })

    else:
        # n == 2: pair
        m1, m2 = members[0], members[1]
        if m1 not in pec50s or m2 not in pec50s:
            continue
        e1 = pct_errs.get(m1, 0) > ERROR_THRESHOLD
        e2 = pct_errs.get(m2, 0) > ERROR_THRESHOLD

        if e1 and not e2:
            # --- Rule 2: m1 high-error → remove m1, keep m2 as-is ----------------
            to_remove.add(m1)
            changed_log.append({
                "Molecule Name"  : m1,
                "pEC50_original" : round(pec50s[m1], 4),
                "pEC50_new"      : "REMOVED",
                "pct_error"      : round(pct_errs.get(m1, float("nan")), 2),
                "excluded_from_median": False,
                "series_size"    : n,
                "rule"           : "pair_remove_high_error",
            })
        elif e2 and not e1:
            # --- Rule 2: m2 high-error → remove m2, keep m1 as-is ----------------
            to_remove.add(m2)
            changed_log.append({
                "Molecule Name"  : m2,
                "pEC50_original" : round(pec50s[m2], 4),
                "pEC50_new"      : "REMOVED",
                "pct_error"      : round(pct_errs.get(m2, float("nan")), 2),
                "excluded_from_median": False,
                "series_size"    : n,
                "rule"           : "pair_remove_high_error",
            })
        else:
            # --- Rule 3: both low-error or both high-error → mean -----------------
            new_val = float(np.mean([pec50s[m1], pec50s[m2]]))
            rule    = "pair_mean"
            for m in (m1, m2):
                orig = pec50s[m]
                replacement[m] = new_val
                if abs(orig - new_val) > 0.001:
                    changed_log.append({
                        "Molecule Name"  : m,
                        "pEC50_original" : round(orig, 4),
                        "pEC50_new"      : round(new_val, 4),
                        "pct_error"      : round(pct_errs.get(m, float("nan")), 2),
                        "excluded_from_median": False,
                        "series_size"    : n,
                        "rule"           : rule,
                    })


# ── Build output dataframe ────────────────────────────────────────────────────
out = base.copy()
out["pEC50"] = out.apply(
    lambda row: replacement.get(row["Molecule Name"], row["pEC50"]),
    axis=1,
)
out = out[~out["Molecule Name"].isin(to_remove)].copy()
out["weight"] = np.where(out["pEC50"] < INACTIVE_THRESHOLD, INACTIVE_WEIGHT, 1.0)

out[["Molecule Name", "SMILES", "pEC50", "pEC50_counter", "weight"]].to_csv(
    OUT_CSV, index=False
)
print(f"Removed {len(to_remove)} high-error compounds from pairs")
print(f"Saved: {OUT_CSV}  ({len(out)} rows, was {len(base)})")


# ── Summary ───────────────────────────────────────────────────────────────────
df_log = pd.DataFrame(changed_log)
modified = df_log[df_log["rule"] != "pair_remove_high_error"]
print(f"\nCompounds removed            : {len(to_remove)}")
print(f"Compounds with changed pEC50 : {len(modified)}")
print(f"Compounds unchanged          : {len(all_ids) - len(modified) - len(to_remove)}")

if not df_log.empty:
    print(f"\nChanges by rule:\n{df_log['rule'].value_counts().to_string()}")
    df_log["delta"] = pd.to_numeric(df_log["pEC50_new"], errors="coerce").sub(df_log["pEC50_original"]).abs()
    print(f"\nTop 15 largest pEC50 changes:")
    print(df_log.sort_values("delta", ascending=False).head(15).to_string(index=False))

print(f"\npEC50 distribution — original vs smoothed:")
print(pd.DataFrame({
    "original": base["pEC50"],
    "smoothed": out["pEC50"],
}).describe().round(3).to_string())

print(f"\nWeight distribution:")
print(out["weight"].value_counts().to_string())
