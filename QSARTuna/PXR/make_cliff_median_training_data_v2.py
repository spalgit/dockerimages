"""
Activity-cliff median smoothing v2 — with --inactive_only flag.

Adds a mode that only smooths series where ALL members have pEC50 < 3.5
(pure noise in the inactive zone). Mixed series containing any active
compound (pEC50 >= 3.5) are left completely untouched, preserving genuine
SAR signal at the activity boundary.

Usage:
    # Smooth all cliff series (same as v1):
    python make_cliff_median_training_data_v2.py

    # Smooth only all-inactive series (recommended):
    python make_cliff_median_training_data_v2.py --inactive_only

Outputs (written to /home/spal/dockerimages/QSARTuna/PXR/):
    Without --inactive_only:
        train_cliff_median_v2.csv
        train_cliff_median_v2_ptr_v3.csv

    With --inactive_only:
        train_cliff_median_v2_inactive_only.csv
        train_cliff_median_v2_inactive_only_ptr_v3.csv
"""

import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import norm

# ── Paths ─────────────────────────────────────────────────────────────────────
CLIFF_CSV = "/home/spal/QSARtuna/notebooks/Compounds_with_activity_cliff.csv"
BASE_CSV  = "/home/spal/dockerimages/QSARTuna/PXR/train_multitask_weighted.csv"
OUT_DIR   = "/home/spal/dockerimages/QSARTuna/PXR"

# PTR v3 parameters
PTR_THRESHOLD = 5.0
PTR_STD       = 0.55
PTR_CLIP      = 1e-9

INACTIVE_THRESHOLD = 3.5
INACTIVE_WEIGHT    = 2.5


# ── Union-Find ────────────────────────────────────────────────────────────────
def make_uf(ids):
    return {c: c for c in ids}

def find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

def union(parent, a, b):
    parent[find(parent, a)] = find(parent, b)


# ── PTR transform ─────────────────────────────────────────────────────────────
def ptr_forward(y):
    result = np.full(len(y), np.nan)
    mask = np.isfinite(y)
    result[mask] = norm.cdf(y[mask], PTR_THRESHOLD, PTR_STD).clip(PTR_CLIP, 1 - PTR_CLIP)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inactive_only", action="store_true",
        help="Only smooth series where ALL members have pEC50 < 3.5. "
             "Mixed series (any member pEC50 >= 3.5) are left untouched.",
    )
    args = parser.parse_args()

    # ── Load data ─────────────────────────────────────────────────────────────
    cliff = pd.read_csv(CLIFF_CSV)
    base  = pd.read_csv(BASE_CSV)
    pec50_lookup  = base.set_index("Molecule Name")["pEC50"]

    # ── Build connected components ────────────────────────────────────────────
    all_ids = set(cliff["Molecule Name Ref"]) | set(cliff["Molecule Name Query"])
    parent  = make_uf(all_ids)
    for _, row in cliff.iterrows():
        union(parent, row["Molecule Name Ref"], row["Molecule Name Query"])

    series = defaultdict(list)
    for cid in all_ids:
        series[find(parent, cid)].append(cid)

    # ── Classify series ───────────────────────────────────────────────────────
    all_inactive_series  = {}
    mixed_series         = {}

    for root, members in series.items():
        pec50s = [pec50_lookup[m] for m in members if m in pec50_lookup]
        if all(p < INACTIVE_THRESHOLD for p in pec50s):
            all_inactive_series[root] = members
        else:
            mixed_series[root] = members

    print(f"Total series            : {len(series)}")
    print(f"All-inactive series     : {len(all_inactive_series)}  "
          f"({sum(len(v) for v in all_inactive_series.values())} compounds)")
    print(f"Mixed series (>=3.5)    : {len(mixed_series)}  "
          f"({sum(len(v) for v in mixed_series.values())} compounds)")
    print()

    if args.inactive_only:
        series_to_smooth = all_inactive_series
        label = "inactive_only"
        print("Mode: smoothing ALL-INACTIVE series only. Mixed series untouched.\n")
    else:
        series_to_smooth = {**all_inactive_series, **mixed_series}
        label = ""
        print("Mode: smoothing ALL cliff series.\n")

    # ── Compute median per series and build replacement map ───────────────────
    series_median = {}
    changed_log   = []

    for root, members in series_to_smooth.items():
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
                        "Molecule Name"    : m,
                        "pEC50_original"   : round(float(orig), 4),
                        "pEC50_smoothed"   : round(median, 4),
                        "series_size"      : len(members),
                        "series_pEC50_range": f"{min(pec50s):.3f}–{max(pec50s):.3f}",
                        "all_inactive"     : all(p < INACTIVE_THRESHOLD for p in pec50s),
                    })

    print(f"Compounds with changed pEC50 : {len(changed_log)}")
    print(f"Compounds unchanged          : {len(all_ids) - len(changed_log)}\n")

    # ── Apply smoothing ───────────────────────────────────────────────────────
    smoothed = base.copy()
    smoothed["pEC50"] = smoothed.apply(
        lambda row: series_median.get(row["Molecule Name"], row["pEC50"]),
        axis=1,
    )
    smoothed["weight"] = np.where(
        smoothed["pEC50"] < INACTIVE_THRESHOLD, INACTIVE_WEIGHT, 1.0
    )
    assert len(smoothed) == len(base)

    # ── Output filenames ──────────────────────────────────────────────────────
    suffix   = f"_v2{'_' + label if label else ''}"
    out_raw  = f"{OUT_DIR}/train_cliff_median{suffix}.csv"
    out_ptr  = f"{OUT_DIR}/train_cliff_median{suffix}_ptr_v3.csv"

    smoothed[["Molecule Name", "SMILES", "pEC50", "pEC50_counter", "weight"]].to_csv(
        out_raw, index=False
    )
    print(f"Saved: {out_raw}")

    ptr = smoothed.copy()
    ptr["pEC50"]         = ptr_forward(ptr["pEC50"].values)
    ptr["pEC50_counter"] = ptr_forward(ptr["pEC50_counter"].values)
    ptr[["Molecule Name", "SMILES", "pEC50", "pEC50_counter", "weight"]].to_csv(
        out_ptr, index=False
    )
    print(f"Saved: {out_ptr}")

    # ── Summary ───────────────────────────────────────────────────────────────
    df_log = pd.DataFrame(changed_log)
    if not df_log.empty:
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

    print(f"\nNext step — update the resource line in your YAML:")
    print(f"  resource: {out_ptr}   (PTR v3)")
    print(f"  resource: {out_raw}   (direct pEC50 regression)")


if __name__ == "__main__":
    main()
