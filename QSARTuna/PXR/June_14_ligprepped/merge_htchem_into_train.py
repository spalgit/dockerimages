"""
Merge corrected HTChem (crudes + semi-pure) compounds into the existing
PXR training set, deduplicating by canonical SMILES at every step.

Priority for the same compound across sources:
  semi-pure > crudes  (higher purity → lower corrected SE)
  existing SDF data is never replaced.

Output: f1_plus_htchem_train.csv
  Columns: SMILES, Name, pEC50, pEC50_counter, pEC50_std.error
  SMILES are written with explicit H to match the ligprepped SDF format.
"""

import pandas as pd
from rdkit import Chem

SDF_PATH     = "train_set_AND_phase_one_results_4392_ligpreped_f_1_n_1_2_3.sdf"
CRUDES_PATH  = "/home/spal/OpenAdmet_After_Phase1/pxr-challenge-train-test_train_crudes.csv"
SEMI_PATH    = "/home/spal/OpenAdmet_After_Phase1/pxr-challenge-train-test_semi_pure.csv"
EXISTING_CSV = "f1_train_4392.csv"

# ── helpers ──────────────────────────────────────────────────────────────────

def to_canon(smi):
    try:
        m = Chem.MolFromSmiles(str(smi))
        return Chem.MolToSmiles(m) if m else None
    except Exception:
        return None

def with_explicit_h(smi):
    """Return SMILES with explicit H, matching the ligprepped SDF export format."""
    try:
        m = Chem.MolFromSmiles(str(smi))
        if m is None:
            return None
        return Chem.MolToSmiles(Chem.AddHs(m))
    except Exception:
        return None

def is_numeric(val):
    try:
        float(val)
        return True
    except (TypeError, ValueError):
        return False

# ── 1. Build deduplication set from the existing training SDF ─────────────

print("Loading existing SDF …")
suppl = Chem.SDMolSupplier(SDF_PATH, removeHs=False)
seen_canon = set()
for m in suppl:
    if m is None:
        continue
    m_noh = Chem.RemoveHs(m)
    c = Chem.MolToSmiles(m_noh)
    if c:
        seen_canon.add(c)
print(f"  {len(seen_canon)} unique canonical SMILES in existing SDF")

# ── 2. Load and filter semi-pure (higher quality — process first) ──────────

print("\nLoading semi-pure CSV …")
semi = pd.read_csv(SEMI_PATH)
semi["canon"] = semi["SMILES"].apply(to_canon)

# Quality filter: corrected pEC50 must be a real number
semi["valid"] = semi["Corrected Semi-Pure pEC50 (log)"].apply(is_numeric)
semi_clean = semi[semi["valid"] & semi["canon"].notna()].copy()
print(f"  {len(semi_clean)}/{len(semi)} rows pass quality filter")

# Deduplicate against SDF
semi_new = semi_clean[~semi_clean["canon"].isin(seen_canon)].copy()
print(f"  {len(semi_clean) - len(semi_new)} already in SDF → {len(semi_new)} new compounds")

# Build rows
semi_rows = []
for _, row in semi_new.iterrows():
    semi_rows.append({
        "SMILES": with_explicit_h(row["SMILES"]),
        "Name": row["OCNT_ID"],
        "pEC50": float(row["Corrected Semi-Pure pEC50 (log)"]),
        "pEC50_counter": float("nan"),
        "pEC50_std.error": float(row["Corrected Semi-Pure pEC50 ±1 SE (log)"])
            if is_numeric(row.get("Corrected Semi-Pure pEC50 ±1 SE (log)")) else float("nan"),
        "source": "htchem_semi_pure",
    })

# Add semi-pure canon SMILES to seen set so crudes don't duplicate them
seen_canon.update(semi_new["canon"])

# ── 3. Load and filter crudes ─────────────────────────────────────────────

print("\nLoading crudes CSV …")
crudes = pd.read_csv(CRUDES_PATH)
crudes["canon"] = crudes["SMILES"].apply(to_canon)

# Quality filter A: Crude Peak Area (pA*min) >= 1.0 (LLOQ)
crudes["above_lloq"] = crudes["Crude Peak Area (pA*min)"].apply(
    lambda v: is_numeric(v) and float(v) >= 1.0
)
# Quality filter B: corrected pEC50 must be a real number
crudes["has_corrected"] = crudes["Corrected Crude pEC50 (log)"].apply(is_numeric)

crudes_clean = crudes[crudes["above_lloq"] & crudes["has_corrected"] & crudes["canon"].notna()].copy()
print(f"  {len(crudes_clean)}/{len(crudes)} rows pass LLOQ + corrected-pEC50 filter")

# Deduplicate against SDF and already-added semi-pure
crudes_new = crudes_clean[~crudes_clean["canon"].isin(seen_canon)].copy()
print(f"  {len(crudes_clean) - len(crudes_new)} already in SDF/semi-pure → {len(crudes_new)} new compounds")

crudes_rows = []
for _, row in crudes_new.iterrows():
    crudes_rows.append({
        "SMILES": with_explicit_h(row["SMILES"]),
        "Name": row["OCNT_ID"],
        "pEC50": float(row["Corrected Crude pEC50 (log)"]),
        "pEC50_counter": float("nan"),
        "pEC50_std.error": float(row["Corrected Crude pEC50 ±1 SE (log)"])
            if is_numeric(row.get("Corrected Crude pEC50 ±1 SE (log)")) else float("nan"),
        "source": "htchem_crudes",
    })

# ── 4. Load existing CSV and append new rows ───────────────────────────────

print("\nLoading existing training CSV …")
existing = pd.read_csv(EXISTING_CSV)
# Add missing columns so concat aligns properly
if "pEC50_std.error" not in existing.columns:
    existing["pEC50_std.error"] = float("nan")
if "source" not in existing.columns:
    existing["source"] = "original"
print(f"  {len(existing)} existing compounds")

new_df = pd.DataFrame(semi_rows + crudes_rows)
print(f"\nNew HTChem compounds to add: {len(new_df)} "
      f"({len(semi_rows)} semi-pure + {len(crudes_rows)} crudes)")

# Drop rows where SMILES generation failed
new_df = new_df[new_df["SMILES"].notna()]

merged = pd.concat([existing, new_df], ignore_index=True)

# ── 5. Final deduplication pass (safety net on canonical SMILES) ───────────
merged["_canon"] = merged["SMILES"].apply(to_canon)
before = len(merged)
merged = merged.drop_duplicates(subset="_canon", keep="first")
merged = merged.drop(columns="_canon")
after = len(merged)
if before != after:
    print(f"  Removed {before - after} duplicates in final pass")

# ── 6. Save ───────────────────────────────────────────────────────────────

out_cols = ["SMILES", "Name", "pEC50", "pEC50_counter", "pEC50_std.error", "source"]
out_path = f"f1_plus_htchem_train_{len(merged)}.csv"
merged[out_cols].to_csv(out_path, index=False)

print(f"\nSaved → {out_path}")
print(f"  Original : {len(existing):>5} compounds")
print(f"  Added    : {len(merged) - len(existing):>5} HTChem compounds")
print(f"  Total    : {len(merged):>5} compounds")
print(f"\npEC50 range in new compounds: "
      f"{new_df['pEC50'].min():.2f} – {new_df['pEC50'].max():.2f}")
print(f"SE range (new):              "
      f"{new_df['pEC50_std.error'].min():.3f} – {new_df['pEC50_std.error'].max():.3f}")
print("\nDone.")
