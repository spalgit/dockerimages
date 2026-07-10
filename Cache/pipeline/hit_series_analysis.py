"""Characterize chemical matter among PGK2 DEL hits: synthon-level series vs. structural clusters."""

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina

TRAIN_PATH = "PGK2_train_labeled.parquet"

df = pd.read_parquet(TRAIN_PATH)
hits = df[df["LABEL"] == 1].copy()
print(f"Total hits: {len(hits)}")

# --- Synthon-level series: how many hits share 2 of 3 building blocks? ---
parts = hits["compound"].str.split("-", expand=True)
hits["library"], hits["BB1"], hits["BB2"], hits["BB3"] = (
    parts[0],
    parts[1],
    parts[2],
    parts[3],
)

print("\n--- Synthon-level series (2 of 3 BBs fixed, 3rd varies) ---")
for fixed in [
    ("library", "BB1", "BB2"),
    ("library", "BB1", "BB3"),
    ("library", "BB2", "BB3"),
]:
    grp = hits.groupby(list(fixed)).size()
    series_groups = grp[grp > 1]
    print(
        f"Fixed {fixed[1:]}: {len(series_groups)} series-groups, "
        f"covering {series_groups.sum()} hits (largest group: {series_groups.max() if len(series_groups) else 0})"
    )

print("\n--- Single-BB co-occurrence (any one BB shared) ---")
for fixed in [("library", "BB1"), ("library", "BB2"), ("library", "BB3")]:
    grp = hits.groupby(list(fixed)).size()
    series_groups = grp[grp > 1]
    print(
        f"Fixed {fixed[1:]}: {len(series_groups)} groups, "
        f"covering {series_groups.sum()} hits (largest group: {series_groups.max() if len(series_groups) else 0})"
    )

# --- Structural series via Bemis-Murcko scaffold (O(n), avoids the O(n^2)
# pairwise-Tanimoto memory blowup that OOM-killed this on a dense DEL
# similarity graph where most hits share one fixed central scaffold) ---
print("\n--- Bemis-Murcko scaffold grouping ---")
from rdkit.Chem.Scaffolds import MurckoScaffold

def get_scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:
        return None

hits["murcko_scaffold"] = hits["SMILES"].apply(get_scaffold)
valid = hits["murcko_scaffold"].notna()
print(f"Valid structures for scaffold analysis: {valid.sum()}")

scaffold_counts = hits.loc[valid, "murcko_scaffold"].value_counts()
n_singleton = (scaffold_counts == 1).sum()
series_counts = scaffold_counts[scaffold_counts > 1]

print(f"Distinct Murcko scaffolds: {scaffold_counts.shape[0]}")
print(f"Scaffolds seen exactly once (singleton chemotype): {n_singleton}")
print(
    f"Scaffolds seen >1 time (series): {len(series_counts)}, "
    f"covering {series_counts.sum()} hits ({series_counts.sum() / valid.sum():.1%} of hits)"
)
print(f"Largest scaffold-based series: {scaffold_counts.max()} hits")
print("\nTop 15 scaffolds by hit count:")
print(scaffold_counts.head(15).to_string())
