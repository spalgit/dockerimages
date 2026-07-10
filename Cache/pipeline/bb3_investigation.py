"""Investigate the dominant BB3 shared by 5,329 hits: real pharmacophore feature or promiscuity artifact?"""

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, FilterCatalog
from rdkit.Chem.FilterCatalog import FilterCatalogParams

TRAIN_PATH = "PGK2_train_labeled.parquet"
DEDUPED_PATH = "PGK2_selection_deduped.parquet"
BB_DIR = "../OpeDELLibrary/BBids_SMILES"

HIT_COUNT_PGK2_MIN = 3
HIT_COUNT_NTC_MAX = 0
HIT_HISTORIC_HITS_MAX = 5

train = pd.read_parquet(TRAIN_PATH)
hits = train[train["LABEL"] == 1].copy()
parts = hits["compound"].str.split("-", expand=True)
hits["library"], hits["BB1"], hits["BB2"], hits["BB3"] = (
    parts[0],
    parts[1],
    parts[2],
    parts[3],
)

# --- identify the dominant (library, BB3) group ---
grp = hits.groupby(["library", "BB3"]).size().sort_values(ascending=False)
top_library, top_bb3 = grp.index[0]
print(f"Dominant group: library={top_library}, BB3={top_bb3}, hits in this group={grp.iloc[0]}")
print("\nTop 5 (library, BB3) groups among hits:")
print(grp.head(5))

# --- look up its SMILES ---
bb_csv = f"{BB_DIR}/{top_library}.csv"
bb_df = pd.read_csv(bb_csv)
bb_df["BB3_ID"] = bb_df["BB3_ID"].astype(str).str.replace(r"\.0$", "", regex=True)
match = bb_df[bb_df["BB3_ID"] == str(top_bb3)]
bb3_smiles = match["BB3_SMILES"].iloc[0] if len(match) else None
print(f"\nBB3 SMILES ({top_library}, ID={top_bb3}): {bb3_smiles}")

if bb3_smiles:
    mol = Chem.MolFromSmiles(bb3_smiles)
    if mol:
        print(f"MolWt: {Descriptors.MolWt(mol):.1f}, LogP: {Descriptors.MolLogP(mol):.2f}, "
              f"HBD: {Descriptors.NumHDonors(mol)}, HBA: {Descriptors.NumHAcceptors(mol)}, "
              f"RotBonds: {Descriptors.NumRotatableBonds(mol)}, RingCount: {Descriptors.RingCount(mol)}")

        params = FilterCatalogParams()
        params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
        catalog = FilterCatalog.FilterCatalog(params)
        matches = catalog.GetMatches(mol)
        if matches:
            print(f"PAINS matches: {[m.GetDescription() for m in matches]}")
        else:
            print("No PAINS substructure alerts.")

# --- background enrichment: is this BB3 generally hot, or hit-specific? ---
dedup = pd.read_parquet(DEDUPED_PATH)
dparts = dedup["compound"].str.split("-", expand=True)
dedup["library"], dedup["BB1"], dedup["BB2"], dedup["BB3"] = (
    dparts[0],
    dparts[1],
    dparts[2],
    dparts[3],
)
is_hit_full = (
    (dedup["count_PGK2"] >= HIT_COUNT_PGK2_MIN)
    & (dedup["count_NTC"] <= HIT_COUNT_NTC_MAX)
    & (dedup["historic_hits"] < HIT_HISTORIC_HITS_MAX)
)
dedup["LABEL"] = is_hit_full.astype(int)

subset = dedup[(dedup["library"] == top_library) & (dedup["BB3"] == top_bb3)]
overall_hit_rate = dedup["LABEL"].mean()
subset_hit_rate = subset["LABEL"].mean()
print(f"\nTotal screened compounds sharing this (library, BB3): {len(subset)}")
print(f"Hit rate within this BB3 group: {subset_hit_rate:.2%}")
print(f"Overall background hit rate (all compounds): {overall_hit_rate:.2%}")
print(f"Enrichment ratio: {subset_hit_rate / overall_hit_rate:.1f}x")

print("\ncount_PGK2 / count_NTC / count_PGK2_with_inhibitor / historic_hits stats for ALL compounds with this BB3 (not just hits):")
print(subset[["count_PGK2", "count_PGK2_with_inhibitor", "count_NTC", "zscore_PGK2", "historic_hits"]].describe())

print("\nSame stats, split by hit/non-hit within this BB3 group:")
print(subset.groupby("LABEL")[["count_PGK2", "count_PGK2_with_inhibitor", "count_NTC", "historic_hits"]].mean())

# --- how many distinct BB1/BB2 partners does this BB3 pair with among hits? ---
bb3_hits = hits[(hits["library"] == top_library) & (hits["BB3"] == top_bb3)]
print(f"\nAmong hits with this BB3: {bb3_hits['BB1'].nunique()} distinct BB1 partners, "
      f"{bb3_hits['BB2'].nunique()} distinct BB2 partners (out of {len(bb3_hits)} hits)")
