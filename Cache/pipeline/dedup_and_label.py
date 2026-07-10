"""Dedupe PGK2_selection.parquet by SMILES and label hits (orthosteric + allosteric).

Aggregation scheme per the challenge's "Deduplicate Selection" wiki page:
  count_*    -> sum
  zscore_*   -> Stouffer's Z: sum(z) / sqrt(n)
  historic_hits -> max
  compound, SMILES -> first
"""

import numpy as np
import pandas as pd
from rdkit import Chem

RAW_PATH = "../PGK2_selection.parquet"
DEDUP_PATH = "PGK2_selection_deduped.parquet"
LABELED_PATH = "PGK2_train_labeled.parquet"

HIT_COUNT_PGK2_MIN = 3
HIT_COUNT_NTC_MAX = 0
HIT_HISTORIC_HITS_MAX = 5
NEGATIVE_RATIO = 2  # negatives sampled per hit


def is_valid_smiles(smiles: str) -> bool:
    if not isinstance(smiles, str) or len(smiles) <= 10:
        return False
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    return any(atom.GetSymbol() == "C" for atom in mol.GetAtoms())


def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df[df["compound"].notna() & df["SMILES"].notna()]
    valid_mask = df["SMILES"].apply(is_valid_smiles)
    df = df[valid_mask]
    print(f"Cleaning: {before} -> {len(df)} rows")

    before = len(df)
    agg = df.groupby("SMILES", sort=False).agg(
        compound=("compound", "first"),
        count_PGK2=("count_PGK2", "sum"),
        count_PGK2_with_inhibitor=("count_PGK2_with_inhibitor", "sum"),
        count_NTC=("count_NTC", "sum"),
        zscore_PGK2=("zscore_PGK2", lambda z: z.sum() / np.sqrt(len(z))),
        zscore_PGK2_with_inhibitor=(
            "zscore_PGK2_with_inhibitor",
            lambda z: z.sum() / np.sqrt(len(z)),
        ),
        zscore_NTC=("zscore_NTC", lambda z: z.sum() / np.sqrt(len(z))),
        historic_hits=("historic_hits", "max"),
    ).reset_index()
    print(f"Dedup by SMILES: {before} -> {len(agg)} rows")
    return agg


def label(df: pd.DataFrame) -> pd.DataFrame:
    is_hit = (
        (df["count_PGK2"] >= HIT_COUNT_PGK2_MIN)
        & (df["count_NTC"] <= HIT_COUNT_NTC_MAX)
        & (df["historic_hits"] < HIT_HISTORIC_HITS_MAX)
    )
    df = df.copy()
    df["LABEL"] = is_hit.astype(int)
    n_hits = df["LABEL"].sum()
    print(f"Hits: {n_hits} ({n_hits / len(df):.4%} of {len(df)} unique compounds)")
    return df


def subsample(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    hits = df[df["LABEL"] == 1]
    negatives_pool = df[df["LABEL"] == 0]
    n_neg = min(len(negatives_pool), len(hits) * NEGATIVE_RATIO)
    negatives = negatives_pool.sample(n=n_neg, random_state=seed)
    out = pd.concat([hits, negatives], ignore_index=True).sample(
        frac=1, random_state=seed
    )
    print(f"Training set: {len(hits)} hits + {len(negatives)} negatives = {len(out)} rows")
    return out


if __name__ == "__main__":
    raw = pd.read_parquet(RAW_PATH)
    print(f"Loaded raw selection: {raw.shape}")

    deduped = dedupe(raw)
    deduped.to_parquet(DEDUP_PATH, index=False)
    print(f"Saved deduped file: {DEDUP_PATH}")

    labeled = label(deduped)
    train_set = subsample(labeled)
    train_set.to_parquet(LABELED_PATH, index=False)
    print(f"Saved labeled training set: {LABELED_PATH}")
