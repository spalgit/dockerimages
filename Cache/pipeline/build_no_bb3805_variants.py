"""Build BB3-805-excluded training variants, for the ablation experiment
comparing against the full (BB3-805-included) training data.

Removes ALL compounds carrying library `qDOS30` BB3 ID `805`
(`OC(=O)c1cnc2[nH]cnc2c1`, the dominant DEL chemotype found by
bb3_investigation.py) from the training pool entirely -- both hits (5,329)
and non-hits (8,880), 14,209 total (0.19% of the 7.46M deduped pool) --
before labeling and negative sampling. This is a clean ablation: removing
only the hits and leaving the non-hits in the negative pool would just
teach the opposite shortcut ("contains BB3-805 -> predict inactive").

Rationale (2026-07-17): BB3-805 makes up 19.3% of all labeled hits despite
being a single homogeneous chemotype, and both the ChemProp and LightGBM
ensembles independently rank BB3-805-carrying validation compounds at or
near the very top -- yet those specific compounds are confirmed non-hits
(see PGK2_VALIDATION_SUBMISSION_LOG.md). This script does NOT delete BB3-805
from the project's data -- the original PGK2_train_labeled.parquet and
PGK2_train_broad_2to1/50to50.parquet are untouched -- it builds a parallel
"no_bb3805" variant so the two can be compared head-to-head on the real
validation split, per the strategy doc's own suggested robustness check.

Produces (without touching the originals):
  - PGK2_train_labeled_no_bb3805.parquet       (for first_model_pipeline.py / LightGBM)
  - PGK2_train_broad_2to1_no_bb3805.parquet    (for chemprop, cluster-stratified negatives)
  - PGK2_train_broad_50to50_no_bb3805.parquet  (for chemprop, cluster-stratified negatives)

Reuses the cached PGK2_negative_pool_clusters.parquet unchanged: the
no-bb3805 negative pool is a strict subset of the negative pool the cache
was built from (we're only removing compounds, not adding any), so no
re-fingerprinting/reclustering is needed.

Usage:
    conda activate chemprop
    python build_no_bb3805_variants.py
"""

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dedup_and_label import label, subsample  # noqa: E402
from build_train_variants import (  # noqa: E402
    NEGATIVE_RATIO_2TO1,
    get_negative_pool_with_clusters,
    make_variant,
)

DEDUP_PATH = HERE / "PGK2_selection_deduped.parquet"
EXCLUDE_LIBRARY = "qDOS30"
EXCLUDE_BB3 = "805"

OUT_LABELED = HERE / "PGK2_train_labeled_no_bb3805.parquet"
OUT_2TO1 = HERE / "PGK2_train_broad_2to1_no_bb3805.parquet"
OUT_5050 = HERE / "PGK2_train_broad_50to50_no_bb3805.parquet"


def exclude_bb3805(df: pd.DataFrame) -> pd.DataFrame:
    parts = df["compound"].str.split("-", expand=True)
    library, bb3 = parts[0], parts[3]
    mask = (library == EXCLUDE_LIBRARY) & (bb3 == EXCLUDE_BB3)
    print(
        f"Excluding {mask.sum()} compounds carrying library={EXCLUDE_LIBRARY} "
        f"BB3={EXCLUDE_BB3} (out of {len(df)} total)"
    )
    return df[~mask].reset_index(drop=True)


def main():
    print(f"Loading deduped selection from {DEDUP_PATH}...")
    deduped = pd.read_parquet(DEDUP_PATH)
    print(f"Loaded: {deduped.shape}")

    filtered = exclude_bb3805(deduped)
    labeled = label(filtered)

    # -- LightGBM-style labeled+subsampled file (matches PGK2_train_labeled.parquet) --
    train_set = subsample(labeled)
    train_set.to_parquet(OUT_LABELED, index=False)
    print(f"Saved LightGBM-style labeled training set: {OUT_LABELED}")

    # -- ChemProp-style cluster-stratified negative sampling variants --
    hits = labeled[labeled["LABEL"] == 1].reset_index(drop=True)
    negatives_pool = labeled[labeled["LABEL"] == 0].reset_index(drop=True)
    print(f"Hits: {len(hits)}  Negative pool: {len(negatives_pool)}")

    negatives_pool = get_negative_pool_with_clusters(negatives_pool)
    cluster_labels = negatives_pool["cluster"].to_numpy()

    make_variant(
        "2:1 (no BB3-805)", hits, negatives_pool, cluster_labels,
        len(hits) * NEGATIVE_RATIO_2TO1, OUT_2TO1,
    )
    make_variant(
        "50:50 (no BB3-805)", hits, negatives_pool, cluster_labels,
        len(hits), OUT_5050,
    )


if __name__ == "__main__":
    main()
