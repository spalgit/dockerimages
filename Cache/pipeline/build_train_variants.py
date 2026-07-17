"""Build PGK2 training-set variants with cluster-stratified negative
sampling, per the "net recommendation" in
`~/Cache/Literature/Iqbal_et_al_DEL_ML_Analysis_and_CACHE_Strategy.md`:

  (a) add a 50:50-balanced training variant alongside the existing 2:1
      negative:positive variant.
  (b) switch negative sampling from uniform random to cluster-stratified
      (MiniBatch K-Means on Morgan fingerprints), matching Iqbal et al.'s
      downsampling method (Methods: "Downsampling approach" — MiniBatchKMeans
      into 100 clusters, representatives drawn from each cluster).

Labeling: uses the broad (no-competitor) hit criteria from
dedup_and_label.py — count_PGK2 >= 3, count_NTC == 0, historic_hits < 5 —
per the CACHE organizers' explicit guidance that training compounds should
be selected from data tested in the absence of a competitor (i.e. the
count_PGK2 condition, not count_PGK2_with_inhibitor). This file previously
imported the stricter orthosteric criteria from label_orthosteric.py
(which additionally requires count_PGK2_with_inhibitor < 0.1 * count_PGK2,
a condition that only exists for compounds tested *with* the competitor
present) — that was a mismatch with both the organizer guidance and this
project's own 2026-07-06 decision to train on the broad orthosteric+
allosteric label. Fixed 2026-07-17; see PGK2_VALIDATION_SUBMISSION_LOG.md.

Cluster assignments for the full 7.46M-compound negative pool are cached to
PGK2_negative_pool_clusters.parquet (fingerprinting + clustering takes
~20-30 min) so additional ratio variants can be produced later without
recomputing fingerprints/clustering. Since broad-label negatives are a
subset of the previously-cached orthosteric-label negatives (the broad
label has strictly more hits, hence fewer negatives), the existing cache
remains valid here — no need to recompute.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from scipy import sparse
from sklearn.cluster import MiniBatchKMeans

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from dedup_and_label import label  # noqa: E402  (reuse existing labeling logic)

RDLogger.DisableLog("rdApp.*")

DEDUP_PATH = HERE / "PGK2_selection_deduped.parquet"
CLUSTERS_PATH = HERE / "PGK2_negative_pool_clusters.parquet"
OUT_2TO1 = HERE / "PGK2_train_broad_2to1.parquet"
OUT_5050 = HERE / "PGK2_train_broad_50to50.parquet"

N_CLUSTERS = 100  # matches Iqbal et al.'s downsampling method
FP_RADIUS = 2
FP_NBITS = 2048
CHUNK_SIZE = 200_000
SEED = 42
NEGATIVE_RATIO_2TO1 = 2


def compute_fp_chunk_csr(smiles_list, gen):
    """Sparse Morgan-FP CSR block for one chunk; skips unparseable SMILES."""
    rows, cols = [], []
    valid_mask = np.zeros(len(smiles_list), dtype=bool)
    row_i = 0
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        on_bits = np.fromiter(gen.GetFingerprint(mol).GetOnBits(), dtype=np.int32)
        rows.append(np.full(len(on_bits), row_i, dtype=np.int32))
        cols.append(on_bits)
        valid_mask[i] = True
        row_i += 1
    if row_i == 0:
        return sparse.csr_matrix((0, FP_NBITS), dtype=np.uint8), valid_mask
    row_idx = np.concatenate(rows)
    col_idx = np.concatenate(cols)
    data = np.ones(len(row_idx), dtype=np.uint8)
    mat = sparse.csr_matrix(
        (data, (row_idx, col_idx)), shape=(row_i, FP_NBITS), dtype=np.uint8
    )
    return mat, valid_mask


def build_fingerprints(smiles_series):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_NBITS)
    smiles_list = smiles_series.tolist()
    n = len(smiles_list)
    chunks, valid_masks = [], []
    for start in range(0, n, CHUNK_SIZE):
        chunk = smiles_list[start : start + CHUNK_SIZE]
        mat, valid_mask = compute_fp_chunk_csr(chunk, gen)
        chunks.append(mat)
        valid_masks.append(valid_mask)
        print(f"  fingerprinted {min(start + CHUNK_SIZE, n)}/{n}", flush=True)
    fp_matrix = sparse.vstack(chunks, format="csr")
    valid = np.concatenate(valid_masks)
    return fp_matrix, valid


def cluster_negatives(fp_matrix, n_clusters=N_CLUSTERS, seed=SEED):
    n = fp_matrix.shape[0]
    batch_size = 10_000
    n_epochs = 3  # ~3 passes over the full pool
    max_iter = n_epochs * int(np.ceil(n / batch_size))
    print(f"  MiniBatchKMeans: n={n} n_clusters={n_clusters} max_iter={max_iter}")
    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=batch_size,
        max_iter=max_iter,
        random_state=seed,
        n_init=3,
    )
    return km.fit_predict(fp_matrix)


def stratified_sample(cluster_labels, n_target, seed=SEED):
    """Sample n_target indices proportional to cluster size (largest-remainder
    rounding), then randomly without replacement within each cluster."""
    rng = np.random.default_rng(seed)
    idx_by_cluster = {c: np.where(cluster_labels == c)[0] for c in np.unique(cluster_labels)}
    cluster_sizes = {c: len(idx) for c, idx in idx_by_cluster.items()}
    total = sum(cluster_sizes.values())

    raw_alloc = {c: n_target * size / total for c, size in cluster_sizes.items()}
    alloc = {c: int(np.floor(v)) for c, v in raw_alloc.items()}
    remainder = n_target - sum(alloc.values())
    by_frac = sorted(raw_alloc.items(), key=lambda kv: kv[1] - alloc[kv[0]], reverse=True)
    for c, _ in by_frac[:remainder]:
        alloc[c] += 1

    selected = []
    for c, k in alloc.items():
        k = min(k, cluster_sizes[c])
        if k > 0:
            selected.append(rng.choice(idx_by_cluster[c], size=k, replace=False))
    sel_idx = np.concatenate(selected)
    rng.shuffle(sel_idx)
    return sel_idx[:n_target]


def get_negative_pool_with_clusters(negatives_pool):
    if CLUSTERS_PATH.exists():
        print(f"Loading cached negative-pool clusters from {CLUSTERS_PATH}")
        cached = pd.read_parquet(CLUSTERS_PATH)
        merged = negatives_pool.merge(
            cached[["compound", "cluster"]], on="compound", how="inner"
        ).reset_index(drop=True)
        return merged

    print("Computing Morgan fingerprints for negative pool "
          f"({len(negatives_pool)} compounds)...")
    fp_matrix, valid = build_fingerprints(negatives_pool["SMILES"])
    negatives_pool = negatives_pool.loc[valid].reset_index(drop=True)
    print(f"  {fp_matrix.shape[0]} valid fingerprints computed")

    print(f"Clustering negative pool into {N_CLUSTERS} clusters (MiniBatchKMeans)...")
    negatives_pool = negatives_pool.copy()
    negatives_pool["cluster"] = cluster_negatives(fp_matrix)

    negatives_pool[["compound", "SMILES", "cluster"]].to_parquet(CLUSTERS_PATH, index=False)
    print(f"  Saved cluster assignments to {CLUSTERS_PATH}")
    return negatives_pool


def make_variant(name, hits, negatives_pool, cluster_labels, n_neg, out_path):
    n_neg = min(n_neg, len(negatives_pool))
    sel_idx = stratified_sample(cluster_labels, n_neg)
    negs = negatives_pool.iloc[sel_idx].drop(columns=["cluster"], errors="ignore")
    out = (
        pd.concat([hits, negs], ignore_index=True)
        .sample(frac=1, random_state=SEED)
        .reset_index(drop=True)
    )
    out.to_parquet(out_path, index=False)
    print(f"[{name}] {len(hits)} hits + {len(negs)} negatives = {len(out)} rows -> {out_path}")


def main():
    print(f"Loading deduped selection from {DEDUP_PATH}...")
    deduped = pd.read_parquet(DEDUP_PATH)
    print(f"Loaded: {deduped.shape}")

    labeled = label(deduped)
    hits = labeled[labeled["LABEL"] == 1].reset_index(drop=True)
    negatives_pool = labeled[labeled["LABEL"] == 0].reset_index(drop=True)
    print(f"Hits: {len(hits)}  Negative pool: {len(negatives_pool)}")

    negatives_pool = get_negative_pool_with_clusters(negatives_pool)
    cluster_labels = negatives_pool["cluster"].to_numpy()

    make_variant("2:1", hits, negatives_pool, cluster_labels,
                 len(hits) * NEGATIVE_RATIO_2TO1, OUT_2TO1)
    make_variant("50:50", hits, negatives_pool, cluster_labels,
                 len(hits), OUT_5050)


if __name__ == "__main__":
    main()
