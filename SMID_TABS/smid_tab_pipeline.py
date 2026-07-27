#!/usr/bin/env python
"""
Parallel SMID (B...N) + TAB (torsion-angle-bin) featurisation and clustering.

Rewrite of SMID_and_TAB_clustering.ipynb as a multiprocessing CLI application.

Pipeline
--------
  1. Stream the input SDF, filter on heavy-atom count, assign unique ids.
  2. In parallel (one process per CPU), for every molecule:
       - embed `--num-confs` conformers (ETKDGv3) and UFF-optimise them,
       - SMID: smallest boron...nitrogen distance over all conformers / B,N pairs,
       - TAB : torsion angles of the lowest-energy conformer, binned.
     Both descriptors reuse the *same* conformer ensemble (see --separate-conformers).
  3. Impute / scale / KMeans on the TAB + SMID features.
  4. Write a features CSV and an SDF carrying all computed properties.

Results are checkpointed per chunk, so an interrupted run can be resumed with
`--resume`.

Example
-------
  python smid_tab_pipeline.py \
      -i Enumeration_From_amine_library_Wuxi_Enamine.sdf \
      -r comps_for_ref.sdf \
      -o screened_2.sdf --max-hac 32 --n-clusters 30 --n-jobs 30
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Keep the numeric libraries single-threaded inside every process.  Without this
# BLAS/OpenMP spawn N threads *per worker* and the machine thrashes.  Must be set
# before numpy/rdkit pull in their native libraries in the workers, hence also in
# the pool initialiser below.
# ----------------------------------------------------------------------------
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

from rdkit import Chem, RDLogger                       # noqa: E402
from rdkit.Chem import AllChem, TorsionFingerprints    # noqa: E402

RDLogger.DisableLog("rdApp.*")

# ----------------------------------------------------------------------------
# Binning definitions
# ----------------------------------------------------------------------------
# SMID (B...N distance, Angstrom).  np.digitize with these edges yields bin 0 for
# d < 1 A, bin 10 for d >= 20 A.
DEFAULT_SMID_BINS = (1, 3, 5, 7, 9, 11, 13, 15, 17, 20)

# Torsion angle bins.  Angles are wrapped to [-180, 180) first, so these 13 edges
# define exactly 12 bins of 30 degrees, indexed 0..11.
DEFAULT_TORSION_BINS = (-180, -150, -120, -90, -60, -30,
                        0, 30, 60, 90, 120, 150, 180)
N_TORSION_BINS = len(DEFAULT_TORSION_BINS) - 1


# ============================================================================
# Molecular helpers
# ============================================================================
def find_boron_atoms(mol):
    return [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 5]


def find_amine_nitrogen_atoms(mol, max_heavy_neighbours=2):
    """Non-aromatic, singly-bonded nitrogens with <= `max_heavy_neighbours`.

    NB: this is the notebook's original definition.  It keeps primary *and*
    secondary amines, and it does NOT exclude amide / carbamate nitrogens (a Boc
    nitrogen is only single-bonded, so it passes).  For a Boc-protected diamine
    library that is usually what you want; pass --exclude-amide-n if it is not.
    """
    out = []
    for a in mol.GetAtoms():
        if a.GetAtomicNum() != 7 or a.GetIsAromatic():
            continue
        if any(b.GetBondTypeAsDouble() >= 2 for b in a.GetBonds()):
            continue
        if sum(1 for nbr in a.GetNeighbors() if nbr.GetAtomicNum() > 1) > max_heavy_neighbours:
            continue
        out.append(a.GetIdx())
    return out


def _is_amide_like(atom):
    """True if the nitrogen is bonded to a C that carries a =O / =S."""
    for nbr in atom.GetNeighbors():
        if nbr.GetAtomicNum() != 6:
            continue
        for b in nbr.GetBonds():
            other = b.GetOtherAtom(nbr)
            if b.GetBondTypeAsDouble() >= 2 and other.GetAtomicNum() in (8, 16):
                return True
    return False


def wrap_angle(a):
    """Map an angle in degrees to [-180, 180).

    RDKit's CalculateTorsionAngles returns angles in [0, 360); the notebook binned
    them against [-180, 180] edges, which collapsed everything >= 180 into the
    final bin.  This wrap is what makes DEFAULT_TORSION_BINS correct.
    """
    return (a + 180.0) % 360.0 - 180.0


def bin_values(values, edges):
    """Bin `values` into len(edges)-1 bins, clipping the open ends."""
    if len(values) == 0:
        return np.empty(0, dtype=int)
    idx = np.digitize(np.asarray(values, dtype=float), np.asarray(edges[1:-1], dtype=float))
    return idx.astype(int)


def bin_smid(value, edges=DEFAULT_SMID_BINS):
    return int(np.digitize(float(value), np.asarray(edges, dtype=float)))


# ============================================================================
# Per-molecule work (executed inside worker processes)
# ============================================================================
_CFG: dict = {}


def _worker_init(cfg):
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"
    RDLogger.DisableLog("rdApp.*")
    global _CFG
    _CFG = cfg


def _embed(mol, num_confs, seed, optimize, prune_rms):
    """Embed + (optionally) UFF-optimise.  Returns (conf_ids, energies|None)."""
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.numThreads = 1            # process-level parallelism only
    params.useSmallRingTorsions = True
    if prune_rms and prune_rms > 0:
        params.pruneRmsThresh = prune_rms

    conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params))
    if not conf_ids:
        # ETKDG occasionally fails on strained cages; random coords almost always works.
        params.useRandomCoords = True
        conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params))
    if not conf_ids:
        raise ValueError("conformer generation failed")

    energies = None
    if optimize:
        # One batched call instead of a Python loop over UFFOptimizeMolecule --
        # same result, markedly faster, and it hands back the energies for free.
        # UFF (not MMFF94) is required here: MMFF has no boron parameters.
        if AllChem.UFFHasAllMoleculeParams(mol):
            res = AllChem.UFFOptimizeMoleculeConfs(mol, numThreads=1, maxIters=500)
            energies = [e for _conv, e in res]
    return conf_ids, energies


def compute_features(uid, smiles):
    """SMID + TAB descriptors for one molecule.  Never raises."""
    cfg = _CFG
    rec = {
        "uid": uid,
        "smid_bn": np.nan, "smid_bin": np.nan,
        "boron_idx": -1, "nitrogen_idx": -1, "best_conf_id": -1,
        "n_torsions": 0, "tab_bins": [], "tab_conf_id": -1,
        "smid_error": None, "tab_error": None,
    }

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        rec["smid_error"] = rec["tab_error"] = "SMILES parse failed"
        return rec

    molh = Chem.AddHs(mol)

    b_atoms = find_boron_atoms(molh)
    n_atoms = find_amine_nitrogen_atoms(molh, cfg["max_heavy_neighbours"])
    if cfg["exclude_amide_n"]:
        n_atoms = [i for i in n_atoms if not _is_amide_like(molh.GetAtomWithIdx(i))]

    # Cheap failures first: no point embedding 20 conformers of a molecule that
    # cannot yield a SMID at all.
    if not b_atoms:
        rec["smid_error"] = "No boron atom found."
    if not n_atoms:
        rec["smid_error"] = "No terminal nitrogen atom found."

    want_smid = bool(b_atoms and n_atoms)
    n_confs = cfg["num_confs"] if want_smid else 1

    try:
        conf_ids, energies = _embed(molh, n_confs, cfg["seed"],
                                    cfg["optimize"], cfg["prune_rms"])
    except Exception as exc:                                  # noqa: BLE001
        rec["smid_error"] = rec["smid_error"] or str(exc)
        rec["tab_error"] = str(exc)
        return rec

    # ---------------- SMID -------------------------------------------------
    if want_smid:
        try:
            b_arr = np.asarray(b_atoms, dtype=int)
            n_arr = np.asarray(n_atoms, dtype=int)
            best = (np.inf, -1, -1, -1)
            for cid in conf_ids:
                pos = molh.GetConformer(cid).GetPositions()
                # |B| x |N| distance block in one shot instead of a Python
                # double loop over GetBondLength.
                d = np.linalg.norm(pos[b_arr][:, None, :] - pos[n_arr][None, :, :], axis=-1)
                i, j = np.unravel_index(np.argmin(d), d.shape)
                if d[i, j] < best[0]:
                    best = (float(d[i, j]), int(b_arr[i]), int(n_arr[j]), int(cid))
            rec["smid_bn"] = best[0]
            rec["smid_bin"] = bin_smid(best[0], cfg["smid_bins"])
            rec["boron_idx"], rec["nitrogen_idx"], rec["best_conf_id"] = best[1], best[2], best[3]
        except Exception as exc:                              # noqa: BLE001
            rec["smid_error"] = str(exc)

    # ---------------- TAB --------------------------------------------------
    try:
        if cfg["separate_conformers"]:
            tab_mol = Chem.AddHs(Chem.Mol(mol))
            tab_ids, tab_en = _embed(tab_mol, 1, cfg["seed"], cfg["optimize"], 0.0)
        else:
            tab_mol, tab_ids, tab_en = molh, conf_ids, energies

        if cfg["tab_conformer"] == "lowest_energy" and tab_en:
            cid = int(tab_ids[int(np.argmin(tab_en))])
        else:
            cid = int(tab_ids[0])

        tors, ring_tors = TorsionFingerprints.CalculateTorsionLists(tab_mol)
        angles_raw = TorsionFingerprints.CalculateTorsionAngles(
            tab_mol, tors, ring_tors, confId=cid)

        # CalculateTorsionAngles returns [([angle, ...], symmetry_norm), ...].
        # Element [0] is the list of symmetry-equivalent angles; element [1] is a
        # constant (180.0 / 120.0 / ...), NOT an angle.
        angles = [wrap_angle(float(t[0][0])) for t in angles_raw if t and len(t[0])]

        rec["tab_bins"] = [int(b) for b in bin_values(angles, cfg["torsion_bins"])]
        rec["n_torsions"] = len(angles)
        rec["tab_conf_id"] = cid
    except Exception as exc:                                  # noqa: BLE001
        rec["tab_error"] = str(exc)

    return rec


def _run_chunk(chunk):
    return [compute_features(uid, smi) for uid, smi in chunk]


# ============================================================================
# I/O
# ============================================================================
def stream_records(path, max_hac=None, id_prop=None, limit=None, source_tag=""):
    """First pass: ids + SMILES only, so a 135 MB SDF never sits in RAM as mols.

    Yields dicts and, as a side effect, records the position of each molecule in
    the file so the writer pass can find it again.
    """
    records = []
    for pos, mol in enumerate(Chem.SDMolSupplier(str(path))):
        if limit is not None and len(records) >= limit:
            break
        if mol is None:
            continue

        hac = None
        if max_hac is not None:
            if mol.HasProp("HAC"):
                try:
                    hac = float(mol.GetProp("HAC"))
                except ValueError:
                    hac = None
            if hac is None:
                hac = float(mol.GetNumHeavyAtoms())
            if hac > max_hac:
                continue

        if id_prop and mol.HasProp(id_prop):
            name = mol.GetProp(id_prop)
        elif mol.HasProp("ID"):
            name = mol.GetProp("ID")
        elif mol.GetProp("_Name"):
            name = mol.GetProp("_Name")
        else:
            name = f"mol{pos}"

        records.append({
            "uid": f"{source_tag}{len(records)}",
            "ID": name,
            "sdf_pos": pos,
            "HAC": hac if hac is not None else float(mol.GetNumHeavyAtoms()),
            "SMILES": Chem.MolToSmiles(mol),
        })
    return records


def load_checkpoint(path):
    done = {}
    if path and Path(path).exists():
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue          # truncated final line from a hard kill
                done[rec["uid"]] = rec
    return done


# ============================================================================
# Feature assembly + clustering
# ============================================================================
def build_feature_frame(meta_df, results, torsion_bins, positional=True):
    """meta_df: uid/ID/SMILES.  results: list of per-molecule dicts."""
    res_df = pd.DataFrame(results).set_index("uid")
    res_df = res_df.reindex(meta_df["uid"].values)

    n_bins = len(torsion_bins) - 1
    tab_bins = [b if isinstance(b, (list, tuple)) else [] for b in res_df["tab_bins"].values]

    hist = np.zeros((len(res_df), n_bins), dtype=float)
    max_t = max((len(b) for b in tab_bins), default=0)
    pos = np.full((len(res_df), max_t), np.nan) if positional and max_t else None

    for i, bins in enumerate(tab_bins):
        if not bins:
            continue
        b = np.asarray(bins, dtype=int)
        hist[i] = np.bincount(b, minlength=n_bins)[:n_bins]
        if pos is not None:
            pos[i, :len(b)] = b

    out = meta_df.reset_index(drop=True).copy()
    for c in ("smid_bn", "smid_bin", "boron_idx", "nitrogen_idx",
              "best_conf_id", "n_torsions", "tab_conf_id", "smid_error", "tab_error"):
        out[c] = res_df[c].values

    hist_cols = [f"tab_hist_{k:02d}" for k in range(n_bins)]
    out[hist_cols] = hist

    pos_cols = []
    if pos is not None:
        pos_cols = [f"torsion_bin_{k}" for k in range(max_t)]
        out[pos_cols] = pos

    return out, hist_cols, pos_cols


def cluster(df, feature_cols, n_clusters, seed=42):
    from sklearn.cluster import KMeans
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    # Drop all-NaN / zero-variance columns: they add nothing and the imputer
    # cannot fill a column that is entirely missing.
    X = X.loc[:, X.notna().any(axis=0)]
    X_imp = SimpleImputer(strategy="median").fit_transform(X)
    keep = X_imp.std(axis=0) > 0
    X_imp = X_imp[:, keep]
    X_scaled = StandardScaler().fit_transform(X_imp)

    k = min(n_clusters, len(df))
    km = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = km.fit_predict(X_scaled)
    return labels, km.inertia_, X_scaled.shape[1]


# ============================================================================
# Output
# ============================================================================
def write_sdf(out_path, df, input_path, ref_path, prop_cols):
    """Second pass over the SDFs: re-read each kept molecule and attach results.

    Streaming here means the original 3D coordinates and SD tags survive without
    ever holding 42 k mol objects in memory.
    """
    by_source = {}
    for src, sub in df.groupby("source"):
        by_source[src] = {int(p): i for i, p in zip(sub.index, sub["sdf_pos"])}

    writer = Chem.SDWriter(str(out_path))
    written = 0
    for src, path in (("ref", ref_path), ("lib", input_path)):
        if path is None or src not in by_source:
            continue
        wanted = by_source[src]
        for pos, mol in enumerate(Chem.SDMolSupplier(str(path))):
            if mol is None or pos not in wanted:
                continue
            row = df.loc[wanted[pos]]
            for col in prop_cols:
                val = row[col]
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    continue
                mol.SetProp(str(col), str(val))
            mol.SetProp("_Name", str(row["uid"]))
            writer.write(mol)
            written += 1
    writer.close()
    return written


# ============================================================================
# Main
# ============================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    io = p.add_argument_group("input / output")
    io.add_argument("-i", "--input", required=True, help="library SDF")
    io.add_argument("-r", "--ref", default=None, help="reference SDF (written first)")
    io.add_argument("-o", "--output", default="screened_2.sdf", help="output SDF")
    io.add_argument("--csv", default=None, help="features CSV (default: <output>.csv)")
    io.add_argument("--id-prop", default=None, help="SD tag to use as ID (default: ID/_Name)")
    io.add_argument("--max-hac", type=float, default=32.0,
                    help="drop molecules above this heavy-atom count (0 = no filter)")
    io.add_argument("--limit", type=int, default=None,
                    help="only process the first N library molecules (smoke test)")

    cm = p.add_argument_group("conformers / descriptors")
    cm.add_argument("--num-confs", type=int, default=20, help="conformers per molecule for SMID")
    cm.add_argument("--seed", type=int, default=0xF00D)
    cm.add_argument("--no-optimize", dest="optimize", action="store_false",
                    help="skip UFF optimisation (much faster, cruder geometries)")
    cm.add_argument("--prune-rms", type=float, default=0.0,
                    help="ETKDG pruneRmsThresh; >0 discards near-duplicate conformers")
    cm.add_argument("--separate-conformers", action="store_true",
                    help="embed a fresh single conformer for TAB instead of reusing "
                         "the SMID ensemble (matches the notebook, ~2x slower)")
    cm.add_argument("--tab-conformer", choices=("lowest_energy", "first"),
                    default="lowest_energy", help="which conformer the torsions come from")
    cm.add_argument("--max-heavy-neighbours", type=int, default=2,
                    help="1 = primary amines only, 2 = primary + secondary (notebook default)")
    cm.add_argument("--exclude-amide-n", action="store_true",
                    help="ignore amide/carbamate (incl. Boc) nitrogens")

    cl = p.add_argument_group("clustering")
    cl.add_argument("--n-clusters", type=int, default=30)
    cl.add_argument("--cluster-features", choices=("hist", "positional", "both"),
                    default="hist",
                    help="hist = order-invariant torsion-bin counts (recommended); "
                         "positional = the notebook's torsion_bin_<j> columns")
    cl.add_argument("--no-smid-in-clustering", dest="smid_in_clustering",
                    action="store_false")
    cl.add_argument("--no-cluster", action="store_true", help="featurise only")

    rt = p.add_argument_group("runtime")
    rt.add_argument("-j", "--n-jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    rt.add_argument("--chunk-size", type=int, default=25,
                    help="molecules per task; larger amortises IPC, smaller = finer progress")
    rt.add_argument("--checkpoint", default=None,
                    help="JSONL checkpoint file (default: <output>.progress.jsonl)")
    rt.add_argument("--resume", action="store_true", help="reuse an existing checkpoint")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    t0 = time.time()

    out_path = Path(args.output)
    csv_path = Path(args.csv) if args.csv else out_path.with_suffix(".features.csv")
    ckpt_path = Path(args.checkpoint) if args.checkpoint \
        else out_path.with_suffix(".progress.jsonl")
    max_hac = args.max_hac if args.max_hac and args.max_hac > 0 else None

    # ---- 1. read metadata -------------------------------------------------
    print(f"[1/5] reading structures", flush=True)
    records = []
    if args.ref:
        ref_recs = stream_records(args.ref, max_hac=None, id_prop=args.id_prop,
                                  source_tag="ref_")
        for r in ref_recs:
            r["source"] = "ref"
        records += ref_recs
        print(f"      reference : {len(ref_recs):>7,d} molecules ({args.ref})")

    lib_recs = stream_records(args.input, max_hac=max_hac, id_prop=args.id_prop,
                              limit=args.limit, source_tag="lib_")
    for r in lib_recs:
        r["source"] = "lib"
    records += lib_recs
    print(f"      library   : {len(lib_recs):>7,d} molecules "
          f"({args.input}{f', HAC <= {max_hac:g}' if max_hac else ''})")

    if not records:
        sys.exit("no molecules to process")
    meta_df = pd.DataFrame(records)

    # ---- 2. parallel featurisation ---------------------------------------
    cfg = {
        "num_confs": args.num_confs,
        "seed": args.seed,
        "optimize": args.optimize,
        "prune_rms": args.prune_rms,
        "separate_conformers": args.separate_conformers,
        "tab_conformer": args.tab_conformer,
        "max_heavy_neighbours": args.max_heavy_neighbours,
        "exclude_amide_n": args.exclude_amide_n,
        "smid_bins": DEFAULT_SMID_BINS,
        "torsion_bins": DEFAULT_TORSION_BINS,
    }

    done = load_checkpoint(ckpt_path) if args.resume else {}
    if done:
        print(f"      resuming  : {len(done):,d} molecules already in {ckpt_path}")
    elif ckpt_path.exists():
        ckpt_path.unlink()

    todo = [(r["uid"], r["SMILES"]) for r in records if r["uid"] not in done]
    chunks = [todo[i:i + args.chunk_size] for i in range(0, len(todo), args.chunk_size)]

    n_jobs = max(1, min(args.n_jobs, len(chunks) or 1))
    print(f"[2/5] featurising {len(todo):,d} molecules on {n_jobs} processes "
          f"({args.num_confs} confs, UFF={'on' if args.optimize else 'off'})", flush=True)

    results = list(done.values())
    if chunks:
        try:
            from tqdm.auto import tqdm
        except ImportError:
            def tqdm(it, **kw):
                return it

        ctx = mp.get_context("fork")
        with ctx.Pool(n_jobs, initializer=_worker_init, initargs=(cfg,)) as pool, \
                open(ckpt_path, "a") as ckpt:
            bar = tqdm(pool.imap_unordered(_run_chunk, chunks),
                       total=len(chunks), unit="chunk", smoothing=0.05)
            for chunk_res in bar:
                results.extend(chunk_res)
                for rec in chunk_res:
                    ckpt.write(json.dumps(rec) + "\n")
                ckpt.flush()      # a 3 h run must survive a kill

    dt = time.time() - t0
    print(f"      done in {dt/60:.1f} min ({len(records)/max(dt, 1e-9):.1f} mol/s)")

    # ---- 3. assemble features --------------------------------------------
    print("[3/5] assembling features", flush=True)
    need_pos = args.cluster_features in ("positional", "both")
    df, hist_cols, pos_cols = build_feature_frame(
        meta_df, results, DEFAULT_TORSION_BINS, positional=need_pos)

    n_smid_fail = int(df["smid_bn"].isna().sum())
    n_tab_fail = int((df["n_torsions"] == 0).sum())
    print(f"      SMID ok: {len(df) - n_smid_fail:,d}/{len(df):,d}   "
          f"TAB ok: {len(df) - n_tab_fail:,d}/{len(df):,d}")
    if n_smid_fail:
        top = df.loc[df["smid_bn"].isna(), "smid_error"].value_counts().head(3)
        for msg, cnt in top.items():
            print(f"        SMID failure: {cnt:>6,d}  {msg}")

    # ---- 4. cluster -------------------------------------------------------
    if not args.no_cluster:
        feat = []
        if args.cluster_features in ("hist", "both"):
            feat += hist_cols + ["n_torsions"]
        if args.cluster_features in ("positional", "both"):
            feat += pos_cols
        if args.smid_in_clustering:
            feat.append("smid_bin")
        feat = [c for c in feat if c in df.columns]

        print(f"[4/5] KMeans k={args.n_clusters} on {len(feat)} features "
              f"({args.cluster_features})", flush=True)
        labels, inertia, n_used = cluster(df, feat, args.n_clusters)
        df["cluster"] = labels
        sizes = pd.Series(labels).value_counts()
        print(f"      {n_used} usable features, inertia={inertia:,.0f}, "
              f"cluster sizes {sizes.min()}-{sizes.max()} (median {int(sizes.median())})")
    else:
        print("[4/5] clustering skipped")

    # ---- 5. write ---------------------------------------------------------
    print("[5/5] writing output", flush=True)
    df.to_csv(csv_path, index=False)

    prop_cols = [c for c in df.columns if c not in ("sdf_pos", "source")]
    n = write_sdf(out_path, df, args.input, args.ref, prop_cols)
    print(f"      {csv_path}  ({len(df):,d} rows x {df.shape[1]} cols)")
    print(f"      {out_path}  ({n:,d} molecules)")
    print(f"total wall time {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
