#!/usr/bin/env python
"""
Synthetic Accessibility (SA) scoring for the CYP-Challenge NON_TDI sets.

Wraps the vendored RDKit ``SA_Score`` contrib library (Ertl & Schuffenhauer,
J. Cheminform. 2009, 1:8) that sits next to this script in ./SA_Score/, so the
scorer works inside a container/VM image even when the RDKit Contrib share
directory is not installed.

Score range is 1 (easy to make) to 10 (hard to make).

Examples
--------
# score the blinded test set
python run_sa_score.py -i cyp-challenge-TEST-BLINDED_ligprepped.csv \
    -o test_sa.csv

# score the training set on the raw (pre-LigPrep) SMILES, 8 workers
python run_sa_score.py -i cyp-challenge-TRAIN_direct_inhibition_with_single_shot_LABELLED.csv \
    --smiles-column SMILES_raw --n-jobs 8 -o train_sa.csv

# a plain .smi / .txt list, or a single molecule on the command line
python run_sa_score.py -i molecules.smi -o out.csv
python run_sa_score.py --smiles "CC(=O)Oc1ccccc1C(=O)O"
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
from rdkit import Chem, RDLogger

HERE = os.path.dirname(os.path.abspath(__file__))


def _load_sascorer():
    """Import sascorer: prefer the vendored copy, fall back to RDKit Contrib."""
    local = os.path.join(HERE, "SA_Score")
    if os.path.isfile(os.path.join(local, "sascorer.py")):
        sys.path.insert(0, local)
    else:
        from rdkit.Chem import RDConfig

        sys.path.insert(0, os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer  # noqa: E402

    return sascorer


sascorer = _load_sascorer()


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def sa_score(smiles: str, sanitize: bool = True):
    """SA score for one SMILES string; returns None if it cannot be parsed."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    try:
        return float(sascorer.calculateScore(mol))
    except Exception:
        return None


def _worker(smiles):
    return sa_score(smiles)


def score_series(smiles_list, n_jobs: int = 1):
    """SA scores for an iterable of SMILES, optionally in parallel."""
    smiles_list = list(smiles_list)
    if n_jobs and n_jobs > 1 and len(smiles_list) > 100:
        import multiprocessing as mp

        chunk = max(1, len(smiles_list) // (n_jobs * 4))
        with mp.Pool(n_jobs) as pool:
            return list(pool.imap(_worker, smiles_list, chunksize=chunk))
    return [sa_score(s) for s in smiles_list]


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def read_input(path: str, smiles_column: str | None, name_column: str | None):
    """Read a CSV/TSV, a whitespace-delimited .smi list, or an SDF."""
    ext = os.path.splitext(path)[1].lower()

    if ext in (".sdf", ".sd"):
        rows = []
        for i, mol in enumerate(Chem.SDMolSupplier(path)):
            if mol is None:
                rows.append({"Molecule_Name": f"mol_{i}", "SMILES": None})
                continue
            rows.append(
                {
                    "Molecule_Name": mol.GetProp("_Name") or f"mol_{i}",
                    "SMILES": Chem.MolToSmiles(mol),
                }
            )
        return pd.DataFrame(rows), "SMILES", "Molecule_Name"

    if ext in (".smi", ".ism", ".txt"):
        df = pd.read_csv(path, sep=r"\s+", header=None, comment="#")
        df.columns = (["SMILES", "Molecule_Name"] + [f"col_{i}" for i in range(2, df.shape[1])])[
            : df.shape[1]
        ]
        name = "Molecule_Name" if "Molecule_Name" in df.columns else None
        return df, "SMILES", name

    sep = "\t" if ext in (".tsv", ".tab") else ","
    df = pd.read_csv(path, sep=sep)

    if smiles_column is None:
        for cand in ("SMILES", "smiles", "Smiles", "canonical_smiles", "SMILES_raw"):
            if cand in df.columns:
                smiles_column = cand
                break
        else:
            raise SystemExit(
                f"No SMILES column found in {path}; pass --smiles-column. "
                f"Columns are: {list(df.columns)}"
            )
    elif smiles_column not in df.columns:
        raise SystemExit(
            f"--smiles-column '{smiles_column}' not in {path}. Columns are: {list(df.columns)}"
        )

    if name_column is None:
        for cand in ("Molecule_Name", "ID", "Name", "name", "molecule_id"):
            if cand in df.columns:
                name_column = cand
                break
    return df, smiles_column, name_column


# --------------------------------------------------------------------------- #
def main(argv=None):
    p = argparse.ArgumentParser(
        description="Compute RDKit synthetic accessibility (SA) scores.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-i", "--input", help="CSV/TSV/.smi/SDF file of molecules")
    p.add_argument("--smiles", help="score a single SMILES string and exit")
    p.add_argument("-o", "--output", help="output CSV (default: <input>_sa.csv)")
    p.add_argument("--smiles-column", default=None, help="SMILES column name (auto-detected)")
    p.add_argument("--name-column", default=None, help="identifier column name (auto-detected)")
    p.add_argument("--score-column", default="SA_Score", help="name of the column to write")
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="if set, also write <score-column>_pass (True when SA <= threshold)",
    )
    p.add_argument(
        "--keep-columns",
        default=None,
        help="comma-separated subset of input columns to carry through (default: all)",
    )
    p.add_argument("--n-jobs", type=int, default=1, help="worker processes")
    p.add_argument("--quiet", action="store_true", help="silence RDKit parse warnings")
    args = p.parse_args(argv)

    if args.quiet:
        RDLogger.DisableLog("rdApp.*")

    if args.smiles:
        score = sa_score(args.smiles)
        print("None" if score is None else f"{score:.3f}")
        return 0

    if not args.input:
        p.error("one of -i/--input or --smiles is required")

    df, smiles_col, name_col = read_input(args.input, args.smiles_column, args.name_column)
    print(f"Read {len(df):,} rows from {args.input} (SMILES column: {smiles_col})")

    scores = score_series(df[smiles_col], n_jobs=args.n_jobs)
    df[args.score_column] = scores

    if args.threshold is not None:
        df[f"{args.score_column}_pass"] = [
            None if s is None else bool(s <= args.threshold) for s in scores
        ]

    if args.keep_columns:
        keep = [c.strip() for c in args.keep_columns.split(",") if c.strip()]
        missing = [c for c in keep if c not in df.columns]
        if missing:
            raise SystemExit(f"--keep-columns not in input: {missing}")
        cols = keep + [c for c in (args.score_column, f"{args.score_column}_pass") if c in df.columns]
        df = df[cols]

    out = args.output or os.path.splitext(args.input)[0] + "_sa.csv"
    df.to_csv(out, index=False)

    ok = df[args.score_column].notna()
    n_bad = int((~ok).sum())
    print(f"Wrote {out}")
    if ok.any():
        s = df.loc[ok, args.score_column]
        print(
            f"SA score: n={ok.sum():,}  mean={s.mean():.2f}  sd={s.std():.2f}  "
            f"min={s.min():.2f}  median={s.median():.2f}  max={s.max():.2f}"
        )
        if args.threshold is not None:
            n_pass = int((s <= args.threshold).sum())
            print(f"  SA <= {args.threshold}: {n_pass:,} / {int(ok.sum()):,} ({100 * n_pass / ok.sum():.1f}%)")
    if n_bad:
        print(f"WARNING: {n_bad:,} molecule(s) could not be parsed/scored (left blank)")
        if name_col:
            bad = df.loc[~ok, name_col].head(10).tolist()
            print(f"  first unparsed: {bad}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
