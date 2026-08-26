"""Auxiliary CYP450 substrate labels from Ni et al. 2025, for the multitask inhibition model.

Loads the curated substrate / non-substrate dataset published alongside

    Ni, Y.-H. *et al.* "Curated CYP450 Interaction Dataset: Covering the Majority of
    Phase I Drug Metabolism." *Scientific Data* **12**, 1427 (2025).
    doi:10.1038/s41597-025-05753-8 — data: doi:10.6084/m9.figshare.26630515 (CC BY 4.0)

and reshapes it into the wide ``SMILES x isoform`` label matrix that
``train_multitask_cyp.py`` stacks in as extra task columns.

Why this is only ever an *auxiliary* task
-----------------------------------------
Ni et al. label whether a compound is a **substrate** of an isoform — whether the enzyme
metabolises it. The challenge targets are **direct inhibition** pIC50 and TDI. These are
different, sometimes anti-correlated, properties, so a substrate label is never a
surrogate for a pIC50 value. The bet is narrower: substrate recognition and competitive
inhibition both hinge on fitting the same active site, so forcing the shared trees to
also separate substrates from non-substrates may buy the primary tasks a better
representation. That bet is what ``--aux-substrate`` measures.

Download
--------
``fetch_substrate_data()`` pulls the twelve per-isoform CSVs straight from the Figshare
record; ``python cyp_substrate_aux.py --download`` does the same from the shell. Only the
four plain ``Name, SMILES, Label, Source`` files per isoform are fetched — the much larger
precomputed ``*_PF.csv`` PubChem-fingerprint matrices are skipped, since we featurize with
the same ECFP4 + RDKit 2D pipeline as the primary tasks.

Curation applied on top of theirs
---------------------------------
* SMILES are re-canonicalised with RDKit and unparseable rows dropped.
* Their published training and testing files are merged per isoform. They overlap by 3-6
  compounds per isoform, which matters for reproducing their reported numbers but not
  here, where the whole set is training data for an auxiliary task.
* A compound labelled inconsistently for the same isoform (both substrate and
  non-substrate, across the merged files) is dropped **for that isoform only** — its
  labels for other isoforms are kept.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_AUX_DIR = PROJECT_ROOT / "data" / "external" / "ni2025_cyp450"

FIGSHARE_API = "https://api.figshare.com/v2/articles/26630515"

#: Every isoform in the Ni et al. release.
ALL_ISOFORMS = ["CYP1A2", "CYP2C9", "CYP2C19", "CYP2D6", "CYP2E1", "CYP3A4"]

#: The four the challenge actually assays. CYP2C19 and CYP2E1 are still worth offering as
#: auxiliary tasks — they add shared-encoder signal without adding a scored endpoint —
#: which is why ``--aux-isoforms`` exists.
CHALLENGE_ISOFORMS = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]


# --------------------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------------------


def fetch_substrate_data(aux_dir: Path = DEFAULT_AUX_DIR, force: bool = False) -> Path:
    """Download the twelve per-isoform label CSVs from Figshare into ``aux_dir``.

    Skips the ``*_PF.csv`` / ``*.xlsx`` PubChem-fingerprint files (~40 MB) — we featurize
    from SMILES ourselves. Existing files are left alone unless ``force`` is set.
    """
    aux_dir = Path(aux_dir)
    aux_dir.mkdir(parents=True, exist_ok=True)

    with urllib.request.urlopen(FIGSHARE_API, timeout=120) as response:
        record = json.load(response)

    wanted = [
        f
        for f in record["files"]
        if f["name"].endswith(".csv") and not f["name"].endswith("_PF.csv")
    ]
    for entry in wanted:
        target = aux_dir / entry["name"]
        if target.exists() and not force:
            continue
        urllib.request.urlretrieve(entry["download_url"], target)
        print(f"downloaded {entry['name']} ({target.stat().st_size:,} bytes)")

    print(f"{len(wanted)} label file(s) in {aux_dir}  [{record['license']['name']}, {record['doi']}]")
    return aux_dir


# --------------------------------------------------------------------------------------
# Loading and reshaping
# --------------------------------------------------------------------------------------


def canonical_smiles(smiles) -> str | None:
    """Canonical isomeric SMILES, or None if RDKit cannot parse the input."""
    mol = Chem.MolFromSmiles(str(smiles))
    return Chem.MolToSmiles(mol) if mol is not None else None


def murcko_scaffold(smiles) -> str:
    """Bemis-Murcko scaffold SMILES; ``""`` for acyclic molecules and parse failures.

    An empty scaffold is not an identity — every acyclic molecule shares it — so callers
    matching scaffolds between sets must exclude ``""`` and fall back to exact structure.
    """
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return ""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except ValueError:
        return ""


def load_substrate_labels(
    aux_dir: Path = DEFAULT_AUX_DIR,
    isoforms: list[str] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load the substrate labels as a wide ``SMILES x isoform`` frame.

    Returns
    -------
    pandas.DataFrame
        Columns ``SMILES`` (RDKit-canonical), ``scaffold`` (Bemis-Murcko), and one
        ``<isoform>_is_substrate`` column per requested isoform holding 1.0 (substrate),
        0.0 (non-substrate) or NaN (that isoform was never assayed for this compound).

    Raises
    ------
    FileNotFoundError
        If the CSVs are missing — run ``fetch_substrate_data()`` first.

    """
    aux_dir = Path(aux_dir)
    isoforms = list(isoforms or CHALLENGE_ISOFORMS)

    per_isoform = {}
    for isoform in isoforms:
        frames = []
        for split in ("trainingset", "testingset"):
            path = aux_dir / f"{isoform}_{split}.csv"
            if not path.exists():
                raise FileNotFoundError(
                    f"{path} not found — run `python cyp_substrate_aux.py --download` "
                    f"to fetch the Ni et al. dataset from Figshare."
                )
            frames.append(pd.read_csv(path))
        merged = pd.concat(frames, ignore_index=True)

        # ~61 rows per isoform carry malformed phosphate SMILES ("[PH](=O)(=O)O" — a
        # pentavalent P written with an explicit hydrogen), which RDKit rightly rejects.
        # They are endogenous CoA thioesters and nucleotides, almost all non-substrates and
        # well outside the challenge's chemical space, so they are dropped rather than
        # repaired by guesswork.
        merged["SMILES"] = merged["SMILES"].map(canonical_smiles)
        n_unparseable = merged["SMILES"].isna().sum()
        merged = merged.dropna(subset=["SMILES"])

        # One row per compound. A compound with both labels for this isoform is ambiguous;
        # drop it here rather than let a coin-flip label into the auxiliary target.
        agreement = merged.groupby("SMILES")["Label"].nunique()
        conflicted = set(agreement.index[agreement > 1])
        merged = merged[~merged["SMILES"].isin(conflicted)]
        labels = merged.groupby("SMILES")["Label"].first().astype(float)

        per_isoform[f"{isoform}_is_substrate"] = labels
        if verbose:
            n_pos = int((labels == 1).sum())
            print(
                f"{isoform}: {len(labels)} compounds ({n_pos} substrate / "
                f"{len(labels) - n_pos} non-substrate)"
                + (f", dropped {len(conflicted)} conflicting" if conflicted else "")
                + (f", {n_unparseable} unparseable" if n_unparseable else "")
            )

    wide = pd.DataFrame(per_isoform).reset_index().rename(columns={"index": "SMILES"})
    wide["scaffold"] = wide["SMILES"].map(murcko_scaffold)
    label_cols = [c for c in wide.columns if c.endswith("_is_substrate")]
    wide = wide[["SMILES", "scaffold", *label_cols]]

    if verbose:
        print(
            f"auxiliary set: {len(wide)} unique compounds, "
            f"{int(wide[label_cols].notna().sum().sum())} labels across {len(label_cols)} isoform(s)"
        )
    return wide


def drop_holdout_leakage(
    aux_df: pd.DataFrame, holdout_smiles, verbose: bool = True
) -> pd.DataFrame:
    """Remove auxiliary compounds that would let the model peek at a scaffold hold-out.

    The substrate labels carry no pIC50 information, so this is not label leakage. But the
    scaffold split's premise is that no hold-out *core* was seen in training, and an
    auxiliary row puts that molecule's features in front of the shared trees. Leaving them
    in would flatter the auxiliary model for the wrong reason.

    Matching is on Bemis-Murcko scaffold, plus exact canonical structure to catch the
    acyclic molecules whose scaffold is the empty string and so cannot be matched that way.
    """
    holdout_smiles = [canonical_smiles(s) for s in holdout_smiles]
    holdout_structures = {s for s in holdout_smiles if s}
    holdout_scaffolds = {murcko_scaffold(s) for s in holdout_structures}
    holdout_scaffolds.discard("")

    blocked = aux_df["scaffold"].isin(holdout_scaffolds) | aux_df["SMILES"].isin(
        holdout_structures
    )
    if verbose and blocked.any():
        print(
            f"auxiliary rows dropped for hold-out scaffold overlap: {int(blocked.sum())} "
            f"of {len(aux_df)}"
        )
    return aux_df.loc[~blocked].reset_index(drop=True)


def map_labels_to_target_scale(
    labels: np.ndarray, y_primary: np.ndarray, spread: float
) -> np.ndarray:
    """Map binary substrate labels onto the pIC50 scale of the primary tasks.

    A shared LightGBM fits one L2 objective across every stacked row, and split gain scales
    with target variance. Left as raw 0/1 against pIC50 values spanning several log units,
    the auxiliary rows would contribute almost no gain and the auxiliary task would be
    silently ignored. Mapping non-substrate and substrate to ``mu -/+ spread * sigma`` of
    the primary labels puts both on a comparable footing; ``spread=0.5`` separates the two
    classes by one standard deviation of the primary target.

    The task one-hot block means the trees can still tell auxiliary rows apart and
    specialise, so this rescaling does not distort the primary predictions.
    """
    finite = np.asarray(y_primary, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    mu, sigma = float(finite.mean()), float(finite.std())
    return mu + (2.0 * np.asarray(labels, dtype=np.float64) - 1.0) * spread * sigma


def main(argv=None) -> None:
    """Download the dataset and print a summary of the auxiliary label matrix."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Fetch the CSVs from Figshare")
    parser.add_argument("--force", action="store_true", help="Re-download files that already exist")
    parser.add_argument("--aux-dir", type=Path, default=DEFAULT_AUX_DIR)
    parser.add_argument("--isoforms", nargs="+", default=CHALLENGE_ISOFORMS, choices=ALL_ISOFORMS)
    args = parser.parse_args(argv)

    if args.download:
        fetch_substrate_data(args.aux_dir, force=args.force)
    load_substrate_labels(args.aux_dir, args.isoforms)


if __name__ == "__main__":
    main()
