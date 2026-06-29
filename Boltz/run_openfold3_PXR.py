#!/usr/bin/env python3
"""
PXR (Pregnane X Receptor) co-folding with OpenFold3 (structure prediction only).

Input:  CSV with SMILES and ID columns
Output: per-compound subdirs containing predicted CIF structures

Usage (conda activate openfold3 first):
    python run_openfold3_PXR.py batch1.csv
    python run_openfold3_PXR.py batch1.csv --out_dir PXR_OF3_Batch1
"""

import argparse
import json
import os
import subprocess
from pathlib import Path

import pandas as pd


# ── Full human PXR sequence ────────────────────────────────────────────────────
# UniProt O75469 (NR1I2_HUMAN), canonical isoform 1, 434 aa
PXR_SEQUENCE = (
    "MEVRPKESWNHADFVHCEDTESVPGKPSVNADEEVGGPQICRVCGDKATGYHFNVMTCEG"
    "CKGFFRRAMKRNARLRCPFRKGACEITRKTRRQCQACRLRKCLESGMKKEMIMSDEAVEE"
    "RRALIKRKKSERTGTQPLGVQGLTEEQRMMIRELMDAQMKTFDTTFSHFKNFRLPGVLSS"
    "GCELPESLQAPSREEAAKWSQVRKDLCSLKVSLQLRGEDGSVWNYKPPADSGGKEIFSLL"
    "PHMADMSTYMFKGIISFAKVISYFRDLPIEDQISLLKGAAFELCQLRFNTVFNAETGTWE"
    "CGRLSYCLEDTAGGFQQLLLEPMLKFHYMLKKLQLHEEEYVLMQAISLFSPDRPGVLQHR"
    "VVDQLQEQFAITLKSYIECNRPQPAHRFLFLKIMAMLTELRSINAQHTQRLLRIQDIHPF"
    "ATPLMQELFGITGS"
)

# Checkpoint search order — first existing path wins
def _find_path(*candidates):
    for p in candidates:
        if Path(p).expanduser().exists():
            return str(Path(p).expanduser())
    return None  # let openfold3 auto-download if None

OF3_CKPT = _find_path(
    "/mnt/data/sandeep/openfold3_weights/checkpoints/of3-p2-155k.pt",
    "/mnt/data/sandeep/openfold3_weights/checkpoints/of3_ft3_v1.pt",
    "/mnt/data/sandeep/openfold3_weights/of3_ft3_v1.pt",
    "~/.openfold3/of3_ft3_v1.pt",
)


def load_ligands(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    cols_lower = [c.lower() for c in df.columns]
    smiles_col = next((c for c, cl in zip(df.columns, cols_lower) if "smiles" in cl), None)
    id_col = next(
        (c for c, cl in zip(df.columns, cols_lower)
         if cl in ("id", "id_clean", "structure", "molecule name", "ocnt_id")),
        None,
    )
    if smiles_col is None:
        raise ValueError(f"No SMILES column found. Columns: {list(df.columns)}")
    df["_smiles"] = df[smiles_col].astype(str).str.strip()
    df["_id"] = df[id_col].astype(str).str.strip() if id_col else df.index.astype(str)
    df = df.dropna(subset=["_smiles"])
    df = df[df["_smiles"].str.lower() != "nan"]
    print(f"Loaded {len(df)} ligands from {csv_path}  (SMILES={smiles_col}, ID={id_col})")
    return df


def make_query_json(compound_id: str, smiles: str, json_path: Path):
    query = {
        "seeds": [42],
        "queries": {
            compound_id: {
                "query_name": compound_id,
                "chains": [
                    {
                        "molecule_type": "PROTEIN",
                        "chain_ids": ["A"],
                        "sequence": PXR_SEQUENCE,
                        "non_canonical_residues": None,
                        "smiles": None,
                        "ccd_codes": None,
                        "paired_msa_file_paths": None,
                        "main_msa_file_paths": None,
                        "template_alignment_file_path": None,
                        "template_entry_chain_ids": None,
                        "sdf_file_path": None,
                    },
                    {
                        "molecule_type": "LIGAND",
                        "chain_ids": ["Z"],
                        "sequence": None,
                        "non_canonical_residues": None,
                        "smiles": smiles,
                        "ccd_codes": None,
                        "paired_msa_file_paths": None,
                        "main_msa_file_paths": None,
                        "template_alignment_file_path": None,
                        "template_entry_chain_ids": None,
                        "sdf_file_path": None,
                    },
                ],
                "use_msas": True,
                "use_paired_msas": True,
                "use_main_msas": True,
                "covalent_bonds": None,
            }
        },
    }
    with open(json_path, "w") as f:
        json.dump(query, f, indent=2)


def run_openfold3(query_json: Path, out_dir: Path) -> bool:
    cmd = [
        "run_openfold", "predict",
        "--query-json", str(query_json),
        "--use-msa-server", "true",
        "--use-templates", "true",
        "--output-dir", str(out_dir),
    ]
    if OF3_CKPT:
        cmd += ["--inference-ckpt-path", OF3_CKPT]

    env = os.environ.copy()
    # Redirect openfold3 auto-download cache to the large data partition
    env.setdefault("OPENFOLD_CACHE", "/mnt/data/sandeep/openfold3_weights")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, env=env)
        ok = result.returncode == 0
        if not ok:
            print(f"    FAIL (exit {result.returncode})")
            tail = (result.stderr or result.stdout or "")[-600:]
            if tail:
                print(f"    {tail}")
        return ok
    except subprocess.TimeoutExpired:
        print("    TIMEOUT (60 min)")
        return False
    except Exception as e:
        print(f"    ERROR: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="PXR OpenFold3 co-folding predictions"
    )
    parser.add_argument("smiles_csv", help="Input CSV with ID and SMILES columns")
    parser.add_argument("--out_dir", default="pxr_openfold3_results",
                        help="Root output directory")
    args = parser.parse_args()

    results_root = Path(args.out_dir)
    results_root.mkdir(exist_ok=True)
    json_dir = results_root / "query_jsons"
    json_dir.mkdir(exist_ok=True)

    df = load_ligands(args.smiles_csv)

    print(f"\nGenerating query JSONs → {json_dir}")
    compounds = []
    for _, row in df.iterrows():
        cid = row["_id"].replace("/", "-").replace(" ", "_")[:60]
        smiles = row["_smiles"]
        json_path = json_dir / f"{cid}.json"
        make_query_json(cid, smiles, json_path)
        compounds.append((cid, smiles, json_path))
    print(f"Generated {len(compounds)} query JSONs")

    if OF3_CKPT:
        print(f"\nCheckpoint: {OF3_CKPT}")
    else:
        print(f"\nCheckpoint: auto-download to $OPENFOLD_CACHE")

    print(f"\nRunning {len(compounds)} OpenFold3 predictions (sequential)...")
    successful = 0
    for i, (cid, smiles, json_path) in enumerate(compounds, 1):
        compound_out = results_root / cid
        existing = list(compound_out.rglob("*.cif")) if compound_out.exists() else []
        if existing:
            print(f"[{i}/{len(compounds)}] SKIP {cid} (already done: {len(existing)} CIF files)")
            successful += 1
            continue

        compound_out.mkdir(exist_ok=True)
        print(f"[{i}/{len(compounds)}] {cid}")
        ok = run_openfold3(json_path, compound_out)
        if ok:
            successful += 1

    print(f"\nCompleted: {successful}/{len(compounds)} successful")
    print(f"Structures in: {results_root}/")


if __name__ == "__main__":
    main()
