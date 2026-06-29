#!/usr/bin/env python3
"""
PXR (Pregnane X Receptor) co-folding + binding-affinity prediction with AQAffinity
(OpenFold3 + SandboxAQ affinity head)

Input:  input_ligands_Batch_1.csv  (ID,smiles,ID_clean,smiles_clean columns)
Output: pxr_aqaffinity_results/  — per-compound subdirs + summary CSV

Usage (conda activate sandboxaq first):
    python run_aqaffinity_PXR.py input_ligands_Batch_1.csv
    python run_aqaffinity_PXR.py input_ligands_Batch_1.csv --out_dir my_results
"""

import argparse
import json
import subprocess
import os
import numpy as np
import pandas as pd
from pathlib import Path

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

# Model weights — checked in order, first existing path wins
def _find_path(*candidates):
    for p in candidates:
        if Path(p).expanduser().exists():
            return str(Path(p).expanduser())
    return str(Path(candidates[-1]).expanduser())  # fallback (will error at runtime)

OF3_CKPT = _find_path(
    "~/.openfold3/of3_ft3_v1.pt",
)
AFFINITY_CKPT = _find_path(
    "~/.openfold3/model_weights/model_weights_only.pt",  # VM path (downloaded via setup script)
    "~/aqaffinity/model_weights/model_weights_only.pt",   # local WSL path
)


def load_ligands(csv_path: str) -> pd.DataFrame:
    """Load ligands from CSV, handling the standard 4-column format."""
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
        raise ValueError(f"No SMILES column found in {csv_path}. Columns: {list(df.columns)}")

    df["_smiles"] = df[smiles_col].astype(str).str.strip()
    df["_id"] = df[id_col].astype(str).str.strip() if id_col else df.index.astype(str)
    df = df.dropna(subset=["_smiles"])
    df = df[df["_smiles"].str.lower() != "nan"]
    print(f"Loaded {len(df)} ligands from {csv_path}  (SMILES={smiles_col}, ID={id_col})")
    return df


def make_query_json(compound_id: str, smiles: str, json_path: Path):
    """Write a single-compound AQAffinity query JSON for PXR."""
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


def run_aqaffinity(query_json: Path, out_dir: Path) -> bool:
    """Run aqaffinity predict for a single compound. Returns True on success."""
    cmd = [
        "aqaffinity", "predict",
        "--query_json", str(query_json),
        "--inference_ckpt_path", OF3_CKPT,
        "--binding_affinity_ckpt_path", AFFINITY_CKPT,
        "--use_msa_server", "true",
        "--use_templates", "true",
        "--output_dir", str(out_dir),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        ok = result.returncode == 0
        if not ok:
            print(f"    FAIL (exit {result.returncode})")
            tail = (result.stderr or result.stdout or "")[-500:]
            if tail:
                print(f"    {tail}")
        return ok
    except subprocess.TimeoutExpired:
        print("    TIMEOUT (60 min)")
        return False
    except Exception as e:
        print(f"    ERROR: {e}")
        return False


def collect_affinity(compound_out_dir: Path, compound_id: str) -> list[dict]:
    """
    Parse all _binding_head.txt files under compound_out_dir.
    Each file contains a single float (predicted pKd/pIC50).
    """
    rows = []
    for txt in sorted(compound_out_dir.rglob("*_binding_head.txt")):
        try:
            value = float(txt.read_text().strip())
            # parse seed and sample from filename:
            # <id>_seed_<n>_sample_<m>_binding_head.txt
            parts = txt.stem.split("_")
            seed = next((parts[i + 1] for i, p in enumerate(parts) if p == "seed"), "?")
            sample = next((parts[i + 1] for i, p in enumerate(parts) if p == "sample"), "?")
            rows.append({
                "compound_id": compound_id,
                "seed": seed,
                "sample": sample,
                "affinity_pred": value,
                "txt_file": str(txt.relative_to(compound_out_dir.parent)),
            })
        except Exception as e:
            print(f"    Parse error {txt}: {e}")
    return rows


def parse_all_results(results_root: Path) -> pd.DataFrame:
    """Collect all binding_head.txt outputs and compute per-compound statistics."""
    all_rows = []
    for txt in sorted(results_root.rglob("*_binding_head.txt")):
        compound_id = txt.parts[len(results_root.parts)]
        try:
            value = float(txt.read_text().strip())
            all_rows.append({"compound_id": compound_id, "affinity_pred": value,
                              "txt_file": str(txt.relative_to(results_root))})
        except Exception:
            pass

    if not all_rows:
        return pd.DataFrame()

    raw = pd.DataFrame(all_rows)
    summary = (
        raw.groupby("compound_id")["affinity_pred"]
        .agg(["mean", "std", "count"])
        .rename(columns={"mean": "affinity_mean", "std": "affinity_std", "count": "n_samples"})
        .reset_index()
    )
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="PXR AQAffinity co-folding predictions (sandboxaq env)"
    )
    parser.add_argument("smiles_csv", help="Input CSV with ID and SMILES columns")
    parser.add_argument("--out_dir", default="pxr_aqaffinity_results",
                        help="Root output directory (default: pxr_aqaffinity_results)")
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

    print(f"\nRunning {len(compounds)} AQAffinity predictions (sequential)...")
    successful = 0
    for i, (cid, smiles, json_path) in enumerate(compounds, 1):
        compound_out = results_root / cid
        # Skip if already completed
        existing = list(compound_out.rglob("*_binding_head.txt")) if compound_out.exists() else []
        if existing:
            print(f"[{i}/{len(compounds)}] SKIP {cid} (already done: {len(existing)} result files)")
            successful += 1
            continue

        compound_out.mkdir(exist_ok=True)
        print(f"[{i}/{len(compounds)}] {cid}")
        ok = run_aqaffinity(json_path, compound_out)
        if ok:
            successful += 1

    print(f"\nCompleted: {successful}/{len(compounds)} successful")

    # Collect and summarise results
    summary = parse_all_results(results_root)
    if not summary.empty:
        out_csv = results_root / "pxr_aqaffinity_predictions.csv"
        summary.to_csv(out_csv, index=False)

        print(f"\nTop 10 predicted PXR binders (highest affinity_mean):")
        top = summary.nlargest(10, "affinity_mean")[
            ["compound_id", "affinity_mean", "affinity_std", "n_samples"]
        ].round(3)
        print(top.to_string(index=False))

        print(f"\nResults → {out_csv}")
    else:
        print("No affinity results collected yet.")


if __name__ == "__main__":
    main()
