#!/usr/bin/env python3
"""
PXR (Pregnane X Receptor) Binding-Affinity Modelling with Boltz-2
Input:  smiles,ID CSV  (or the OpenAdmet 4-column format: structure,smiles,Molecule Name,OCNT_ID)
Output: Boltz-2 pIC50 / binary-binder predictions for each compound

Target
  PXR  =  Nuclear receptor subfamily 1 group I member 2 (NR1I2)
           UniProt O75469 (NR1I2_HUMAN)  ·  434 aa canonical isoform 1

Chain topology
  Chain A — full-length human PXR
  Chain L — query ligand (SMILES)

Key binding-pocket residues (full-protein numbering, from 1ILG SR12813 co-crystal):
  S208   H3  — direct H-bond with ligand
  M246   H4/H5 loop
  S247   H4/H5 — H-bond donor
  F264   β-sheet — hydrophobic lid
  C284   β-sheet
  Q285   β-sheet — polar contact
  W299   H7  — critical hydrophobic/H-bond
  Y306   H7/H8 loop
  H407   H12 (AF2 helix)
  R410   AF2 helix — key agonist contact

References
  Watkins et al. 2001  Science 292:2329 (PDB 1ILG, SR12813)
  Watkins et al. 2003  Mol Cell 11:1353 (PDB 1NRL, hyperforin)
"""

import pandas as pd
import yaml
import subprocess
import json
import os
import time
from pathlib import Path
import numpy as np
import argparse

# ── Full human PXR sequence ────────────────────────────────────────────────────
# UniProt O75469 (NR1I2_HUMAN), canonical isoform 1, 434 aa
# Retrieved 2026-06-08 from https://www.uniprot.org/uniprot/O75469.fasta
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

# ── PXR ligand-binding pocket (1ILG contacts, full-protein numbering) ──────────
#
#  S208  H3  H-bond donor to most scaffolds
#  M246  H4-H5 loop hydrophobic contact
#  S247  H4-H5 H-bond donor
#  F264  β-sheet Phe — aromatic lid
#  C284  β-sheet Cys — forms part of the floor
#  Q285  β-sheet Gln — polar contact
#  W299  H7 Trp — dominant hydrophobic + H-bond anchor
#  Y306  H7/H8 Tyr — flexible lid contact
#  H407  AF2 helix His — gating residue for agonists
#  R410  AF2 helix Arg — agonist H-bond
PXR_POCKET = [
    ["A", 208],   # H3   Ser  — H-bond to ligand
    ["A", 246],   # H4/5 Met  — hydrophobic
    ["A", 247],   # H4/5 Ser  — H-bond to ligand
    ["A", 264],   # β3   Phe  — aromatic lid
    ["A", 284],   # β4   Cys
    ["A", 285],   # β4   Gln  — polar contact
    ["A", 299],   # H7   Trp  — anchor H-bond + hydrophobic
    ["A", 306],   # H7/8 Tyr  — flexible lid
    ["A", 407],   # AF2  His  — agonist gate
    ["A", 410],   # AF2  Arg  — agonist H-bond
]


class PXRBoltzPredictor:
    def __init__(self, out_dir="pxr_boltz_results"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(exist_ok=True)
        self.yaml_dir = self.out_dir / "yamls"
        self.yaml_dir.mkdir(exist_ok=True)

    def parse_smiles_csv(self, smiles_csv):
        """
        Accept two input formats:
          • Simple  : smiles,ID  (no header)
          • OpenAdmet: structure,smiles,Molecule Name,OCNT_ID  (with header)
        """
        raw = pd.read_csv(smiles_csv, nrows=0)
        if "smiles" in [c.lower() for c in raw.columns]:
            df = pd.read_csv(smiles_csv)
            # normalise column names to lower-case for lookup
            df.columns = [c.lower().strip() for c in df.columns]
            df["ID_clean"] = (
                df.get("molecule name", df.get("ocnt_id", df.get("structure", df.index.astype(str))))
                .astype(str).str.strip()
            )
            df["smiles_clean"] = df["smiles"].astype(str).str.strip()
        else:
            # headerless smiles,ID format
            df = pd.read_csv(smiles_csv, header=None, names=["smiles", "ID"])
            df["ID_clean"] = df["ID"].astype(str).str.strip()
            df["smiles_clean"] = df["smiles"].astype(str).str.strip()

        df = df.dropna(subset=["smiles_clean"])
        df = df[df["smiles_clean"] != "nan"]
        print(f"Loaded {len(df)} PXR ligands")
        return df

    def create_yaml(self, smiles, compound_id, ligand_idx):
        """Create YAML: PXR (chain A) + ligand (chain L) with pocket constraints."""
        yaml_data = {
            "version": 1,
            "sequences": [
                {
                    "protein": {
                        "id": "A",
                        "sequence": PXR_SEQUENCE,
                        "msa": "empty"
                    }
                },
                {
                    "ligand": {
                        "id": "L",
                        "smiles": smiles
                    }
                }
            ],
            "properties": [
                {"affinity": {"binder": "L"}}
            ],
            "constraints": [
                {"pocket": {"binder": "L", "contacts": PXR_POCKET}}
            ]
        }

        # Sanitise compound_id for use as a filename
        safe_id = str(compound_id).replace("/", "-").replace(" ", "_")[:60]
        yaml_path = self.yaml_dir / f"pxr_{ligand_idx:04d}_{safe_id}.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
        return yaml_path

    def run_boltz_prediction(self, yaml_file):
        """Run Boltz-2 structure + affinity prediction."""
        cmd = [
            "boltz", "predict", str(yaml_file),
            "--use_msa_server",
            "--cache", "~/.boltz",
            "--checkpoint", "/home/spal/.boltz/boltz2_conf.ckpt",
            "--diffusion_samples", "5",
            "--accelerator", "gpu",
            "--out_dir", str(self.out_dir)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            ok = result.returncode == 0
            print(f"  {'OK' if ok else 'FAIL'} (exit {result.returncode})")
            if not ok and result.stderr:
                print(f"  stderr: {result.stderr[-300:]}")
            return ok
        except subprocess.TimeoutExpired:
            print("  TIMEOUT (30 min)")
            return False
        except Exception as e:
            print(f"  ERROR: {e}")
            return False

    def parse_results(self):
        """Collect Boltz-2 affinity JSON outputs into a DataFrame."""
        results = []

        for json_file in self.out_dir.rglob("**/affinity*.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)

                pred_values = [
                    data.get("affinity_pred_value",  0),
                    data.get("affinity_pred_value1", 0),
                    data.get("affinity_pred_value2", 0)
                ]
                binary_probs = [
                    data.get("affinity_probability_binary",  0),
                    data.get("affinity_probability_binary1", 0),
                    data.get("affinity_probability_binary2", 0)
                ]

                mean_pic50 = np.mean(pred_values)
                # derive compound ID from yaml stem: pxr_0001_<ID>
                parts = json_file.stem.split("_")
                compound_id = "_".join(parts[2:]) if len(parts) > 2 else json_file.stem

                results.append({
                    "compound_ID":      compound_id,
                    "json_file":        str(json_file.relative_to(self.out_dir)),
                    "boltz_pIC50":      mean_pic50,
                    "boltz_IC50_uM":    10 ** (-mean_pic50),
                    "boltz_IC50_nM":    10 ** (-mean_pic50) * 1_000,
                    "binder_prob":      np.mean(binary_probs),
                    "pIC50_std":        np.std(pred_values),
                })
            except Exception as e:
                print(f"  Parse error {json_file}: {e}")

        return pd.DataFrame(results)

    def run(self, smiles_csv):
        """Full PXR Boltz-2 prediction pipeline."""
        print("PXR (NR1I2 / O75469) BOLTZ-2 BINDING-AFFINITY PREDICTIONS")
        print(f"Pocket residues : {[r[1] for r in PXR_POCKET]}")
        print(f"Output dir      : {self.out_dir}")

        df = self.parse_smiles_csv(smiles_csv)
        df.to_csv(self.out_dir / "input_ligands.csv", index=False)

        # Generate all YAMLs first
        yaml_files = []
        for idx, row in df.iterrows():
            yf = self.create_yaml(row["smiles_clean"], row["ID_clean"], idx)
            yaml_files.append((yf, row))
        print(f"\nGenerated {len(yaml_files)} YAML inputs → {self.yaml_dir}")

        # Run predictions
        print(f"\nRunning {len(yaml_files)} Boltz-2 predictions...")
        successful = 0
        for yaml_file, row in yaml_files:
            print(f"\n[{row['ID_clean']}]  {yaml_file.name}")
            if self.run_boltz_prediction(yaml_file):
                successful += 1
            time.sleep(1)  # MSA server rate-limit courtesy delay

        # Collect and report
        results_df = self.parse_results()
        if not results_df.empty:
            out_csv = self.out_dir / "pxr_boltz_predictions.csv"
            results_df.to_csv(out_csv, index=False)

            print(f"\nSUCCESSFUL: {successful}/{len(yaml_files)}")
            print("\nTop 5 predicted PXR binders:")
            top = results_df.nlargest(5, "boltz_pIC50")[
                ["compound_ID", "boltz_IC50_nM", "boltz_pIC50", "binder_prob"]
            ].round(2)
            print(top.to_string(index=False))
            top.to_csv(self.out_dir / "top_pxr_binders.csv", index=False)

        print(f"\nOUTPUT")
        print(f"  Predictions : {self.out_dir}/pxr_boltz_predictions.csv")
        print(f"  Top binders : {self.out_dir}/top_pxr_binders.csv")
        print(f"  Input copy  : {self.out_dir}/input_ligands.csv")
        print(f"  YAMLs       : {self.yaml_dir}")
        print(f"  Structures  : {self.out_dir}")

        return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PXR (NR1I2) Boltz-2 binding-affinity predictions"
    )
    parser.add_argument(
        "smiles_csv",
        help=(
            "Input CSV. Two formats accepted:\n"
            "  1) smiles,ID  (no header, simple)\n"
            "  2) structure,smiles,Molecule Name,OCNT_ID  (OpenAdmet header)"
        )
    )
    parser.add_argument(
        "--out_dir", default="pxr_boltz_results",
        help="Output directory (default: pxr_boltz_results)"
    )
    args = parser.parse_args()

    predictor = PXRBoltzPredictor(args.out_dir)
    results = predictor.run(args.smiles_csv)
