#!/usr/bin/env python3
"""
Batch Boltz-2 predictions for CYP3A4 inhibitors.

For each inhibitor in the SMILES CSV, generates a YAML containing:
  - Chain A: CYP3A4 protein (9MS1 sequence)
  - Chain L: inhibitor (one per run)
  - Chain H: heme cofactor (hardcoded HEME_SMILES, present in every run)

Activation:
    conda run -n boltz2 python run_boltz_cyp3a4.py [smiles_csv] [--out_dir DIR]

Defaults to smiles_cyp3a4_inhibitors.csv in the working directory.
"""

import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import subprocess

# Heme B cofactor SMILES (iron-porphyrin with two propionate side-chains)
HEME_SMILES = "C=CC1=C(C)C2=Cc3c(C)c(CCC(=O)O)c4n3[Fe@SP2]35<-N6=C(C=c7c(C=C)c(C)c(n73)=CC1=N->52)C(C)=C(CCC(=O)O)C6=C4"

# CYP3A4 sequence from PDB 9MS1, Chain A (includes C-terminal His-tag from crystallisation construct)
CYP3A4_SEQUENCE = (
    "MALYGTHSHGLFKKLGIPGPTPLPFLGNILSYHKGFCMFDMECHKKYGKVWGFYDGQQPVLAITDPDMIKTVLVKECYSVFTNRRPFGPVGFM"
    "KSAISIAEDEEWKRLRSLLSPTFTSGKLKEMVPIIAQYGDVLVRNLRREAETGKPVTLKDVFGAYSMDVITSTSFGVNIDSLNNPQDPFVENTK"
    "KLLRFDFLDPFFLSITVFPFLIPILEVLNICVFPREVTNFLRKSVKRMKESRLEDTQKHRVDFLQLMIDSQNSKETESHKALSDLELVAQSIIFI"
    "FAGYETTSSVLSFIMYELATHPDVQQKLQEEIDAVLPNKAPPTYDTVLQMEYLDMVVNETLRLFPIAMRLERVCKKDVEINGMFIPKGVVVMIPS"
    "YALHRDPKYWTEPEKFLPERFSKKNKDNIDPYIYTPFGSGPRNCIGMRFALMNMKLALIRVLQNFSFKPCKETQIPLKLSLGGLLQPEKPVVLKV"
    "ESRDGTVSGAHHHH"
)

NUM_DIFFUSION_SAMPLES = 1


class Cyp3A4BoltzPredictor:
    def __init__(self, out_dir="cyp3a4_boltz_results"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(exist_ok=True)
        self.yaml_dir = self.out_dir / "yamls"
        self.yaml_dir.mkdir(exist_ok=True)

    def parse_smiles_csv(self, smiles_csv):
        """Read all rows from CSV (smiles,ID) as inhibitors; heme is hardcoded."""
        df = pd.read_csv(smiles_csv)
        df.columns = [c.strip() for c in df.columns]
        df = df.dropna(subset=["smiles"]).copy()
        df["smiles"] = df["smiles"].astype(str).str.strip()
        df["ID"] = df["ID"].astype(str).str.strip()
        df = df.reset_index(drop=True)

        print(f"Heme cofactor  : hardcoded ({HEME_SMILES[:60]}...)")
        print(f"Inhibitors ({len(df)}): {df['ID'].tolist()}")
        return df

    def create_yaml(self, inhibitor_smiles, inhibitor_id, idx, heme_smiles=HEME_SMILES):
        """Write a Boltz YAML: CYP3A4 (A) + inhibitor (L) + heme (H)."""
        yaml_data = {
            "version": 1,
            "sequences": [
                {"protein": {"id": "A", "sequence": CYP3A4_SEQUENCE}},
                {"ligand": {"id": "L", "smiles": inhibitor_smiles}},
                {"ligand": {"id": "H", "smiles": heme_smiles}},
            ],
            "properties": [
                {"affinity": {"binder": "L"}}
            ],
        }

        yaml_path = self.yaml_dir / f"cyp3a4_{idx:03d}_{inhibitor_id}.yaml"
        with open(yaml_path, "w") as fh:
            yaml.dump(yaml_data, fh, default_flow_style=False, sort_keys=False)
        return yaml_path

    def run_boltz(self, yaml_file):
        """Execute a single Boltz prediction job."""
        cmd = [
            "boltz", "predict", str(yaml_file),
            "--use_msa_server",
            "--use_potentials",
            "--cache", "~/.boltz",
            "--checkpoint", "/home/spal/.boltz/boltz2_conf.ckpt",
            "--accelerator", "gpu",
            "--out_dir", str(self.out_dir),
            "--diffusion_samples", str(NUM_DIFFUSION_SAMPLES),
        ]
        print(f"  cmd: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            ok = result.returncode == 0
            print(f"  exit: {result.returncode}  ({'OK' if ok else 'FAILED'})")
            if not ok:
                print(f"  stderr: {result.stderr[-600:]}")
            return ok
        except subprocess.TimeoutExpired:
            print("  TIMEOUT (1800 s)")
            return False
        except Exception as exc:
            print(f"  ERROR: {exc}")
            return False

    def parse_affinity_results(self):
        """Collect and average affinity JSON outputs from all completed runs."""
        rows = []
        for json_file in sorted(self.out_dir.glob("**/affinity*cyp3a4*.json")):
            try:
                with open(json_file) as fh:
                    data = json.load(fh)

                pred_vals = [data.get("affinity_pred_value", np.nan)]
                bin_probs = [data.get("affinity_probability_binary", np.nan)]
                for i in range(1, NUM_DIFFUSION_SAMPLES):
                    pred_vals.append(data.get(f"affinity_pred_value{i}", np.nan))
                    bin_probs.append(data.get(f"affinity_probability_binary{i}", np.nan))

                pred_vals = [v for v in pred_vals if not np.isnan(v)]
                bin_probs  = [v for v in bin_probs  if not np.isnan(v)]

                # YAML stem: cyp3a4_000_<inhibitorID>  →  drop first two tokens
                ligand_id = "_".join(json_file.stem.split("_")[2:])
                rows.append({
                    "ID":                   ligand_id,
                    "json_file":            json_file.name,
                    "n_samples":            len(pred_vals),
                    "boltz_pIC50_mean":     np.mean(pred_vals),
                    "boltz_pIC50_std":      np.std(pred_vals),
                    "boltz_IC50_uM":        10 ** np.mean(pred_vals),
                    "boltz_IC50_nM":        10 ** np.mean(pred_vals) * 1000,
                    "boltz_binder_prob":    np.mean(bin_probs),
                })
            except Exception as exc:
                print(f"Parse error {json_file}: {exc}")
        return pd.DataFrame(rows)

    def run(self, smiles_csv):
        """Full pipeline: parse → YAML → predict → collect results."""
        print("=" * 60)
        print("CYP3A4 Boltz-2 predictions  (inhibitor + heme cofactor)")
        print("=" * 60)

        inhibitors = self.parse_smiles_csv(smiles_csv)
        inhibitors.to_csv(self.out_dir / "input_inhibitors.csv", index=False)

        # Generate YAMLs
        yaml_files = []
        for idx, row in inhibitors.iterrows():
            yf = self.create_yaml(row["smiles"], row["ID"], idx)
            yaml_files.append((yf, row))
            print(f"  YAML: {yf.name}")

        # Run predictions
        print(f"\nLaunching {len(yaml_files)} Boltz jobs ...")
        successful = 0
        for yf, row in yaml_files:
            print(f"\n[{row['ID']}]  {yf.name}")
            if self.run_boltz(yf):
                successful += 1
            time.sleep(1)   # MSA-server politeness delay

        # Collect results
        results = self.parse_affinity_results()
        out_csv = self.out_dir / "cyp3a4_boltz_predictions.csv"
        results.to_csv(out_csv, index=False)

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Jobs completed: {successful}/{len(yaml_files)}")
        if not results.empty:
            print("\nResults (sorted by pIC50):")
            cols = ["ID", "n_samples", "boltz_pIC50_mean", "boltz_pIC50_std",
                    "boltz_IC50_nM", "boltz_binder_prob"]
            print(results.sort_values("boltz_pIC50_mean", ascending=False)[cols]
                  .round(3).to_string(index=False))
        else:
            print("No affinity JSON files found yet — check individual run logs.")

        print(f"\nOutput directory : {self.out_dir.resolve()}")
        print(f"Predictions CSV  : {out_csv.resolve()}")
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CYP3A4 Boltz-2 predictions — each inhibitor run with the heme cofactor"
    )
    parser.add_argument(
        "smiles_csv",
        nargs="?",
        default="smiles_cyp3a4_inhibitors.csv",
        help="SMILES CSV with header 'smiles,ID'; all rows are treated as inhibitors",
    )
    parser.add_argument(
        "--out_dir",
        default="cyp3a4_boltz_results",
        help="Output directory (default: cyp3a4_boltz_results)",
    )
    args = parser.parse_args()

    predictor = Cyp3A4BoltzPredictor(args.out_dir)
    predictor.run(args.smiles_csv)
