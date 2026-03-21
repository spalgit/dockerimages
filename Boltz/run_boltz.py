#!/usr/bin/env python3
"""
Batch Boltz-2 for NCK1 (NO experimental data needed)
Input: smiles,ID (ID can be any identifier)
Output: predictions.csv with pIC50, IC50 predictions
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

NCK1_SEQUENCE = """GGNPWYYGKVTRHQAEMALNERGHEGDFLIRDSESSPNDFSVSLKAQGKNKHFKVQLKETVYCIGQRKFSTMEELVEHYKKAPIFTSEQGEKLYLVKHLS"""

class SOCS3BoltzPredictor:
    def __init__(self, out_dir="nck1_boltz_results"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(exist_ok=True)
        self.yaml_dir = self.out_dir / "yamls"
        self.yaml_dir.mkdir(exist_ok=True)

    def parse_smiles_csv(self, smiles_csv):
        """Parse simple format: smiles,ID (ID=any identifier, no IC50 needed)"""
        df = pd.read_csv(smiles_csv, header=None, names=['smiles', 'ID'])

        # Use ID as-is (no IC50 parsing)
        df['ID_clean'] = df['ID'].astype(str)
        df['smiles_clean'] = df['smiles'].astype(str)

        print(f"Loaded {len(df)} NCK1 ligands")
        print(f"First few IDs: {df['ID_clean'].head().tolist()}")
        return df

    def create_yaml(self, smiles, ligand_id, ligand_idx):
        """Create single-ligand YAML for NCK1"""
        yaml_data = {
            "version": 1,
            "sequences": [
                {"protein": {"id": "A", "sequence": NCK1_SEQUENCE}},
                {"ligand": {"id": "L", "smiles": smiles}}
            ],
            "properties": [
                {"affinity": {"binder": "L"}}
            ]
        }

        yaml_path = self.yaml_dir / f"nck1_{ligand_idx:03d}_{ligand_id}.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_data, f, default_flow_style=False)
        return yaml_path

    def run_boltz(self, yaml_file):
        """Run Boltz prediction"""
        cmd = [
            "boltz", "predict", str(yaml_file),
            "--use_msa_server", "--cache", "~/.boltz", "--checkpoint", "/home/spal/.boltz/boltz2_conf.ckpt",
            "--accelerator", "gpu", "--out_dir", str(self.out_dir)
        ]

        print(f"  Running: {cmd}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            print(f"  Status: {'✓' if result.returncode == 0 else '✗'} ({result.returncode})")
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            print("  TIMEOUT")
            return False
        except Exception as e:
            print(f"  ERROR: {e}")
            return False

    def parse_affinity(self):
        """Parse all affinity JSONs"""
        results = []
        for json_file in self.out_dir.glob("**/affinity*nck1*.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)

                # Ensemble average (3 predictions)
                pred_values = [data.get('affinity_pred_value', 0),
                              data.get('affinity_pred_value1', 0),
                              data.get('affinity_pred_value2', 0)]
                binary_probs = [data.get('affinity_probability_binary', 0),
                               data.get('affinity_probability_binary1', 0),
                               data.get('affinity_probability_binary2', 0)]

                nck1_id = json_file.stem.split('_')[-1].replace('.json', '')
                results.append({
                    'ID': nck1_id,
                    'json_file': json_file.name,
                    'boltz_pIC50': np.mean(pred_values),
                    'boltz_IC50_uM': 10**np.mean(pred_values),
                    'boltz_IC50_nM': 10**np.mean(pred_values) * 1000,
                    'boltz_binder_prob': np.mean(binary_probs),
                    'pred_std': np.std(pred_values)
                })
            except Exception as e:
                print(f"Parse error {json_file}: {e}")
                continue

        return pd.DataFrame(results)

    def run_full_benchmark(self, smiles_csv):
        """Complete NCK1 prediction pipeline"""
        print("🔬 NCK1 Boltz-2 Binding Affinity Predictions")

        # Parse input
        df = self.parse_smiles_csv(smiles_csv)
        df.to_csv(self.out_dir / "input_ligands.csv", index=False)

        # Create YAMLs
        yaml_files = []
        for idx, row in df.iterrows():
            yaml_file = self.create_yaml(row['smiles_clean'], row['ID_clean'], idx)
            yaml_files.append((yaml_file, row))

        # Run predictions
        print(f"\n⚡ Running {len(yaml_files)} NCK1 predictions...")
        successful = 0
        for yaml_file, row in yaml_files:
            print(f"\n{yaml_file}")
            print(f"  {row['ID_clean']}... ", end="")
            if self.run_boltz(yaml_file):
                print("✓")
                successful += 1
            else:
                print("✗")
            time.sleep(1)  # MSA server politeness

        # Parse results
        results = self.parse_affinity()
        results.to_csv(self.out_dir / "nck1_boltz_predictions.csv", index=False)

        print("\n📊 NCK1 PREDICTIONS SUMMARY")
        print(f"✅ Successful: {successful}/{len(yaml_files)}")
        print(f"💎 Top 5 binders (highest pIC50):")
        top_binders = results.nlargest(5, 'boltz_pIC50')[['ID', 'boltz_IC50_nM', 'boltz_pIC50', 'boltz_binder_prob']]
        print(top_binders.round(1))

        print(f"\n💾 Results saved:")
        print(f"  - {self.out_dir}/nck1_boltz_predictions.csv")
        print(f"  - {self.out_dir}/yamls/*.yaml")
        print(f"  - {self.out_dir}/input_ligands.csv")

        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NCK1 Boltz-2 predictions")
    parser.add_argument("smiles_csv", help="SMILES CSV: smiles,ID")
    parser.add_argument("--out_dir", default="nck1_results", help="Output directory")
    args = parser.parse_args()

    predictor = SOCS3BoltzPredictor(args.out_dir)
    predictor.run_full_benchmark(args.smiles_csv)

