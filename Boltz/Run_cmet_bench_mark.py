#!/usr/bin/env python3
"""
Batch Boltz-2 for cMet Merck FEP dataset
Input: smiles,ID (with IC50 in ID suffix)
Output: predictions.csv with correlation analysis
"""

import pandas as pd
import yaml
import subprocess
import json
import os
import time
from pathlib import Path
import numpy as np
from scipy.stats import pearsonr
import argparse
import re

CMET_SEQUENCE = """MASQPNSSAKKKEEKGKNIQVVVRCRPFNLAERKASAHSIVECDPVRKEVSVRTGGLADKSSRKTYTFDMVFGASTKQIDVYRSVVCPILDEVIMGYNCTIFAYGQTGTGKTFTMEGERSPNEEYTWEEDPLAGIIPRTLHQIFEKLTDNGTEFSVKVSLLEIYNEELFDLLNPSSDVSERLQMFDDPRNKRGVIIKGLEEITVHNKDEVYQILEKGAAKRTTAATLMNAYSSRSHSVFSVTIHMKETTIDGEELVKIGKLNLVDLAGSENIGRSGAVDKRAREAGNINQSLLTLGRVITALVERTPHVPYRESKLTRILQDSLGGRTRTSIIATISPASLNLEETLSTLEYAHRAKNILNKPEVNQK"""

class CMetBoltzPredictor:
    def __init__(self, out_dir="cmet_boltz_results"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(exist_ok=True)
        self.yaml_dir = self.out_dir / "yamls"
        self.yaml_dir.mkdir(exist_ok=True)
        
    def parse_smiles_csv(self, smiles_csv):
        """Parse your specific format: smiles,ID where ID=CHEMBLxxx_IC50"""
        df = pd.read_csv(smiles_csv, header=None, names=['smiles', 'ID'])
        
        # Extract CHEMBL ID and IC50 from ID column
        df[['chembl_id', 'exp_ic50_nM']] = df['ID'].str.extract(r'(CHEMBL\d+)[_-]?(\d+\.?\d*)')
        df['exp_ic50_nM'] = pd.to_numeric(df['exp_ic50_nM'], errors='coerce')
        df['exp_pIC50'] = -np.log10(df['exp_ic50_nM'] / 1e9)  # Convert nM → M
        
        # Clean SMILES (remove explicit hydrogens for Boltz)
   #     df['smiles_clean'] = df['smiles'].str.replace(r'\[H\]', '', regex=True)
        df['smiles_clean'] = df['smiles']
        
        print(f"Loaded {len(df)} cMet ligands")
        print(f"IC50 range: {df['exp_ic50_nM'].min():.1f} - {df['exp_ic50_nM'].max():.1f} nM")
        return df.dropna(subset=['exp_ic50_nM'])
   

    def create_yaml(self, smiles, chembl_id, ligand_idx):
        """Create single-ligand YAML"""
        yaml_data = {
            "version": 1,
            "sequences": [
                {"protein": {"id": "A", "sequence": CMET_SEQUENCE}},
                {"ligand": {"id": "B", "smiles": smiles}}
            ],
            "properties":[
                {"affinity": {"binder": "B"}}
           ]
        }
        
        yaml_path = self.yaml_dir / f"cmet_{ligand_idx:03d}_{chembl_id}.yaml"
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

        print(cmd)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=18000)
        print(result)

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            return result.returncode == 0
        except:
            return False
    
    def parse_affinity(self):
        """Parse all affinity JSONs"""
        results = []
        for json_file in self.out_dir.glob("**/affinity*cmet*.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                
                # Ensemble average
                pred_values = [data.get('affinity_pred_value', 0),
                              data.get('affinity_pred_value1', 0), 
                              data.get('affinity_pred_value2', 0)]
                binary_probs = [data.get('affinity_probability_binary', 0),
                               data.get('affinity_probability_binary1', 0),
                               data.get('affinity_probability_binary2', 0)]
                
                chembl_id = json_file.stem.split('_')[-1].replace('.json', '')
                results.append({
                    'chembl_id': chembl_id,
                    'json_file': json_file.name,
                    'boltz_pIC50': np.mean(pred_values),
                    'boltz_IC50_uM': 10**np.mean(pred_values),
                    'boltz_IC50_nM': 10**np.mean(pred_values) * 1000,
                    'boltz_binder_prob': np.mean(binary_probs)
                })
            except:
                continue
        
        return pd.DataFrame(results)
    
    def run_full_benchmark(self, smiles_csv):
        """Complete pipeline"""
        print("🔬 cMet Merck FEP Benchmark with Boltz-2")
        
        # Parse input
        df = self.parse_smiles_csv(smiles_csv)
        df.to_csv(self.out_dir / "input_ligands.csv", index=False)
        
        # Create YAMLs
        yaml_files = []
        for idx, row in df.iterrows():
            yaml_file = self.create_yaml(row['smiles_clean'], row['chembl_id'], idx)
            yaml_files.append((yaml_file, row))
        
        # Run predictions
        print(f"\n⚡ Running {len(yaml_files)} Boltz predictions...")
        successful = 0
        for yaml_file, row in yaml_files:
            print(yaml_file)
            print(f"  {row['chembl_id']} (IC50 {row['exp_ic50_nM']:.1f}nM)... ", end="")
            if self.run_boltz(yaml_file):
                print("✓")
                successful += 1
            else:
                print("✗")
            time.sleep(1)  # MSA server politeness
        
        # Parse results
        results = self.parse_affinity()

        print(results)
#        print(chembl_id)
        merged = df.merge(results, on='chembl_id', how='inner')
        
        # Correlation analysis
        r, p = pearsonr(merged['boltz_pIC50'], merged['exp_pIC50'])
        
        # Save
        merged.to_csv(self.out_dir / "cmet_boltz_results.csv", index=False)
        
        # Summary
        print(f"\n📊 BENCHMARK RESULTS")
        print(f"✅ Successful predictions: {successful}/{len(yaml_files)}")
        print(f"📈 Pearson r = {r:.3f} (p={p:.3f})")
        print(f"🔥 Top Boltz predictions:")
        top_boltz = merged.nlargest(5, 'boltz_pIC50')[['chembl_id', 'boltz_IC50_nM', 'exp_ic50_nM']]
        print(top_boltz.round(1))
        
        return merged

if __name__ == "__main__":
    parser = parser = argparse.ArgumentParser()
    parser.add_argument("smiles_csv", help="Your SMILES CSV file")
    parser.add_argument("--out_dir", default="cmet_results", help="Output directory")
    args = parser.parse_args()
    
    predictor = CMetBoltzPredictor(args.out_dir)
    results = predictor.run_full_benchmark(args.smiles_csv)

