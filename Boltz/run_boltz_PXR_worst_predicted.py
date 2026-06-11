#!/usr/bin/env python3
"""
PXR Boltz-2 affinity predictions for worst-predicted compounds and their
training set analogues.

Input  : worst_predicted_with_analogues.csv
         Columns: ID, SMILES, Train_or_Test, pEC50, Test_ID, pEC50_test
Output : pxr_worst_boltz_results/pxr_worst_boltz_predictions.csv

Target
  PXR = Nuclear receptor subfamily 1 group I member 2 (NR1I2)
        UniProt O75469 (NR1I2_HUMAN), 434 aa canonical isoform 1

Affinity output note
  Boltz-2 runs affinity via a dedicated two-module ensemble head, separate
  from the 5 structural diffusion samples.  Three values are written per
  compound regardless of --diffusion_samples:
    affinity_pred_value   — ensemble mean of modules 1 & 2  (primary)
    affinity_pred_value1  — module 1 raw prediction
    affinity_pred_value2  — module 2 raw prediction
  All three are log10(IC50 / µM).  The spread between mod1 and mod2 is the
  built-in uncertainty estimate.  There is no per-structural-pose affinity.

Skip logic
  If the affinity JSON already exists for a compound the run is skipped so
  the script can be restarted safely without repeating work.

CIF compound identification
  YAMLs are named <compound_id>.yaml so Boltz names every output file
  (CIF, JSON, npz) after the compound ID.  Additionally, _entry.id and
  _struct.title are patched into each CIF after generation.

Usage:
    conda activate boltz
    python run_boltz_PXR_worst_predicted.py \\
        --input_csv  ~/OpenAdmet_After_Phase1/worst_predicted_with_analogues.csv \\
        --out_dir    pxr_worst_boltz_results
"""

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


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

# ── PXR ligand-binding pocket (1ILG contacts, full-protein numbering) ──────────
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


def _safe_id(compound_id: str) -> str:
    """Make a compound ID safe for use as a filename."""
    return re.sub(r"[^\w\-]", "_", str(compound_id).strip())[:80]


class PXRWorstPredictedBoltz:

    def __init__(self, out_dir: str = "pxr_worst_boltz_results"):
        self.out_dir  = Path(out_dir)
        self.yaml_dir = self.out_dir / "yamls"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.yaml_dir.mkdir(exist_ok=True)

    # ── Skip check ─────────────────────────────────────────────────────────────
    def _affinity_json_path(self, safe: str) -> Path:
        """Expected location of the affinity JSON for a given compound safe-ID."""
        return (
            self.out_dir
            / f"boltz_results_{safe}"
            / "predictions"
            / safe
            / f"affinity_{safe}.json"
        )

    def _already_done(self, safe: str) -> bool:
        return self._affinity_json_path(safe).exists()

    # ── Input ──────────────────────────────────────────────────────────────────
    def load_input(self, csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        required = {"ID", "SMILES", "Train_or_Test", "pEC50"}
        missing  = required - set(df.columns)
        if missing:
            raise ValueError(f"Input CSV missing columns: {missing}")
        df = df.dropna(subset=["SMILES"])
        df = df[df["SMILES"].astype(str).str.strip().ne("")]
        df["safe_id"] = df["ID"].apply(_safe_id)

        n_test  = (df["Train_or_Test"] == "Test").sum()
        n_train = (df["Train_or_Test"] == "Training").sum()
        print(f"Loaded {len(df)} compounds  ({n_test} test / {n_train} training analogues)")
        return df.reset_index(drop=True)

    # ── YAML creation ──────────────────────────────────────────────────────────
    def create_yaml(self, smiles: str, safe: str) -> Path:
        """
        Create Boltz-2 YAML named <safe_id>.yaml.

        Boltz derives every output filename from the YAML stem, so naming the
        YAML after the compound ID means all CIFs, JSONs and npz files are
        automatically labelled with the compound ID.
        """
        yaml_data = {
            "version": 1,
            "sequences": [
                {
                    "protein": {
                        "id":       "A",
                        "sequence": PXR_SEQUENCE,
                        "msa":      "empty",
                    }
                },
                {
                    "ligand": {
                        "id":     "L",
                        "smiles": str(smiles).strip(),
                    }
                },
            ],
            "properties": [
                {"affinity": {"binder": "L"}}
            ],
            "constraints": [
                {"pocket": {"binder": "L", "contacts": PXR_POCKET}}
            ],
        }
        yaml_path = self.yaml_dir / f"{safe}.yaml"
        with open(yaml_path, "w") as fh:
            yaml.dump(yaml_data, fh, default_flow_style=False, sort_keys=False)
        return yaml_path

    # ── CIF patching ───────────────────────────────────────────────────────────
    def _patch_cif_ids(self, safe: str, compound_id: str):
        """
        Replace _entry.id, _struct.entry_id and _struct.title in every CIF
        produced for this compound so the compound ID is embedded in the file.
        """
        pred_dir = (
            self.out_dir / f"boltz_results_{safe}" / "predictions" / safe
        )
        for cif_path in pred_dir.glob("*.cif"):
            try:
                text = cif_path.read_text()
                text = re.sub(r"(_entry\.id\s+)\S+",        rf"\g<1>{compound_id}", text)
                text = re.sub(r"(_struct\.entry_id\s+)\S+", rf"\g<1>{compound_id}", text)
                text = re.sub(r"(_struct\.title\s+)\S+",    rf"\g<1>'{compound_id}'", text)
                cif_path.write_text(text)
            except Exception as exc:
                print(f"    Warning: could not patch {cif_path.name}: {exc}")

    # ── Boltz-2 run ────────────────────────────────────────────────────────────
    def run_boltz(self, yaml_file: Path) -> bool:
        cmd = [
            "boltz", "predict", str(yaml_file),
            "--use_msa_server",
            "--cache",      os.path.expanduser("~/.boltz"),
            "--checkpoint", os.path.expanduser("~/.boltz/boltz2_conf.ckpt"),
            "--use_potentials",
            "--diffusion_samples",          "5",
            "--diffusion_samples_affinity", "5",
            "--accelerator", "gpu",
            "--out_dir",     str(self.out_dir),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            ok = result.returncode == 0
            print(f"  {'OK' if ok else 'FAIL'} (exit {result.returncode})")
            if not ok and result.stderr:
                print(f"  stderr: {result.stderr[-400:]}")
            return ok
        except subprocess.TimeoutExpired:
            print("  TIMEOUT (30 min)")
            return False
        except Exception as exc:
            print(f"  ERROR: {exc}")
            return False

    # ── Result parsing ─────────────────────────────────────────────────────────
    def parse_results(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """
        Read every affinity JSON under out_dir and merge with input metadata.

        Output columns:
          ID, SMILES, Train_or_Test, pEC50_exp, Test_ID, pEC50_test
          boltz_log10IC50_uM   — ensemble mean  (primary, log10 µM)
          boltz_pIC50          — -log10 IC50 (same scale as pEC50)
          boltz_IC50_nM        — IC50 in nM
          binder_prob          — P(binder), ensemble mean
          boltz_log10IC50_mod1 — module 1 raw value
          boltz_log10IC50_mod2 — module 2 raw value
          binder_prob_mod1/2   — per-module binder probabilities
          affinity_mod_spread  — |mod1 - mod2| uncertainty estimate
        """
        id_lookup = {row["safe_id"]: row for _, row in input_df.iterrows()}

        records = []
        for json_file in sorted(self.out_dir.rglob("affinity*.json")):
            try:
                with open(json_file) as fh:
                    data = json.load(fh)

                # safe_id is the parent directory name
                safe = json_file.parent.name

                v0 = data.get("affinity_pred_value",  np.nan)
                v1 = data.get("affinity_pred_value1", np.nan)
                v2 = data.get("affinity_pred_value2", np.nan)
                b0 = data.get("affinity_probability_binary",  np.nan)
                b1 = data.get("affinity_probability_binary1", np.nan)
                b2 = data.get("affinity_probability_binary2", np.nan)

                def fmt(v): return round(float(v), 4) if not np.isnan(v) else ""

                rec = {
                    "ID":                    safe,
                    "boltz_log10IC50_uM":    fmt(v0),
                    "boltz_pIC50":           fmt(-v0) if v0 != "" and not np.isnan(v0) else "",
                    "boltz_IC50_nM":         round(10**(-v0) * 1000, 2) if not np.isnan(v0) else "",
                    "binder_prob":           fmt(b0),
                    "boltz_log10IC50_mod1":  fmt(v1),
                    "boltz_log10IC50_mod2":  fmt(v2),
                    "binder_prob_mod1":      fmt(b1),
                    "binder_prob_mod2":      fmt(b2),
                    "affinity_mod_spread":   round(abs(v1 - v2), 4)
                                             if not (np.isnan(v1) or np.isnan(v2)) else "",
                }

                if safe in id_lookup:
                    row = id_lookup[safe]
                    rec["SMILES"]        = row["SMILES"]
                    rec["Train_or_Test"] = row["Train_or_Test"]
                    rec["pEC50_exp"]     = row["pEC50"]
                    rec["Test_ID"]       = row.get("Test_ID", "")
                    rec["pEC50_test"]    = row.get("pEC50_test", "")

                records.append(rec)

            except Exception as exc:
                print(f"  Parse error {json_file}: {exc}")

        # Reorder columns
        meta_cols   = ["ID", "SMILES", "Train_or_Test", "pEC50_exp",
                       "Test_ID", "pEC50_test"]
        result_cols = ["boltz_log10IC50_uM", "boltz_pIC50", "boltz_IC50_nM",
                       "binder_prob",
                       "boltz_log10IC50_mod1", "boltz_log10IC50_mod2",
                       "binder_prob_mod1", "binder_prob_mod2",
                       "affinity_mod_spread"]
        df = pd.DataFrame(records)
        cols = [c for c in meta_cols + result_cols if c in df.columns]
        return df[cols] if not df.empty else df

    # ── Main pipeline ──────────────────────────────────────────────────────────
    def run(self, input_csv: str):
        print("PXR BOLTZ-2 — WORST-PREDICTED COMPOUNDS + TRAINING ANALOGUES")
        print(f"Pocket residues : {[r[1] for r in PXR_POCKET]}")
        print(f"Output dir      : {self.out_dir}\n")

        df = self.load_input(input_csv)
        df.to_csv(self.out_dir / "input_compounds.csv", index=False)

        # Generate YAMLs (only for compounds not already done)
        todo, skipped = [], []
        for _, row in df.iterrows():
            safe = row["safe_id"]
            if self._already_done(safe):
                skipped.append(row["ID"])
            else:
                yf = self.create_yaml(row["SMILES"], safe)
                todo.append((yf, row))

        if skipped:
            print(f"Skipping {len(skipped)} already-completed compounds:")
            for s in skipped:
                print(f"  {s}")
            print()

        print(f"Generated {len(todo)} YAML inputs to run → {self.yaml_dir}\n")

        # Run Boltz-2
        n_ok = 0
        for i, (yaml_file, row) in enumerate(todo):
            print(f"[{i+1}/{len(todo)}] [{row['Train_or_Test']}] {row['ID']}")
            if self.run_boltz(yaml_file):
                n_ok += 1
                self._patch_cif_ids(row["safe_id"], row["ID"])
            time.sleep(1)

        # Collect and save results (all runs, including previously completed)
        results = self.parse_results(df)
        if not results.empty:
            out_csv = self.out_dir / "pxr_worst_boltz_predictions.csv"
            results.to_csv(out_csv, index=False)
            print(f"\nSuccessful this run : {n_ok}/{len(todo)}")
            print(f"Total results       : {len(results)}")
            print(f"Saved               : {out_csv}\n")

            # Summary: top predicted binders among test compounds
            test_res = results[results.get("Train_or_Test", pd.Series(dtype=str)) == "Test"].copy()
            if not test_res.empty:
                test_res["boltz_pIC50"] = pd.to_numeric(
                    test_res["boltz_pIC50"], errors="coerce"
                )
                top = (test_res.dropna(subset=["boltz_pIC50"])
                               .nlargest(5, "boltz_pIC50")
                               [["ID", "pEC50_exp", "boltz_pIC50",
                                 "boltz_IC50_nM", "binder_prob", "affinity_mod_spread"]])
                print("Top 5 predicted binders among test compounds:")
                print(top.to_string(index=False))
        else:
            print(f"\nSuccessful : {n_ok}/{len(todo)} — no JSON results found yet.")

        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PXR Boltz-2 predictions for worst-predicted + training analogues"
    )
    parser.add_argument(
        "--input_csv",
        default=os.path.expanduser(
            "~/OpenAdmet_After_Phase1/worst_predicted_with_analogues.csv"
        ),
        help="Input CSV (default: ~/OpenAdmet_After_Phase1/worst_predicted_with_analogues.csv)",
    )
    parser.add_argument(
        "--out_dir", default="pxr_worst_boltz_results",
        help="Output directory (default: pxr_worst_boltz_results)",
    )
    args = parser.parse_args()

    predictor = PXRWorstPredictedBoltz(out_dir=args.out_dir)
    predictor.run(args.input_csv)
