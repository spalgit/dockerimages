"""
Convert the ligprepped SDF files to CSV for use with OpenAdmet YAML configs.

Outputs:
  f1_train_4392.csv  — training set including Phase 1 experimental results
  f2_train_4139.csv  — training set WITHOUT Phase 1 experimental results
  test_phase2.csv    — Phase 2 test set (260 compounds)

Usage:
    conda activate openadmet
    python convert_sdf_to_csv.py
"""

from pathlib import Path

import pandas as pd
from rdkit import Chem

HERE = Path("/home/spal/dockerimages/QSARTuna/PXR/June_14_ligprepped")

REQUIRED_TAGS = ["pEC50"]
OPTIONAL_TAGS = ["pEC50_counter"]


def sdf_to_csv(sdf_path: Path, csv_path: Path) -> pd.DataFrame:
    suppl = Chem.SDMolSupplier(str(sdf_path), removeHs=True)
    rows, skipped = [], 0
    for mol in suppl:
        if mol is None:
            skipped += 1
            continue
        if not all(mol.HasProp(t) for t in REQUIRED_TAGS):
            skipped += 1
            continue
        row = {
            "SMILES": Chem.MolToSmiles(mol),
            "Name": mol.GetProp("_Name") if mol.HasProp("_Name") else "",
        }
        for tag in REQUIRED_TAGS + OPTIONAL_TAGS:
            if mol.HasProp(tag):
                try:
                    row[tag] = float(mol.GetProp(tag))
                except (ValueError, TypeError):
                    row[tag] = None
            else:
                row[tag] = None
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"  {csv_path.name}: {len(df)} molecules written  ({skipped} skipped)")
    return df


print("Converting SDF files to CSV for OpenAdmet...")
print()

sdf_to_csv(
    HERE / "train_set_AND_phase_one_results_4392_ligpreped_f_1.sdf",
    HERE / "f1_train_4392.csv",
)
sdf_to_csv(
    HERE / "train_set_4139_ligpreped_f_2.sdf",
    HERE / "f2_train_4139.csv",
)
sdf_to_csv(
    HERE / "test_phase2_ligprepped_f_2.sdf",
    HERE / "test_phase2.csv",
)

print()
print("Done. CSV files ready for OpenAdmet YAML configs.")
