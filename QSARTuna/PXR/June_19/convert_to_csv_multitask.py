"""
Convert June_19 SDF files to CSV for the multitask LGBM (ERG) model.

Outputs:
  f1_plus_htchem_train_4827_multitask.csv   -- SMILES, pEC50, pEC50_counter
  test_pahse2_ligprep_redone_smiles.csv     -- SMILES only (blinded test, no labels)

Run:
  conda activate openadmet-models
  cd ~/dockerimages/QSARTuna/PXR/June_19
  python convert_to_csv_multitask.py
"""

import pandas as pd
from rdkit import Chem

HERE = "/home/spal/dockerimages/QSARTuna/PXR/June_19"


def sdf_to_df(path, include_targets=True):
    rows = []
    for mol in Chem.SDMolSupplier(path):
        if mol is None:
            continue
        props = mol.GetPropsAsDict()
        row = {"SMILES": props.get("SMILES") or Chem.MolToSmiles(mol)}
        if include_targets:
            row["pEC50"] = props.get("pEC50", float("nan"))
            row["pEC50_counter"] = props.get("pEC50_counter", float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)


train_df = sdf_to_df(f"{HERE}/f1_plus_htchem_train_4827.sdf", include_targets=True)
print(
    f"Training set: {len(train_df)} compounds, "
    f"{train_df['pEC50'].notna().sum()} with pEC50, "
    f"{train_df['pEC50_counter'].notna().sum()} with pEC50_counter"
)
train_df.to_csv(f"{HERE}/f1_plus_htchem_train_4827_multitask.csv", index=False)

test_df = sdf_to_df(f"{HERE}/test_pahse2_ligprep_redone.sdf", include_targets=False)
print(f"Test set: {len(test_df)} compounds (blinded, no labels)")
test_df.to_csv(f"{HERE}/test_pahse2_ligprep_redone_smiles.csv", index=False)

print("Done.")
