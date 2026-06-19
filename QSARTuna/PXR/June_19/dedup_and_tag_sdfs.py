"""
For each SDF keep only the first conformer per molecule name, strip all
existing Schrodinger/Qikprop properties, then attach the relevant columns
from the corresponding CSV.

Outputs (in ~/dockerimages/QSARTuna/PXR/June_19/):
  f1_plus_htchem_train_4827.sdf   — 4827 training compounds
  test_phase2_260.sdf             — 260 test compounds

Note: test_phase2.csv (June_14_ligprepped) is empty because the script
that generates it skips molecules without pEC50.  final_test_file.csv is
used instead — it contains Molecule Name + SMILES for all 260 test compounds.
"""

import pandas as pd
from rdkit import Chem
from pathlib import Path

JUNE19 = Path("/home/spal/dockerimages/QSARTuna/PXR/June_19")
JUNE14 = Path("/home/spal/dockerimages/QSARTuna/PXR/June_14_ligprepped")
PXR    = Path("/home/spal/dockerimages/QSARTuna/PXR")

TASKS = [
    dict(
        sdf_in  = JUNE19 / "f1_plus_htchem_ligprepped_ALL_REDONE_n_3_2_1.sdf",
        csv     = JUNE14 / "f1_plus_htchem_train_4827.csv",
        name_col= "Name",          # CSV column that matches SDF _Name
        sdf_out = JUNE19 / "f1_plus_htchem_train_4827.sdf",
    ),
    dict(
        sdf_in  = JUNE19 / "test_pahse2_ligprep_redone_n_3_2_1.sdf",
        csv     = PXR    / "final_test_file.csv",
        name_col= "Molecule Name",
        sdf_out = JUNE19 / "test_phase2_260.sdf",
    ),
]


def process(sdf_in, csv, name_col, sdf_out):
    print(f"\n{'='*60}")
    print(f"SDF  : {sdf_in.name}")
    print(f"CSV  : {csv.name}")

    df = pd.read_csv(csv)
    # Build name → row dict (first occurrence wins if CSV has duplicates)
    lookup = {str(row[name_col]): row for _, row in df.iloc[::-1].iterrows()}
    csv_cols = [c for c in df.columns if c != name_col]
    print(f"CSV columns to attach: {csv_cols}")
    print(f"CSV rows: {len(df)}  |  unique names: {len(lookup)}")

    suppl = Chem.SDMolSupplier(str(sdf_in), removeHs=False)

    writer    = Chem.SDWriter(str(sdf_out))
    seen      = set()
    written   = skipped_dup = no_csv = errors = 0

    for mol in suppl:
        if mol is None:
            errors += 1
            continue

        mol_name = mol.GetProp("_Name") if mol.HasProp("_Name") else ""

        # Keep only first conformer per name
        if mol_name in seen:
            skipped_dup += 1
            continue
        seen.add(mol_name)

        # Strip all existing properties
        for prop in list(mol.GetPropsAsDict().keys()):
            mol.ClearProp(prop)

        # Attach CSV properties
        if mol_name not in lookup:
            no_csv += 1
            writer.write(mol)
            written += 1
            continue

        row = lookup[mol_name]
        for col in csv_cols:
            val = row[col]
            if pd.isna(val):
                continue
            if isinstance(val, float):
                mol.SetDoubleProp(col, val)
            elif isinstance(val, int):
                mol.SetIntProp(col, val)
            else:
                mol.SetProp(col, str(val))

        writer.write(mol)
        written += 1

    writer.close()

    print(f"Written : {written}  |  dupes dropped: {skipped_dup}  "
          f"|  no CSV match: {no_csv}  |  parse errors: {errors}")
    print(f"Output  : {sdf_out.name}")


for task in TASKS:
    process(**task)

print("\nDone.")
