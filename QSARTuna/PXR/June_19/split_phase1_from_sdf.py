"""
Extract phase1 test compounds from f1_plus_htchem_train_4827.sdf.

Reads molecule names from test_phase1.csv, writes matching molecules to
test_phase1.sdf, and writes the remainder to f1_plus_htchem_train_minus_phase1.sdf.
"""

from pathlib import Path
import pandas as pd
from rdkit import Chem

HERE     = Path("/home/spal/dockerimages/QSARTuna/PXR/June_19")
PXR_DIR  = Path("/home/spal/dockerimages/QSARTuna/PXR")

SRC_SDF  = HERE / "f1_plus_htchem_train_4827.sdf"
CSV_PATH = PXR_DIR / "test_phase1.csv"

OUT_TEST  = HERE / "test_phase1.sdf"
OUT_TRAIN = HERE / "f1_plus_htchem_train_minus_phase1.sdf"

# Load names from CSV
df = pd.read_csv(CSV_PATH)
phase1_names = set(df["Molecule Name"].str.strip())
print(f"Phase1 test names in CSV : {len(phase1_names)}")

suppl = Chem.SDMolSupplier(str(SRC_SDF), removeHs=False)

w_test  = Chem.SDWriter(str(OUT_TEST))
w_train = Chem.SDWriter(str(OUT_TRAIN))

n_test = n_train = n_skip = 0
found_names = set()

for mol in suppl:
    if mol is None:
        n_skip += 1
        continue
    name = mol.GetProp("_Name").strip() if mol.HasProp("_Name") else ""
    if name in phase1_names:
        w_test.write(mol)
        found_names.add(name)
        n_test += 1
    else:
        w_train.write(mol)
        n_train += 1

w_test.close()
w_train.close()

print(f"Molecules written to test_phase1.sdf              : {n_test}")
print(f"Molecules written to f1_plus_htchem_train_minus_phase1.sdf : {n_train}")
print(f"Molecules skipped (None)                           : {n_skip}")

missing = phase1_names - found_names
if missing:
    print(f"\nWARNING: {len(missing)} phase1 names not found in SDF:")
    for m in sorted(missing):
        print(f"  {m}")
else:
    print("\nAll phase1 names found in the SDF.")
