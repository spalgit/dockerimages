"""
Build training SDF: CI-filtered inactive compounds + 3x SMILES augmentation.

Rules applied to pEC50 <= 3.5 compounds:
    Remove if CI upper bound (95% CI from dose-response fit) > 4.0.
    These are ambiguous inactives whose confidence interval overlaps active space.
    Remaining inactives get 3 additional randomized-SMILES copies.

Active compounds (pEC50 > 3.5): kept exactly as-is from the source SDF.

Input SDF : processed_Openadmet_REAL_PXR_train_AND_test_..._moe_prepped.sdf
Input CSV : train.csv  (CI upper bound column)
Output SDF: train_ci_filtered_aug.sdf
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

SOURCE_SDF = Path(
    "/home/spal/dockerimages/QSARTuna/PXR_For_QSARTuna_Modelling/"
    "processed_Openadmet_REAL_PXR_train_AND_test_main_with_side_info_"
    "AND_counter_screen_weighted_moe_prepped.sdf"
)
TRAIN_CSV  = Path("/home/spal/dockerimages/QSARTuna/PXR/train.csv")
OUTPUT_SDF = Path("/home/spal/dockerimages/QSARTuna/PXR/train_ci_filtered_aug.sdf")

INACTIVE_THRESH = 3.5
CI_UPPER_LIMIT  = 4.0
N_AUGMENTATIONS = 3


def randomize_smiles(mol, seed: int) -> str:
    random.seed(seed)
    order = list(range(mol.GetNumAtoms()))
    random.shuffle(order)
    return Chem.MolToSmiles(Chem.RenumberAtoms(mol, order), canonical=False)


# ── CI upper bounds from train.csv ─────────────────────────────────────────────
df_train = pd.read_csv(TRAIN_CSV)
ci_lookup = df_train.set_index("Molecule Name")["pEC50_ci.upper (-log10(molarity))"].to_dict()
print(f"Train compounds loaded: {len(df_train)}")

# ── Load source SDF ────────────────────────────────────────────────────────────
suppl = Chem.SDMolSupplier(str(SOURCE_SDF), removeHs=True)
train_mols = []
skipped = 0
for mol in suppl:
    if mol is None:
        skipped += 1
        continue
    train_mols.append(mol)

print(f"SDF molecules loaded : {len(train_mols)}  ({skipped} skipped)")

# ── Partition active / inactive ────────────────────────────────────────────────
active_mols   = []
inactive_mols = []
removed       = []

for mol in train_mols:
    name  = mol.GetProp("Name") if mol.HasProp("Name") else mol.GetProp("_Name")
    pec50 = float(mol.GetProp("pEC50"))
    if pec50 > INACTIVE_THRESH:
        active_mols.append(mol)
    else:
        ci_up = ci_lookup.get(name)
        if ci_up is not None and float(ci_up) > CI_UPPER_LIMIT:
            removed.append((name, pec50, float(ci_up)))
        else:
            inactive_mols.append(mol)

print(f"\nActive  (pEC50 > {INACTIVE_THRESH})             : {len(active_mols)}")
print(f"Inactive kept  (CI upper <= {CI_UPPER_LIMIT}) : {len(inactive_mols)}")
print(f"Inactive removed (CI upper > {CI_UPPER_LIMIT}) : {len(removed)}")
for name, pec50, ci_up in removed:
    print(f"  {name}: pEC50={pec50:.3f}, CI upper={ci_up:.3f}")

# ── Write output SDF ───────────────────────────────────────────────────────────
writer = Chem.SDWriter(str(OUTPUT_SDF))
n_aug = 0
n_aug_failed = 0

# Active compounds — straight copy, no augmentation
for mol in active_mols:
    mol.SetProp("is_augmented", "0")
    writer.write(mol)

# Inactive compounds — original + 3 randomized-SMILES copies
for idx, mol in enumerate(inactive_mols):
    mol.SetProp("is_augmented", "0")
    writer.write(mol)

    # Derive canonical SMILES from the mol graph (strip 3D for augmentation base)
    canon_smi = Chem.MolToSmiles(mol)
    canon_mol = Chem.MolFromSmiles(canon_smi)
    if canon_mol is None:
        n_aug_failed += 1
        continue

    seen  = {canon_smi}
    count = 0
    tries = 0
    while count < N_AUGMENTATIONS and tries < N_AUGMENTATIONS * 10:
        rand_smi = randomize_smiles(canon_mol, seed=idx * 100 + tries)
        tries += 1
        if rand_smi in seen:
            continue
        seen.add(rand_smi)

        aug_mol = Chem.MolFromSmiles(rand_smi)
        if aug_mol is None:
            continue
        AllChem.Compute2DCoords(aug_mol)

        # Copy all SD tags from original
        name = mol.GetProp("Name") if mol.HasProp("Name") else mol.GetProp("_Name")
        aug_mol.SetProp("_Name", name)
        for prop, val in mol.GetPropsAsDict().items():
            aug_mol.SetProp(prop, str(val))
        aug_mol.SetProp("is_augmented", "1")
        writer.write(aug_mol)
        count += 1
        n_aug += 1

writer.close()

# ── Summary ────────────────────────────────────────────────────────────────────
n_inactive_total = len(inactive_mols) + n_aug
total = len(active_mols) + n_inactive_total
print(f"\nOutput: {OUTPUT_SDF}")
print(f"  Active rows            : {len(active_mols)}")
print(f"  Inactive original rows : {len(inactive_mols)}")
print(f"  Inactive augmented rows: {n_aug}")
print(f"  Total rows             : {total}")
print(f"  Active/inactive ratio  : {len(active_mols) / n_inactive_total:.2f}x")
if n_aug_failed:
    print(f"  WARNING: {n_aug_failed} inactives skipped (SMILES round-trip failed)")
