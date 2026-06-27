"""
Pipeline for Chembl_pxr_ligprepped.sdf:
  1. neutralize_sulfonamide_N  (R-SO2-[N-]-R  → R-SO2-NH-R)
  2. fix_phenol                (c[O-]          → cOH)
  3. fix_amide                 (N=C([O-])      → NC(=O))
  4. neutralize_nplus          ([nH+]          → n  for pyridinium-type)
  5. Pick first conformer per ID
  6. Merge with f1_plus_htchem_ligprepped_ALL_REDONE_n_3_2_1.sdf
     → Chembl_pxr_ligprepped_cleaned_merged.sdf
"""

import os
from rdkit import Chem
from rdkit.Chem import BondType

DIR = "/home/spal/dockerimages/QSARTuna/PXR/June_19"
IN_SDF     = os.path.join(DIR, "Chembl_pxr_ligprepped.sdf")
BASE_SDF   = os.path.join(DIR, "f1_plus_htchem_ligprepped_ALL_REDONE_n_3_2_1.sdf")
OUT_SDF    = os.path.join(DIR, "Chembl_pxr_ligprepped_cleaned_merged.sdf")

# ── Fixing functions (identical logic to the existing scripts) ─────────────────

PATT_SULFONAMIDE = None  # detected by atom traversal
PATT_PHENOL      = Chem.MolFromSmarts("[c][O-]")
PATT_AMIDE       = Chem.MolFromSmarts("[N;X2,X3]=C([O-])")


def neutralize_sulfonamide_N(mol):
    rw = Chem.RWMol(mol)
    neutralized_idx = []
    for atom in rw.GetAtoms():
        if atom.GetAtomicNum() != 7 or atom.GetFormalCharge() != -1:
            continue
        for nbr in atom.GetNeighbors():
            if nbr.GetAtomicNum() != 16:
                continue
            oxygens = [n for n in nbr.GetNeighbors() if n.GetAtomicNum() == 8]
            if len(oxygens) >= 2:
                atom.SetFormalCharge(0)
                neutralized_idx.append(atom.GetIdx())
                break
    if not neutralized_idx:
        return mol, 0
    Chem.SanitizeMol(rw)
    new_mol = Chem.AddHs(rw.GetMol(), onlyOnAtoms=neutralized_idx, addCoords=True)
    return new_mol, len(neutralized_idx)


def fix_phenol(mol):
    matches = mol.GetSubstructMatches(PATT_PHENOL)
    if not matches:
        return mol, 0
    rw = Chem.RWMol(mol)
    o_indices = []
    for _c_idx, o_idx in matches:
        rw.GetAtomWithIdx(o_idx).SetFormalCharge(0)
        o_indices.append(o_idx)
    Chem.SanitizeMol(rw)
    new_mol = Chem.AddHs(rw.GetMol(), onlyOnAtoms=o_indices, addCoords=True)
    return new_mol, len(matches)


def fix_amide(mol):
    matches = mol.GetSubstructMatches(PATT_AMIDE)
    if not matches:
        return mol, 0
    rw = Chem.RWMol(mol)
    n_indices = []
    for n_idx, c_idx, o_idx in matches:
        rw.GetBondBetweenAtoms(n_idx, c_idx).SetBondType(BondType.SINGLE)
        rw.GetBondBetweenAtoms(c_idx, o_idx).SetBondType(BondType.DOUBLE)
        rw.GetAtomWithIdx(o_idx).SetFormalCharge(0)
        n_indices.append(n_idx)
    Chem.SanitizeMol(rw)
    new_mol = Chem.AddHs(rw.GetMol(), onlyOnAtoms=n_indices, addCoords=True)
    return new_mol, len(matches)


def neutralize_nplus(mol):
    rw = Chem.RWMol(mol)
    to_fix = []
    skipped_q = 0
    for atom in rw.GetAtoms():
        if atom.GetAtomicNum() != 7 or atom.GetFormalCharge() != 1 or not atom.GetIsAromatic():
            continue
        h_neighbors = [n for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
        if h_neighbors:
            to_fix.append((atom.GetIdx(), h_neighbors[0].GetIdx()))
        else:
            skipped_q += 1
    if not to_fix:
        return mol, 0, skipped_q
    for n_idx, h_idx in sorted(to_fix, key=lambda x: x[1], reverse=True):
        rw.RemoveAtom(h_idx)
    for atom in rw.GetAtoms():
        if atom.GetAtomicNum() == 7 and atom.GetFormalCharge() == 1 and atom.GetIsAromatic():
            if not [n for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]:
                atom.SetFormalCharge(0)
    for bond in rw.GetBonds():
        if bond.GetBondTypeAsDouble() == 1.5:
            bond.SetBondType(Chem.BondType.SINGLE)
    for atom in rw.GetAtoms():
        atom.SetIsAromatic(False)
    Chem.SanitizeMol(rw)
    return rw.GetMol(), len(to_fix), skipped_q


def apply_all_fixes(mol):
    mol, n1 = neutralize_sulfonamide_N(mol)
    mol, n2 = fix_phenol(mol)
    mol, n3 = fix_amide(mol)
    mol, n4, _ = neutralize_nplus(mol)
    return mol, n1, n2, n3, n4


# ── Step 1: Clean and pick first conformer per ID ─────────────────────────────

sup = Chem.SDMolSupplier(IN_SDF, removeHs=False)

seen_ids = set()
cleaned  = []
stats    = {"total": 0, "errors": 0, "kept": 0,
            "sulfonamide": 0, "phenol": 0, "amide": 0, "nplus": 0}

for mol in sup:
    stats["total"] += 1
    if mol is None:
        stats["errors"] += 1
        continue
    name = mol.GetProp("_Name")
    if name in seen_ids:
        continue  # keep only first conformer per ID
    seen_ids.add(name)
    try:
        mol, n1, n2, n3, n4 = apply_all_fixes(mol)
        stats["sulfonamide"] += n1
        stats["phenol"]      += n2
        stats["amide"]       += n3
        stats["nplus"]       += n4
        cleaned.append(mol)
        stats["kept"] += 1
    except Exception as e:
        print(f"  WARNING: fix failed for {name}: {e}")
        cleaned.append(mol)
        stats["kept"] += 1

print(f"Input SDF         : {stats['total']} molecules  ({stats['errors']} unparseable)")
print(f"Unique IDs kept   : {stats['kept']}")
print(f"Fixes applied:")
print(f"  sulfonamide N-  : {stats['sulfonamide']}")
print(f"  phenol [O-]     : {stats['phenol']}")
print(f"  amide N=C[O-]   : {stats['amide']}")
print(f"  aromatic [n+]   : {stats['nplus']}")

# ── Step 2: Merge with base training SDF ──────────────────────────────────────

writer = Chem.SDWriter(OUT_SDF)

# Write base SDF first
base_sup = Chem.SDMolSupplier(BASE_SDF, removeHs=False)
base_count = 0
for mol in base_sup:
    if mol is None:
        continue
    writer.write(mol)
    base_count += 1

# Append cleaned ChEMBL molecules
for mol in cleaned:
    writer.write(mol)

writer.close()

total_out = base_count + len(cleaned)
print(f"\nBase SDF molecules: {base_count}")
print(f"ChEMBL added      : {len(cleaned)}")
print(f"Total in output   : {total_out}")
print(f"\nOutput: {OUT_SDF}")
