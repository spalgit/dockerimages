"""
Fix wrongly assigned amide groups: N=C([O-]) → NC(=O).

The error: oxygen is deprotonated and the C=N / C-O bond order is inverted.
Fix:
  1. Change C=N double bond → C-N single bond
  2. Change C-O(-) single bond → C=O double bond
  3. Set O formal charge from -1 to 0
  4. Add an explicit H (with 3D coords) to the now-secondary-amide N

Input:  *_n_1.sdf files (sulfonamide + [n+] already fixed)
Output: *_n_1_2.sdf files
"""
from rdkit import Chem
from rdkit.Chem import BondType
import os

PATT = Chem.MolFromSmarts("[N;X2,X3]=C([O-])")

FILES = [
    "f1_plus_htchem_ligprepped_ALL_REDONE_n_3.sdf",
    "test_pahse2_ligprep_redone_n_3.sdf",
]


def fix_amide(mol):
    """
    For every N=C([O-]) match:
      - flip bond orders (C=N → C-N, C-O(-) → C=O)
      - neutralise O
      - add explicit H to N with 3D coords
    Returns (new_mol, n_fixed).
    """
    matches = mol.GetSubstructMatches(PATT)
    if not matches:
        return mol, 0

    rw = Chem.RWMol(mol)
    n_indices_to_protonate = []

    for n_idx, c_idx, o_idx in matches:
        # Flip bond types
        rw.GetBondBetweenAtoms(n_idx, c_idx).SetBondType(BondType.SINGLE)
        rw.GetBondBetweenAtoms(c_idx, o_idx).SetBondType(BondType.DOUBLE)
        # Neutralise oxygen
        rw.GetAtomWithIdx(o_idx).SetFormalCharge(0)
        n_indices_to_protonate.append(n_idx)

    Chem.SanitizeMol(rw)

    # Add explicit H (with 3D coords) only to the fixed N atoms
    new_mol = Chem.AddHs(rw.GetMol(), onlyOnAtoms=n_indices_to_protonate, addCoords=True)
    return new_mol, len(matches)


for fname in FILES:
    base, ext = os.path.splitext(fname)          # e.g. "…_n_1", ".sdf"
    out_fname = base + "_2" + ext                # → "…_n_1_2.sdf"

    sup = Chem.SDMolSupplier(fname, removeHs=False)
    writer = Chem.SDWriter(out_fname)

    total = fixed_mols = fixed_groups = errors = 0
    for mol in sup:
        total += 1
        if mol is None:
            errors += 1
            print(f"  WARNING: could not parse molecule #{total} in {fname}")
            continue
        new_mol, n = fix_amide(mol)
        if n:
            fixed_mols += 1
            fixed_groups += n
        writer.write(new_mol)

    writer.close()
    print(
        f"{fname}\n"
        f"  → {out_fname}\n"
        f"  total={total}  molecules fixed={fixed_mols}  "
        f"amide groups corrected={fixed_groups}  errors={errors}\n"
    )
