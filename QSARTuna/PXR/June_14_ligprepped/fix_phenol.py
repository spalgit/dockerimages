"""
Neutralize phenolic oxygen atoms: c[O-] → c(OH).

Fix:
  1. Set O formal charge from -1 to 0
  2. Add explicit H (with 3D coords) to the oxygen

Input:  *_n_1_2.sdf files
Output: *_n_1_2_3.sdf files
"""
from rdkit import Chem
import os

PATT = Chem.MolFromSmarts("[c][O-]")

FILES = [
    "train_set_AND_phase_one_results_4392_ligpreped_f_1_n_1_2.sdf",
    "train_set_4139_ligpreped_f_2_n_1_2.sdf",
    "test_phase2_ligprepped_f_2_n_1_2.sdf",
]


def fix_phenol(mol):
    """Neutralise c[O-] → cOH; add explicit H with 3D coords. Returns (new_mol, n_fixed)."""
    matches = mol.GetSubstructMatches(PATT)
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


for fname in FILES:
    base, ext = os.path.splitext(fname)       # e.g. "…_n_1_2", ".sdf"
    out_fname = base + "_3" + ext             # → "…_n_1_2_3.sdf"

    sup = Chem.SDMolSupplier(fname, removeHs=False)
    writer = Chem.SDWriter(out_fname)

    total = fixed_mols = fixed_groups = errors = 0
    for mol in sup:
        total += 1
        if mol is None:
            errors += 1
            print(f"  WARNING: could not parse molecule #{total} in {fname}")
            continue
        new_mol, n = fix_phenol(mol)
        if n:
            fixed_mols += 1
            fixed_groups += n
        writer.write(new_mol)

    writer.close()
    print(
        f"{fname}\n"
        f"  → {out_fname}\n"
        f"  total={total}  molecules fixed={fixed_mols}  "
        f"phenolic [O-] corrected={fixed_groups}  errors={errors}\n"
    )
