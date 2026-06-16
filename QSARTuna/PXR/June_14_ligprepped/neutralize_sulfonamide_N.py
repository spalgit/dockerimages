"""
Neutralize negatively charged nitrogen atoms in sulfonamide groups
(R-SO2-N(-)-R' → R-SO2-NH-R') across specified SDF files.
"""
from rdkit import Chem
import os

FILES = [
    "train_set_AND_phase_one_results_4392_ligpreped_f_1.sdf",
    "train_set_4139_ligpreped_f_2.sdf",
    "test_phase2_ligprepped_f_2.sdf",
]


def neutralize_sulfonamide_N(mol):
    """Find N(-) bonded to S(=O)(=O), set charge to 0, add explicit H with 3D coords."""
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
        return mol, False
    Chem.SanitizeMol(rw)
    # Add explicit H with estimated 3D coordinates only to the neutralized N atoms
    new_mol = Chem.AddHs(rw.GetMol(), onlyOnAtoms=neutralized_idx, addCoords=True)
    return new_mol, True


for fname in FILES:
    base, ext = os.path.splitext(fname)
    out_fname = base + "_n" + ext

    sup = Chem.SDMolSupplier(fname, removeHs=False)
    writer = Chem.SDWriter(out_fname)

    total = neutralized = errors = 0
    for mol in sup:
        total += 1
        if mol is None:
            errors += 1
            print(f"  WARNING: could not parse molecule #{total} in {fname}")
            continue
        new_mol, changed = neutralize_sulfonamide_N(mol)
        if changed:
            neutralized += 1
        writer.write(new_mol)

    writer.close()
    print(
        f"{fname}\n"
        f"  → {out_fname}\n"
        f"  total={total}  neutralized={neutralized}  errors={errors}\n"
    )
