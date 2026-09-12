"""
Neutralize protonated morpholine nitrogen: [NH+] in a morpholine ring -> N.

A morpholine N is a weak base (morpholine pKa ~6.5, far lower when N-aryl or
flanked by electron-withdrawing groups), so the protonated form LigPrep emits is
not the dominant species at pH 7.4.

Fix:
  1. Remove the explicit H on the ring N
  2. Set the N formal charge from +1 to 0

Quaternary morpholinium (no H, four heavy neighbours) is permanently charged and
is left unchanged - it cannot be neutralized without breaking a C-N bond.

Input:  *_n_1_2_3.sdf files
Output: *_n_1_2_3_4.sdf files
"""
from rdkit import Chem
import os

# 6-membered saturated ring, N and O in a 1,4 relationship, N carrying >=1 H
PATT = Chem.MolFromSmarts("[N+;X4;!H0;R]1[C;X4][C;X4][O;X2;R][C;X4][C;X4]1")

FILES = [
    "train_set_AND_phase_one_results_4392_ligpreped_f_1_n_1_2_3.sdf",
    "train_set_4139_ligpreped_f_2_n_1_2_3.sdf",
    "test_phase2_ligprepped_f_2_n_1_2_3.sdf",
]


def neutralize_morpholine_N(mol):
    """Neutralise morpholine [NH+] -> N.  Returns (new_mol, n_neutralized)."""
    matches = mol.GetSubstructMatches(PATT)
    if not matches:
        return mol, 0

    rw = Chem.RWMol(mol)
    h_to_remove = []
    seen = set()

    for match in matches:
        n_idx = match[0]
        if n_idx in seen:
            continue
        seen.add(n_idx)
        atom = rw.GetAtomWithIdx(n_idx)
        h_nbrs = [nb for nb in atom.GetNeighbors() if nb.GetAtomicNum() == 1]
        if h_nbrs:
            # Tag it: indices shift once the H atoms are removed, so the
            # re-scan below must identify these N by property, not by index.
            atom.SetBoolProp("_morpholine_to_neutralize", True)
            h_to_remove.append(h_nbrs[0].GetIdx())
        else:
            # H is implicit - just drop the charge and one implicit H
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(max(0, atom.GetNumExplicitHs() - 1))

    # Remove H atoms high-index-first to keep lower indices stable
    for h_idx in sorted(h_to_remove, reverse=True):
        rw.RemoveAtom(h_idx)

    for atom in rw.GetAtoms():
        if atom.HasProp("_morpholine_to_neutralize"):
            atom.SetFormalCharge(0)
            atom.ClearProp("_morpholine_to_neutralize")

    Chem.SanitizeMol(rw)
    return rw.GetMol(), len(seen)


if __name__ == "__main__":
    for fname in FILES:
        base, ext = os.path.splitext(fname)       # e.g. "…_n_1_2_3", ".sdf"
        out_fname = base + "_4" + ext             # → "…_n_1_2_3_4.sdf"

        sup = Chem.SDMolSupplier(fname, removeHs=False)
        writer = Chem.SDWriter(out_fname)

        total = fixed_mols = fixed_groups = errors = 0
        for mol in sup:
            total += 1
            if mol is None:
                errors += 1
                print(f"  WARNING: could not parse molecule #{total} in {fname}")
                continue
            new_mol, n = neutralize_morpholine_N(mol)
            if n:
                fixed_mols += 1
                fixed_groups += n
            writer.write(new_mol)

        writer.close()
        print(
            f"{fname}\n"
            f"  → {out_fname}\n"
            f"  total={total}  molecules fixed={fixed_mols}  "
            f"morpholine [NH+] neutralized={fixed_groups}  errors={errors}\n"
        )
