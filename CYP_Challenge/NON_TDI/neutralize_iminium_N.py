"""
Neutralize a protonated amidine / guanidine / iminium nitrogen:
C=[NH+]- -> C=N- .

LigPrep protonates these sp2 nitrogens; the neutral form is wanted here.  Only
the iminium N itself is touched, so a genuinely basic site elsewhere in the same
molecule (e.g. the N-methylpiperazine of clozapine) keeps its charge.

Fix:
  1. Remove one explicit H from the sp2 N+
  2. Set its formal charge from +1 to 0

A fully substituted iminium (no H, e.g. =[N+](C)C in methylene blue or crystal
violet) is permanently charged and is left unchanged.

Input:  *_n_1_2_3_4.sdf files
Output: *_n_1_2_3_4_5.sdf files
"""
from rdkit import Chem
import os

PATT = Chem.MolFromSmarts("[N+;!a;!H0]=[#6]")

FILES = [
    "train_set_AND_phase_one_results_4392_ligpreped_f_1_n_1_2_3_4.sdf",
    "train_set_4139_ligpreped_f_2_n_1_2_3_4.sdf",
    "test_phase2_ligprepped_f_2_n_1_2_3_4.sdf",
]


def neutralize_iminium_N(mol):
    """Neutralise C=[NH+] -> C=N.  Returns (new_mol, n_neutralized)."""
    matches = mol.GetSubstructMatches(PATT)
    if not matches:
        return mol, 0

    rw = Chem.RWMol(mol)
    h_to_remove = []
    seen = set()

    for n_idx, _c_idx in matches:
        if n_idx in seen:
            continue
        seen.add(n_idx)
        atom = rw.GetAtomWithIdx(n_idx)
        h_nbrs = [nb for nb in atom.GetNeighbors() if nb.GetAtomicNum() == 1]
        if h_nbrs:
            # Tag it: indices shift once the H atoms are removed.
            atom.SetBoolProp("_iminium_to_neutralize", True)
            h_to_remove.append(h_nbrs[0].GetIdx())
        else:
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(max(0, atom.GetNumExplicitHs() - 1))

    for h_idx in sorted(h_to_remove, reverse=True):
        rw.RemoveAtom(h_idx)

    for atom in rw.GetAtoms():
        if atom.HasProp("_iminium_to_neutralize"):
            atom.SetFormalCharge(0)
            atom.ClearProp("_iminium_to_neutralize")

    Chem.SanitizeMol(rw)
    return rw.GetMol(), len(seen)


if __name__ == "__main__":
    for fname in FILES:
        base, ext = os.path.splitext(fname)
        out_fname = base + "_5" + ext

        sup = Chem.SDMolSupplier(fname, removeHs=False)
        writer = Chem.SDWriter(out_fname)

        total = fixed_mols = fixed_groups = errors = 0
        for mol in sup:
            total += 1
            if mol is None:
                errors += 1
                print(f"  WARNING: could not parse molecule #{total} in {fname}")
                continue
            new_mol, n = neutralize_iminium_N(mol)
            if n:
                fixed_mols += 1
                fixed_groups += n
            writer.write(new_mol)

        writer.close()
        print(
            f"{fname}\n"
            f"  → {out_fname}\n"
            f"  total={total}  molecules fixed={fixed_mols}  "
            f"iminium [NH+] neutralized={fixed_groups}  errors={errors}\n"
        )
