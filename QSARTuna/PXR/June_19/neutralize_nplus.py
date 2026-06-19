"""
Neutralize protonated aromatic nitrogen [n+] that carries an explicit H,
i.e. pyridinium-type: remove the H and set formal charge to 0.

Quaternary [n+] (no H, all-carbon neighbors) are permanently charged and
are left unchanged — they cannot be neutralized without breaking a C–N bond.

Reads from *_n.sdf files (H-retained after sulfonamide neutralization).
Writes *_n_1.sdf files.
"""
from rdkit import Chem
import os

FILES = [
    "f1_plus_htchem_ligprepped_ALL_REDONE_n_3_2.sdf",
    "test_pahse2_ligprep_redone_n_3_2.sdf",
]


def neutralize_nplus(mol):
    """
    Find aromatic N+ atoms that have an explicit H neighbor (pyridinium-type).
    Remove the H atom and set formal charge to 0.
    Returns (new_mol, n_neutralized, n_skipped_quaternary).
    """
    rw = Chem.RWMol(mol)

    # Collect (n_idx, h_idx) pairs; process H removal in reverse index order
    # to avoid index-shift bugs after atom removal.
    to_fix = []
    skipped_quaternary = 0

    for atom in rw.GetAtoms():
        if atom.GetAtomicNum() != 7 or atom.GetFormalCharge() != 1 or not atom.GetIsAromatic():
            continue
        h_neighbors = [n for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
        if h_neighbors:
            to_fix.append((atom.GetIdx(), h_neighbors[0].GetIdx()))
        else:
            skipped_quaternary += 1

    if not to_fix:
        return mol, 0, skipped_quaternary

    # Remove H atoms high-index-first to keep lower indices stable
    to_fix_sorted = sorted(to_fix, key=lambda x: x[1], reverse=True)
    for n_idx, h_idx in to_fix_sorted:
        rw.RemoveAtom(h_idx)

    # After removals, N indices may have shifted — re-scan for N+ to fix
    for atom in rw.GetAtoms():
        if atom.GetAtomicNum() == 7 and atom.GetFormalCharge() == 1 and atom.GetIsAromatic():
            h_nbrs = [n for n in atom.GetNeighbors() if n.GetAtomicNum() == 1]
            if not h_nbrs:           # H was removed — safe to neutralize
                atom.SetFormalCharge(0)

    # After structural edits the stored aromatic bond types (1.5) are stale
    # and confuse kekulization.  Clear them so SanitizeMol re-perceives
    # aromaticity from scratch based on the updated atom properties.
    for bond in rw.GetBonds():
        if bond.GetBondTypeAsDouble() == 1.5:
            bond.SetBondType(Chem.BondType.SINGLE)
    for atom in rw.GetAtoms():
        atom.SetIsAromatic(False)

    Chem.SanitizeMol(rw)
    return rw.GetMol(), len(to_fix), skipped_quaternary


for fname in FILES:
    base, ext = os.path.splitext(fname)          # e.g. "…_n", ".sdf"
    out_fname = base + "_1" + ext                # → "…_n_1.sdf"

    sup = Chem.SDMolSupplier(fname, removeHs=False)
    writer = Chem.SDWriter(out_fname)

    total = neutralized = skipped_q = errors = 0
    for mol in sup:
        total += 1
        if mol is None:
            errors += 1
            print(f"  WARNING: could not parse molecule #{total} in {fname}")
            continue
        new_mol, n_neut, n_skip = neutralize_nplus(mol)
        neutralized += n_neut
        skipped_q += n_skip
        writer.write(new_mol)

    writer.close()
    print(
        f"{fname}\n"
        f"  → {out_fname}\n"
        f"  total={total}  [n+] neutralized={neutralized}  "
        f"quaternary [n+] skipped={skipped_q}  errors={errors}\n"
    )
