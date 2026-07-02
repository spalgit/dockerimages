#!/usr/bin/env python3
"""
PROTAC-aware MCS overlay: independent MCS per terminus, then combined alignment.

Plain whole-molecule MCS (mcs_overlay.py) breaks down on PROTACs because:
  1. rdFMCS.FindMCS only ever returns ONE connected common substructure. A
     PROTAC's two termini (warhead, E3-ligand) are bridged by a linker that
     usually differs between reference and query, so a connected MCS can only
     ever anchor ONE terminus -- the other floats free during embedding.
  2. If the two termini happen to be chemically similar to each other (common
     in PROTAC series -- e.g. both ends carrying aryl-amide motifs), plain
     MCS has no notion of "these are two independent parts of the molecule"
     and can even latch onto the wrong terminus.

Fix: the caller supplies one SMARTS pattern per terminus (warhead, E3-binder).
For each terminus we crop reference and query down to every substructure
match of that SMARTS, run ordinary FindMCS terminus-fragment vs.
terminus-fragment for every (ref_match, query_match) combination, and keep
whichever pairing gives the largest MCS (handles a SMARTS matching more than
once in a molecule, e.g. a symmetric ring system). The two per-terminus MCS
cores -- both still carrying the reference's original 3D coordinates -- are
then concatenated into a single disconnected "core" mol and handed to the
same constrained-embedding / rigid-alignment / shape-scoring machinery as
mcs_overlay.py (RDKit substructure matching and AlignMol both handle
disconnected pattern mols natively: each fragment just has to be found
somewhere in the query, with no forced relationship between fragments other
than what both PROTAC molecules' own connectivity gives them for free during
embedding).

If a terminus SMARTS doesn't match, or its MCS coverage falls below
--min_mcs_ratio, that terminus's constraint is dropped gracefully rather than
aborting: the run falls back to single-terminus-only or, if neither terminus
usably matches, plain whole-molecule MCS / Crippen O3A (same fallback chain
as mcs_overlay.py).

Usage (run inside the espsim conda environment):
  conda run -n espsim python mcs_overlay_protac.py reference.sdf query.sdf output.sdf \
      --warhead_smarts "..." --e3_smarts "..." [options]

Options (in addition to everything in mcs_overlay.py):
  --warhead_smarts STR   SMARTS for the warhead terminus (optional; omit to
                          anchor only the E3-ligand terminus and leave the
                          warhead + linker fully free to re-embed -- e.g. for
                          "overlay the E3 ligand, let the linker/warhead
                          optimize" runs)
  --e3_smarts STR        SMARTS for the E3-ligand terminus (optional; omit to
                          anchor only the warhead terminus). At least one of
                          --warhead_smarts / --e3_smarts is required.
  --min_terminus_atoms INT
                          Minimum MCS atoms within a terminus to accept it as
                          a constraint (default 3; smaller anchors barely
                          constrain 3D geometry and are dropped instead)
"""

import argparse
import copy
import os
import sys

from rdkit import Chem
from rdkit.Chem import AllChem, rdFMCS, rdMolDescriptors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mcs_overlay import (
    find_mcs_core,
    embed_query,
    align_existing_3d,
    score_conformers,
)

sys.path.insert(0, '/home/spal/espsim')
from espsim import GetEspSim


# ---------------------------------------------------------------------------
# Fragment extraction
# ---------------------------------------------------------------------------

def extract_fragment(mol, atom_indices):
    """
    Return a new mol containing only atom_indices (and the bonds between
    them), preserving 3D coordinates for whichever atoms are kept.
    """
    keep = set(atom_indices)
    rw = Chem.RWMol(mol)
    for idx in sorted((i for i in range(mol.GetNumAtoms()) if i not in keep), reverse=True):
        rw.RemoveAtom(idx)
    frag = rw.GetMol()
    try:
        Chem.SanitizeMol(frag, catchErrors=True)
    except Exception:
        pass
    try:
        frag.UpdatePropertyCache(strict=False)
    except Exception:
        pass
    return frag


# ---------------------------------------------------------------------------
# Per-terminus MCS
# ---------------------------------------------------------------------------

def best_terminus_core(ref_noh, query_noh, smarts_patt, complete_rings, timeout,
                        min_terminus_atoms, label):
    """
    Try every (ref_match, query_match) combination of the terminus SMARTS
    against ref_noh / query_noh, run FindMCS fragment-vs-fragment for each,
    and keep the pairing with the largest MCS. Returns
    (core, mcs_natoms, mcs_ratio) or (None, 0, 0.0) if the SMARTS doesn't
    match both molecules or no pairing clears min_terminus_atoms.
    """
    ref_matches = ref_noh.GetSubstructMatches(smarts_patt, uniquify=True)
    query_matches = query_noh.GetSubstructMatches(smarts_patt, uniquify=True)

    if not ref_matches:
        print(f"    [{label}] SMARTS did not match the reference. Skipping this terminus.")
        return None, 0, 0.0
    if not query_matches:
        print(f"    [{label}] SMARTS did not match the query. Skipping this terminus.")
        return None, 0, 0.0

    best_core, best_n, best_ratio = None, 0, 0.0
    for rm in ref_matches:
        ref_frag = extract_fragment(ref_noh, rm)
        for qm in query_matches:
            query_frag = extract_fragment(query_noh, qm)
            core, n, ratio = find_mcs_core(ref_frag, query_frag, complete_rings, timeout)
            if core is not None and n > best_n:
                best_core, best_n, best_ratio = core, n, ratio

    if len(ref_matches) > 1 or len(query_matches) > 1:
        print(f"    [{label}] {len(ref_matches)} ref match(es) x {len(query_matches)} "
              f"query match(es) tried; best MCS = {best_n} atoms.")

    if best_core is None or best_n < min_terminus_atoms:
        print(f"    [{label}] best terminus MCS ({best_n} atoms) below "
              f"min_terminus_atoms={min_terminus_atoms}. Dropping this terminus.")
        return None, 0, 0.0

    return best_core, best_n, best_ratio


def build_combined_core(ref_noh, query_noh, warhead_patt, e3_patt,
                         complete_rings, timeout, min_mcs_ratio, min_terminus_atoms):
    """
    Build the dual-terminus core. Returns
    (core, method_label, info_dict) where core may be None (both termini
    failed -> caller should fall back to whole-molecule MCS / Crippen O3A).
    info_dict carries per-terminus atom counts / ratios for SDF properties.
    """
    core_w, n_w, ratio_w = None, 0, 0.0
    if warhead_patt is not None:
        core_w, n_w, ratio_w = best_terminus_core(
            ref_noh, query_noh, warhead_patt, complete_rings, timeout,
            min_terminus_atoms, "warhead"
        )
        if core_w is not None and ratio_w < min_mcs_ratio:
            print(f"    [warhead] MCS coverage {ratio_w:.2f} < min_mcs_ratio "
                  f"{min_mcs_ratio:.2f}. Dropping this terminus.")
            core_w, n_w, ratio_w = None, 0, 0.0

    core_e, n_e, ratio_e = None, 0, 0.0
    if e3_patt is not None:
        core_e, n_e, ratio_e = best_terminus_core(
            ref_noh, query_noh, e3_patt, complete_rings, timeout,
            min_terminus_atoms, "E3-ligand"
        )
        if core_e is not None and ratio_e < min_mcs_ratio:
            print(f"    [E3-ligand] MCS coverage {ratio_e:.2f} < min_mcs_ratio "
                  f"{min_mcs_ratio:.2f}. Dropping this terminus.")
            core_e, n_e, ratio_e = None, 0, 0.0

    info = {
        "Warhead_MCS_atoms": n_w, "Warhead_MCS_ratio": ratio_w,
        "E3_MCS_atoms": n_e, "E3_MCS_ratio": ratio_e,
    }

    if core_w is not None and core_e is not None:
        combined = Chem.CombineMols(core_w, core_e)
        print(f"    Dual-terminus core: warhead {n_w} atoms + E3-ligand {n_e} atoms "
              f"= {combined.GetNumAtoms()} anchor atoms.")
        return combined, "PROTAC_dual_terminus_MCS", info
    if core_w is not None:
        print("    Only the warhead terminus matched usably; E3-ligand terminus unconstrained.")
        return core_w, "PROTAC_warhead_only_MCS", info
    if core_e is not None:
        print("    Only the E3-ligand terminus matched usably; warhead terminus unconstrained.")
        return core_e, "PROTAC_E3only_MCS", info

    print("    Neither terminus produced a usable MCS. Falling back to whole-molecule MCS.")
    return None, None, info


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PROTAC-aware MCS overlay: independent per-terminus MCS, "
                     "then combined constrained embedding / rigid alignment."
    )
    parser.add_argument('reference', help='Reference 3D SDF (single molecule)')
    parser.add_argument('query', help='Query SDF (one or more molecules)')
    parser.add_argument('output', help='Output SDF with best-aligned poses')
    parser.add_argument('--warhead_smarts', default=None,
                        help='SMARTS pattern identifying the warhead terminus '
                             '(optional; omit to leave the warhead/linker fully '
                             'free during re-embedding, anchoring only the '
                             'E3-ligand terminus)')
    parser.add_argument('--e3_smarts', default=None,
                        help='SMARTS pattern identifying the E3-ligand terminus '
                             '(optional; omit to anchor only the warhead terminus)')
    parser.add_argument('--num_confs', type=int, default=50,
                        help='Conformers per query molecule (default 50; '
                             'ignored with --use_existing_3d)')
    parser.add_argument('--min_mcs_ratio', type=float, default=0.1,
                        help='Min MCS / terminus-fragment heavy-atom coverage '
                             '(default 0.1); below this, that terminus is dropped')
    parser.add_argument('--min_terminus_atoms', type=int, default=3,
                        help='Min MCS atoms within a terminus to keep it as a '
                             'constraint (default 3)')
    parser.add_argument('--no_complete_rings', action='store_true',
                        help='Allow MCS to match partial rings')
    parser.add_argument('--use_existing_3d', action='store_true',
                        help='Align and score existing 3D poses without re-embedding')
    parser.add_argument('--esp', action='store_true',
                        help='Compute ESP similarity for the best pose')
    parser.add_argument('--partial_charges', default='gasteiger',
                        choices=['gasteiger', 'mmff'],
                        help='Charge method for ESP (default gasteiger)')
    parser.add_argument('--all_confs', action='store_true',
                        help='Write all conformers ranked by shape '
                             '(not just best; ignored with --use_existing_3d)')
    args = parser.parse_args()

    if args.warhead_smarts is None and args.e3_smarts is None:
        sys.exit("Provide at least one of --warhead_smarts / --e3_smarts.")

    warhead_patt = None
    if args.warhead_smarts is not None:
        warhead_patt = Chem.MolFromSmarts(args.warhead_smarts)
        if warhead_patt is None:
            sys.exit(f"Invalid --warhead_smarts: {args.warhead_smarts}")

    e3_patt = None
    if args.e3_smarts is not None:
        e3_patt = Chem.MolFromSmarts(args.e3_smarts)
        if e3_patt is None:
            sys.exit(f"Invalid --e3_smarts: {args.e3_smarts}")

    ref_mol = Chem.MolFromMolFile(args.reference, removeHs=True)
    if ref_mol is None:
        sys.exit(f"Cannot read reference: {args.reference}")
    if ref_mol.GetNumConformers() == 0:
        sys.exit("Reference molecule has no 3D coordinates.")
    ref_noh = ref_mol

    if warhead_patt is not None and not ref_noh.HasSubstructMatch(warhead_patt):
        print("WARNING: warhead SMARTS does not match the reference at all.")
    if e3_patt is not None and not ref_noh.HasSubstructMatch(e3_patt):
        print("WARNING: E3-ligand SMARTS does not match the reference at all.")

    suppl = Chem.SDMolSupplier(args.query, removeHs=True)
    queries = []
    for i, m in enumerate(suppl):
        if m is not None:
            name = m.GetProp('_Name') if m.HasProp('_Name') else f'mol_{i}'
            queries.append((m, name))
    if not queries:
        sys.exit(f"No valid molecules in {args.query}")

    complete_rings = not args.no_complete_rings
    mode = "existing 3D (rigid MCS align)" if args.use_existing_3d else f"{args.num_confs} conformers"
    print(f"Reference : {ref_noh.GetNumAtoms()} heavy atoms")
    print(f"Queries   : {len(queries)} molecules")
    print(f"Mode      : {mode}")

    writer = Chem.SDWriter(args.output)

    for i, (qmol, name) in enumerate(queries):
        print(f"\n[{i+1}/{len(queries)}] {name}  ({qmol.GetNumAtoms()} heavy atoms)")

        core, method_hint, info = build_combined_core(
            ref_noh, qmol, warhead_patt, e3_patt,
            complete_rings, timeout=10,
            min_mcs_ratio=args.min_mcs_ratio,
            min_terminus_atoms=args.min_terminus_atoms,
        )

        core_smi = Chem.MolToSmiles(core) if core is not None else "N/A"
        mcs_natoms = core.GetNumAtoms() if core is not None else 0

        if core is None:
            # whole-molecule fallback, same behaviour as mcs_overlay.py
            core, mcs_natoms, mcs_ratio = find_mcs_core(ref_noh, qmol, complete_rings)
            core_smi = Chem.MolToSmiles(core) if core is not None else "N/A"
            method_hint = "wholemol_MCS_fallback" if core is not None else "CrippenO3A_fallback"

        if args.use_existing_3d:
            if qmol.GetNumConformers() == 0:
                print("  No 3D conformer in query molecule. Skipping.")
                continue
            qmol_work = copy.deepcopy(qmol)
            try:
                method, mcs_rmsd = align_existing_3d(qmol_work, ref_noh, core)
            except ValueError as e:
                print(f"  Alignment failed: {e}. Skipping.")
                continue
            rmsd_str = f"{mcs_rmsd:.3f} A" if mcs_rmsd is not None else "n/a"
            print(f"  Method: {method_hint} / {method}  |  MCS RMSD: {rmsd_str}")
            scores = score_conformers(qmol_work, ref_noh)
            if not scores:
                print("  Shape scoring failed. Skipping.")
                continue
            best_shape, best_cid = scores[0]
            print(f"  ShapeTanimoto : {best_shape:.4f}")

            best_esp = None
            if args.esp:
                try:
                    best_esp = GetEspSim(
                        qmol_work, ref_noh, prbCid=best_cid, refCid=0,
                        partialCharges=args.partial_charges, renormalize=True, nocheck=True,
                    )
                    print(f"  ESP ({args.partial_charges}) : {best_esp:.4f}")
                except Exception as e:
                    print(f"  ESP failed: {e}")

            out_noh = Chem.RemoveHs(qmol_work)
            out_noh.SetProp('_Name', name)
            out_noh.SetProp('ShapeTanimoto', f'{best_shape:.4f}')
            out_noh.SetProp('AlignmentMethod', f'{method_hint}/{method}')
            out_noh.SetProp('MCS_NumAtoms', str(mcs_natoms))
            out_noh.SetProp('MCS_SMILES', core_smi)
            for k, v in info.items():
                out_noh.SetProp(k, str(v))
            if mcs_rmsd is not None:
                out_noh.SetProp('MCS_RMSD', f'{mcs_rmsd:.4f}')
            if best_esp is not None:
                out_noh.SetProp(f'ESPSim_{args.partial_charges}', f'{best_esp:.4f}')
            writer.write(out_noh, confId=best_cid)

        else:
            try:
                qmol_confs, method = embed_query(qmol, core, ref_noh, args.num_confs)
            except ValueError as e:
                print(f"  Embedding failed: {e}. Skipping.")
                continue

            scores = score_conformers(qmol_confs, ref_noh)
            if not scores:
                print("  Shape scoring failed. Skipping.")
                continue

            best_shape, best_cid = scores[0]
            print(f"  Best ShapeTanimoto : {best_shape:.4f}  (conf {best_cid} / {len(scores)})")

            best_esp = None
            if args.esp:
                try:
                    best_esp = GetEspSim(
                        qmol_confs, ref_noh, prbCid=best_cid, refCid=0,
                        partialCharges=args.partial_charges, renormalize=True, nocheck=True,
                    )
                    print(f"  ESP ({args.partial_charges}) : {best_esp:.4f}")
                except Exception as e:
                    print(f"  ESP failed: {e}")

            out_noh = Chem.RemoveHs(qmol_confs)
            out_noh.SetProp('_Name', name)
            out_noh.SetProp('ShapeTanimoto', f'{best_shape:.4f}')
            out_noh.SetProp('AlignmentMethod', f'{method_hint}/{method}')
            out_noh.SetProp('MCS_NumAtoms', str(mcs_natoms))
            out_noh.SetProp('MCS_SMILES', core_smi)
            for k, v in info.items():
                out_noh.SetProp(k, str(v))
            if best_esp is not None:
                out_noh.SetProp(f'ESPSim_{args.partial_charges}', f'{best_esp:.4f}')

            if args.all_confs:
                for rank, (shape, cid) in enumerate(scores):
                    m = copy.deepcopy(out_noh)
                    m.SetProp('ShapeTanimoto', f'{shape:.4f}')
                    m.SetProp('ConformerRank', str(rank + 1))
                    writer.write(m, confId=cid)
            else:
                writer.write(out_noh, confId=best_cid)

    writer.close()
    print(f"\nDone. Output -> {args.output}")


if __name__ == '__main__':
    main()
