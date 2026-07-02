#!/usr/bin/env python3
"""
MCS-constrained multi-conformer overlay with shape and optional ESP scoring.

Default workflow (re-embedding):
  1. Find MCS between reference and each query (heavy atoms).
  2. Extract the MCS core with 3D coordinates from the reference.
  3. Generate N ETKDG conformers of the query pinned to the core (tethered UFF
     minimisation rotates non-core bonds for best shape fit).
  4. Score every conformer by ShapeTanimoto vs. the reference; write the best pose.
  5. Optionally compute ESP similarity for the best pose.

--use_existing_3d workflow (fast; for pre-docked / pre-embedded inputs):
  1. Find MCS between reference and each query.
  2. Rigidly align the existing query conformer onto the reference via the
     MCS atom mapping (AlignMol).  Falls back to Crippen O3A if MCS is absent.
  3. Score ShapeTanimoto (and optionally ESP) and write.

Falls back to Crippen O3A alignment if no MCS match is found or constrained
embedding fails.

Usage (run inside the espsim conda environment):
  conda run -n espsim python mcs_overlay.py reference.sdf query.sdf output.sdf [options]

Options:
  --num_confs INT        Conformers per query (default 50; ignored with --use_existing_3d)
  --min_mcs_ratio FLOAT  Min MCS / reference heavy-atom fraction (default 0.1)
  --no_complete_rings    Allow MCS to match partial rings
  --use_existing_3d      Align and score the existing 3D pose without re-embedding
  --esp                  Compute ESP similarity for the best pose
  --partial_charges STR  gasteiger | mmff  (default gasteiger)
  --all_confs            Write all conformers ranked by shape (ignored with --use_existing_3d)
"""

import argparse
import copy
import sys

from rdkit import Chem
from rdkit.Chem import AllChem, rdFMCS, rdMolAlign, rdMolDescriptors

sys.path.insert(0, '/home/spal/espsim')
from espsim import ConstrainedEmbedMultipleConfs, GetShapeSim, GetEspSim


# ---------------------------------------------------------------------------
# MCS helpers
# ---------------------------------------------------------------------------

def find_mcs_core(ref_noh, query_noh, complete_rings=True, timeout=10):
    """
    Find MCS and return (core_mol, mcs_natoms, mcs_ratio_vs_ref).

    core_mol carries the 3D coordinates of the matching atoms in ref_noh
    (via ReplaceSidechains / DeleteSubstructs).  Returns (None, 0, 0.0)
    when no valid MCS is found.
    """
    mcs = rdFMCS.FindMCS(
        [ref_noh, query_noh],
        threshold=0.0,
        completeRingsOnly=complete_rings,
        timeout=timeout,
        bondCompare=rdFMCS.BondCompare.CompareOrderExact,
        atomCompare=rdFMCS.AtomCompare.CompareElements,
    )

    if not mcs.smartsString or mcs.numAtoms == 0:
        return None, 0, 0.0

    patt = Chem.MolFromSmarts(mcs.smartsString)

    ref_trimmed = AllChem.ReplaceSidechains(ref_noh, patt)
    if ref_trimmed is None:
        return None, 0, 0.0

    core = AllChem.DeleteSubstructs(ref_trimmed, Chem.MolFromSmiles('*'))
    if core is None or core.GetNumAtoms() == 0:
        return None, 0, 0.0

    try:
        core.UpdatePropertyCache()
    except Exception:
        pass

    mcs_ratio = mcs.numAtoms / ref_noh.GetNumAtoms()
    return core, mcs.numAtoms, mcs_ratio


# ---------------------------------------------------------------------------
# Embedding / alignment
# ---------------------------------------------------------------------------

def constrained_embed(query_noh, core, num_confs, seed=42):
    """
    Add Hs, generate num_confs ETKDG conformers pinned to core,
    then tethered-UFF-minimise non-core atoms.
    Returns mol_with_confs (Hs included).
    """
    query_h = Chem.AddHs(copy.deepcopy(query_noh))
    query_h = ConstrainedEmbedMultipleConfs(
        query_h, core, numConfs=num_confs, useTethers=True, randomSeed=seed
    )
    return query_h


def crippen_o3a_embed(query_noh, ref_noh, num_confs, seed=42):
    """
    Free ETKDG + UFF minimisation + Crippen O3A alignment fallback.
    Returns mol_with_confs (heavy atoms only, ref_noh conformer space).
    """
    query_h = Chem.AddHs(copy.deepcopy(query_noh))
    cids = list(AllChem.EmbedMultipleConfs(
        query_h, numConfs=num_confs, randomSeed=seed, pruneRmsThresh=0.5
    ))
    if not cids:
        raise ValueError("ETKDG embedding failed.")

    for cid in cids:
        AllChem.UFFOptimizeMolecule(query_h, confId=cid)

    qcrippen = rdMolDescriptors._CalcCrippenContribs(query_h)
    rcrippen = rdMolDescriptors._CalcCrippenContribs(ref_noh)
    for cid in cids:
        try:
            aln = rdMolAlign.GetCrippenO3A(query_h, ref_noh, qcrippen, rcrippen, cid, 0)
            aln.Align()
        except Exception:
            pass

    return query_h


def embed_query(query_noh, core, ref_noh, num_confs, seed=42):
    """
    Try constrained embed; fall back to Crippen O3A.
    Returns (mol_with_confs, method_label).
    mol_with_confs may contain explicit Hs.
    """
    if core is not None and (
        query_noh.HasSubstructMatch(core)
        or Chem.AddHs(query_noh).HasSubstructMatch(core)
    ):
        try:
            mol = constrained_embed(query_noh, core, num_confs, seed)
            print(f"    {mol.GetNumConformers()} MCS-constrained conformers generated.")
            return mol, "MCS_constrained"
        except ValueError as e:
            print(f"    Constrained embed failed ({e}); using Crippen O3A.")

    mol = crippen_o3a_embed(query_noh, ref_noh, num_confs, seed)
    print(f"    {mol.GetNumConformers()} free conformers + Crippen O3A alignment.")
    return mol, "CrippenO3A"


# ---------------------------------------------------------------------------
# Existing-3D alignment (no re-embedding)
# ---------------------------------------------------------------------------

def align_existing_3d(query_noh, ref_noh, core):
    """
    Rigidly align query_noh's existing conformer onto ref_noh using MCS atom
    mapping.  Returns (method_label, mcs_rmsd).

    If core is None or substructure matching fails, falls back to Crippen O3A.
    query_noh is modified in place (conformer coordinates updated).
    """
    if core is not None:
        ref_match = ref_noh.GetSubstructMatch(core)
        query_match = query_noh.GetSubstructMatch(core)
        if ref_match and query_match:
            atom_map = list(zip(query_match, ref_match))
            rmsd = AllChem.AlignMol(query_noh, ref_noh, atomMap=atom_map)
            return "MCS_rigid", rmsd

    # Fallback: Crippen O3A on the single existing conformer
    try:
        qcrippen = rdMolDescriptors._CalcCrippenContribs(query_noh)
        rcrippen = rdMolDescriptors._CalcCrippenContribs(ref_noh)
        aln = rdMolAlign.GetCrippenO3A(query_noh, ref_noh, qcrippen, rcrippen, 0, 0)
        aln.Align()
        return "CrippenO3A", None
    except Exception as e:
        raise ValueError(f"Crippen O3A fallback failed: {e}")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_conformers(query_mol, ref_noh):
    """
    Return list of (shape_tanimoto, conf_id) sorted best-first.
    Both query_mol and ref_noh should share the same coordinate space
    (i.e. query already aligned to ref).
    """
    scores = []
    for cid in range(query_mol.GetNumConformers()):
        try:
            # GetShapeSim works with mixed H/no-H mols
            shape = GetShapeSim(query_mol, ref_noh, prbCid=cid, refCid=0)
            scores.append((shape, cid))
        except Exception:
            pass
    scores.sort(reverse=True)
    return scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MCS-constrained multi-conformer overlay (shape + optional ESP)."
    )
    parser.add_argument('reference', help='Reference 3D SDF (single molecule)')
    parser.add_argument('query', help='Query SDF (one or more molecules)')
    parser.add_argument('output', help='Output SDF with best-aligned poses')
    parser.add_argument('--num_confs', type=int, default=50,
                        help='Conformers per query molecule (default 50; '
                             'ignored with --use_existing_3d)')
    parser.add_argument('--min_mcs_ratio', type=float, default=0.1,
                        help='Min MCS heavy-atom coverage of reference (default 0.1)')
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

    # --- Load reference ---
    ref_mol = Chem.MolFromMolFile(args.reference, removeHs=True)
    if ref_mol is None:
        sys.exit(f"Cannot read reference: {args.reference}")
    if ref_mol.GetNumConformers() == 0:
        sys.exit("Reference molecule has no 3D coordinates.")
    ref_noh = ref_mol  # already stripped of Hs, keeps 3D

    # --- Load queries ---
    suppl = Chem.SDMolSupplier(args.query, removeHs=True)
    queries = []
    for i, m in enumerate(suppl):
        if m is not None:
            name = m.GetProp('_Name') if m.HasProp('_Name') else f'mol_{i}'
            queries.append((m, name))

    if not queries:
        sys.exit(f"No valid molecules in {args.query}")

    mode = "existing 3D (rigid MCS align)" if args.use_existing_3d else f"{args.num_confs} conformers"
    print(f"Reference : {ref_noh.GetNumAtoms()} heavy atoms")
    print(f"Queries   : {len(queries)} molecules")
    print(f"Mode      : {mode}")

    writer = Chem.SDWriter(args.output)

    for i, (qmol, name) in enumerate(queries):
        print(f"\n[{i+1}/{len(queries)}] {name}  ({qmol.GetNumAtoms()} heavy atoms)")

        # MCS
        core, mcs_natoms, mcs_ratio = find_mcs_core(
            ref_noh, qmol,
            complete_rings=not args.no_complete_rings,
        )

        if core is None:
            print("  No MCS found.")
            core_smi = "N/A"
            mcs_natoms = 0
            mcs_ratio = 0.0
        else:
            core_smi = Chem.MolToSmiles(core)
            print(f"  MCS : {core_smi}  |  {mcs_natoms} atoms  |  "
                  f"{mcs_ratio*100:.1f}% of ref")
            if mcs_ratio < args.min_mcs_ratio:
                print(f"  MCS coverage {mcs_ratio:.2f} < threshold "
                      f"{args.min_mcs_ratio:.2f} → using free alignment.")
                core = None

        # ----- Branch: use existing 3D or re-embed -----
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
            rmsd_str = f"{mcs_rmsd:.3f} Å" if mcs_rmsd is not None else "n/a"
            print(f"  Method: {method}  |  MCS RMSD: {rmsd_str}")
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
                        qmol_work, ref_noh,
                        prbCid=best_cid, refCid=0,
                        partialCharges=args.partial_charges,
                        renormalize=True,
                        nocheck=True,
                    )
                    print(f"  ESP ({args.partial_charges}) : {best_esp:.4f}")
                except Exception as e:
                    print(f"  ESP failed: {e}")

            out_noh = Chem.RemoveHs(qmol_work)
            out_noh.SetProp('_Name', name)
            out_noh.SetProp('ShapeTanimoto', f'{best_shape:.4f}')
            out_noh.SetProp('MCS_NumAtoms', str(mcs_natoms))
            out_noh.SetProp('MCS_CoverageRef', f'{mcs_ratio:.4f}')
            out_noh.SetProp('MCS_SMILES', core_smi)
            out_noh.SetProp('AlignmentMethod', method)
            if mcs_rmsd is not None:
                out_noh.SetProp('MCS_RMSD', f'{mcs_rmsd:.4f}')
            if best_esp is not None:
                out_noh.SetProp(f'ESPSim_{args.partial_charges}', f'{best_esp:.4f}')
            writer.write(out_noh, confId=best_cid)

        else:
            # Re-embed from scratch
            try:
                qmol_confs, method = embed_query(
                    qmol, core, ref_noh, args.num_confs
                )
            except ValueError as e:
                print(f"  Embedding failed: {e}. Skipping.")
                continue

            scores = score_conformers(qmol_confs, ref_noh)
            if not scores:
                print("  Shape scoring failed. Skipping.")
                continue

            best_shape, best_cid = scores[0]
            print(f"  Best ShapeTanimoto : {best_shape:.4f}  "
                  f"(conf {best_cid} / {len(scores)})")

            best_esp = None
            if args.esp:
                try:
                    best_esp = GetEspSim(
                        qmol_confs, ref_noh,
                        prbCid=best_cid, refCid=0,
                        partialCharges=args.partial_charges,
                        renormalize=True,
                        nocheck=True,
                    )
                    print(f"  ESP ({args.partial_charges}) : {best_esp:.4f}")
                except Exception as e:
                    print(f"  ESP failed: {e}")

            out_noh = Chem.RemoveHs(qmol_confs)
            out_noh.SetProp('_Name', name)
            out_noh.SetProp('ShapeTanimoto', f'{best_shape:.4f}')
            out_noh.SetProp('MCS_NumAtoms', str(mcs_natoms))
            out_noh.SetProp('MCS_CoverageRef', f'{mcs_ratio:.4f}')
            out_noh.SetProp('MCS_SMILES', core_smi)
            out_noh.SetProp('AlignmentMethod', method)
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
    print(f"\nDone. Output → {args.output}")


if __name__ == '__main__':
    main()
