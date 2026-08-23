#!/usr/bin/env python3
"""
Parallel PROTAC-aware MCS overlay.

Identical chemistry to mcs_overlay_protac.py -- per-terminus MCS, combined
constrained embedding / rigid alignment, shape (+ optional ESP) scoring -- but
the per-query work is distributed across a process pool. Each query molecule is
fully independent (its own MCS core, its own conformer generation and scoring),
so molecules are farmed out to `--nproc` worker processes.

Determinism / ordering:
  * Results are collected with Pool.imap, so output SDF records appear in the
    same order as the input query file regardless of which worker finished
    first (and regardless of --nproc).
  * Each worker still uses the same fixed random seed as the serial script, so
    a given molecule embeds identically no matter which process handles it.
  * Every worker's progress text is captured and re-printed by the main process
    in input order, so the console log reads exactly like the serial run
    instead of interleaving across processes.

Usage (run inside the espsim conda environment):
  conda run -n espsim python mcs_overlay_protac_parallel.py \
      reference.sdf query.sdf output.sdf \
      --warhead_smarts "..." [--e3_smarts "..."] [--nproc N] [options]

All options from mcs_overlay_protac.py are supported, plus:
  --nproc INT     Number of worker processes (default: all CPU cores).
  --timeout SECS  Per-molecule wall-clock limit (default 20 s). A molecule that
                  exceeds it is killed and skipped, and the run moves on to the
                  next one. The limit covers the whole per-molecule job (MCS
                  core, embedding / alignment, shape + ESP scoring), because
                  RDKit spends that time inside C++ calls that cannot be
                  interrupted cooperatively -- each molecule therefore runs in
                  its own child process that can be terminated outright.
                  --timeout 0 disables the limit and restores the plain
                  Pool-based execution (with --nproc 1 running everything
                  in-process, useful for debugging / profiling).
"""

import argparse
import contextlib
import copy
import io
import multiprocessing as mp
import os
import sys
import time

from multiprocessing import Pool

from rdkit import Chem
from rdkit import RDLogger

# Make sure molecule properties + conformers survive pickling to/from workers.
Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mcs_overlay import (
    find_mcs_core,
    embed_query,
    align_existing_3d,
    score_conformers,
)
from mcs_overlay_protac import build_combined_core

sys.path.insert(0, '/home/spal/espsim')
from espsim import GetEspSim


# ---------------------------------------------------------------------------
# SDF serialisation (workers return text; main concatenates in order)
# ---------------------------------------------------------------------------

def mol_to_sdf_block(mol, conf_id):
    """
    Serialise one conformer of `mol` to a complete SDF record (molblock +
    property block + '$$$$'), matching Chem.SDWriter output. Private/computed
    properties (names starting with '_') are excluded, same as SDWriter.
    """
    out = Chem.MolToMolBlock(mol, confId=conf_id)
    for name in mol.GetPropNames():  # excludes private + computed by default
        out += f">  <{name}>\n{mol.GetProp(name)}\n\n"
    out += "$$$$\n"
    return out


# ---------------------------------------------------------------------------
# Worker: process a single query molecule
# ---------------------------------------------------------------------------

# Per-process globals, populated once by init_worker (avoids re-pickling the
# reference / SMARTS for every task).
_G = {}


def init_worker(ref_path, warhead_smarts, e3_smarts, args_dict):
    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
    RDLogger.DisableLog('rdApp.*')  # keep worker stderr from interleaving
    ref = Chem.MolFromMolFile(ref_path, removeHs=True)
    _G['ref'] = ref
    _G['warhead_patt'] = Chem.MolFromSmarts(warhead_smarts) if warhead_smarts else None
    _G['e3_patt'] = Chem.MolFromSmarts(e3_smarts) if e3_smarts else None
    _G['args'] = argparse.Namespace(**args_dict)


def process_one(task):
    """
    Handle one query molecule. Returns (index, name, log_text, sdf_blocks).
    sdf_blocks is a list of complete SDF record strings (empty if the molecule
    was skipped). Mirrors the per-query body of mcs_overlay_protac.main exactly;
    all progress printing is captured into log_text.
    """
    index, qmol, name = task
    ref_noh = _G['ref']
    warhead_patt = _G['warhead_patt']
    e3_patt = _G['e3_patt']
    args = _G['args']
    complete_rings = not args.no_complete_rings

    buf = io.StringIO()
    blocks = []
    with contextlib.redirect_stdout(buf):
        print(f"[{index + 1}] {name}  ({qmol.GetNumAtoms()} heavy atoms)")

        core, method_hint, info = build_combined_core(
            ref_noh, qmol, warhead_patt, e3_patt,
            complete_rings, timeout=10,
            min_mcs_ratio=args.min_mcs_ratio,
            min_terminus_atoms=args.min_terminus_atoms,
        )

        core_smi = Chem.MolToSmiles(core) if core is not None else "N/A"
        mcs_natoms = core.GetNumAtoms() if core is not None else 0

        if core is None:
            core, mcs_natoms, mcs_ratio = find_mcs_core(ref_noh, qmol, complete_rings)
            core_smi = Chem.MolToSmiles(core) if core is not None else "N/A"
            method_hint = "wholemol_MCS_fallback" if core is not None else "CrippenO3A_fallback"

        try:
            if args.use_existing_3d:
                if qmol.GetNumConformers() == 0:
                    print("  No 3D conformer in query molecule. Skipping.")
                    return index, name, buf.getvalue(), blocks
                qmol_work = copy.deepcopy(qmol)
                try:
                    method, mcs_rmsd = align_existing_3d(qmol_work, ref_noh, core)
                except ValueError as e:
                    print(f"  Alignment failed: {e}. Skipping.")
                    return index, name, buf.getvalue(), blocks
                rmsd_str = f"{mcs_rmsd:.3f} A" if mcs_rmsd is not None else "n/a"
                print(f"  Method: {method_hint} / {method}  |  MCS RMSD: {rmsd_str}")
                scores = score_conformers(qmol_work, ref_noh)
                if not scores:
                    print("  Shape scoring failed. Skipping.")
                    return index, name, buf.getvalue(), blocks
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
                blocks.append(mol_to_sdf_block(out_noh, best_cid))

            else:
                try:
                    qmol_confs, method = embed_query(qmol, core, ref_noh, args.num_confs)
                except ValueError as e:
                    print(f"  Embedding failed: {e}. Skipping.")
                    return index, name, buf.getvalue(), blocks

                scores = score_conformers(qmol_confs, ref_noh)
                if not scores:
                    print("  Shape scoring failed. Skipping.")
                    return index, name, buf.getvalue(), blocks

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
                        blocks.append(mol_to_sdf_block(m, cid))
                else:
                    blocks.append(mol_to_sdf_block(out_noh, best_cid))
        except Exception as e:  # never let one bad molecule kill the pool
            print(f"  Unexpected error: {e}. Skipping.")
            return index, name, buf.getvalue(), []

    return index, name, buf.getvalue(), blocks


# ---------------------------------------------------------------------------
# Timed execution: one killable child process per molecule
# ---------------------------------------------------------------------------

def _worker_entry(task, conn, init_args):
    """
    Child-process entry point: run one molecule and send the result back over
    its private pipe. Under 'fork' the parent has already populated _G, so the
    reference / SMARTS are inherited for free; under 'spawn' we initialise here.
    """
    try:
        if not _G:
            init_worker(*init_args)
        conn.send(process_one(task))
    except Exception as e:
        index, _qmol, name = task
        conn.send((index, name, f"[{index + 1}] {name}\n  Unexpected error: {e}. Skipping.\n", []))
    finally:
        conn.close()


def run_timed(tasks, nproc, timeout, init_args, fh, total):
    """
    Run `tasks` with at most `nproc` molecules in flight, each in its own child
    process. Any child still running after `timeout` seconds is terminated and
    its molecule skipped. Results are written in input order, exactly like the
    Pool.imap path.

    Returns (n_written, n_timeout).
    """
    try:
        ctx = mp.get_context('fork')
        init_worker(*init_args)   # children inherit this; no per-molecule re-read
    except ValueError:            # no fork() on this platform
        ctx = mp.get_context('spawn')

    pending = list(reversed(tasks))   # pop() from the end == input order
    running = {}                      # index -> [proc, reader, start_time, task]
    results = {}                      # index -> (log_text, blocks)
    by_reader = {}                    # reader connection -> index

    next_out = 0
    n_done = 0
    n_written = 0
    n_timeout = 0

    while next_out < total:
        # --- launch as many molecules as the worker budget allows ---
        while pending and len(running) < nproc:
            task = pending.pop()
            reader, writer = ctx.Pipe(duplex=False)
            proc = ctx.Process(target=_worker_entry, args=(task, writer, init_args), daemon=True)
            proc.start()
            writer.close()   # so the reader sees EOF as soon as the child exits
            running[task[0]] = [proc, reader, time.monotonic(), task]
            by_reader[reader] = task[0]

        # --- collect whatever is ready (short wait doubles as the timeout tick) ---
        ready = mp.connection.wait(list(by_reader), timeout=0.1) if by_reader else []
        for reader in ready:
            index = by_reader.pop(reader)
            proc, _reader, _t0, task = running.pop(index)
            try:
                _idx, _name, log_text, blocks = reader.recv()
            except EOFError:   # child died without answering (segfault / OOM kill)
                proc.join()
                log_text = (f"[{index + 1}] {task[2]}\n"
                            f"  Worker died (exit code {proc.exitcode}). Skipping.\n")
                blocks = []
            reader.close()
            proc.join()
            results[index] = (log_text, blocks)

        # --- kill anything past its wall-clock budget ---
        now = time.monotonic()
        for index, (proc, reader, t0, task) in list(running.items()):
            if now - t0 <= timeout:
                continue
            proc.terminate()
            proc.join()
            by_reader.pop(reader, None)
            reader.close()
            del running[index]
            n_timeout += 1
            results[index] = (
                f"[{index + 1}] {task[2]}  ({task[1].GetNumAtoms()} heavy atoms)\n"
                f"  Timed out after {timeout:g} s. Skipping.\n", [])

        # --- emit completed molecules in input order ---
        while next_out in results:
            log_text, blocks = results.pop(next_out)
            n_done += 1
            n_written += write_result(fh, n_done, total, log_text, blocks)
            next_out += 1

    return n_written, n_timeout


def write_result(fh, n_done, total, log_text, blocks):
    """Print one molecule's captured log with a progress tag, write its records."""
    body = log_text[log_text.find(']') + 2:] if log_text.startswith('[') else log_text
    sys.stdout.write(f"\n[{n_done}/{total}] {body}")
    for b in blocks:
        fh.write(b)
    return len(blocks)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Parallel PROTAC-aware MCS overlay (per-query process pool)."
    )
    parser.add_argument('reference', help='Reference 3D SDF (single molecule)')
    parser.add_argument('query', help='Query SDF (one or more molecules)')
    parser.add_argument('output', help='Output SDF with best-aligned poses')
    parser.add_argument('--warhead_smarts', default=None,
                        help='SMARTS for the warhead terminus (optional).')
    parser.add_argument('--e3_smarts', default=None,
                        help='SMARTS for the E3-ligand terminus (optional). '
                             'At least one of --warhead_smarts / --e3_smarts is required.')
    parser.add_argument('--num_confs', type=int, default=50,
                        help='Conformers per query molecule (default 50; '
                             'ignored with --use_existing_3d)')
    parser.add_argument('--min_mcs_ratio', type=float, default=0.1,
                        help='Min MCS / terminus-fragment heavy-atom coverage (default 0.1)')
    parser.add_argument('--min_terminus_atoms', type=int, default=3,
                        help='Min MCS atoms within a terminus to keep it (default 3)')
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
    parser.add_argument('--nproc', type=int, default=os.cpu_count(),
                        help='Number of worker processes (default: all CPU cores)')
    parser.add_argument('--timeout', type=float, default=20.0,
                        help='Per-molecule wall-clock limit in seconds (default 20). '
                             'A molecule exceeding it is killed and skipped. '
                             '0 disables the limit.')
    args = parser.parse_args()

    if args.warhead_smarts is None and args.e3_smarts is None:
        sys.exit("Provide at least one of --warhead_smarts / --e3_smarts.")

    # Validate SMARTS up front (before spawning workers).
    if args.warhead_smarts is not None and Chem.MolFromSmarts(args.warhead_smarts) is None:
        sys.exit(f"Invalid --warhead_smarts: {args.warhead_smarts}")
    if args.e3_smarts is not None and Chem.MolFromSmarts(args.e3_smarts) is None:
        sys.exit(f"Invalid --e3_smarts: {args.e3_smarts}")

    # --- Load reference (also used by workers, re-read there from path) ---
    ref_mol = Chem.MolFromMolFile(args.reference, removeHs=True)
    if ref_mol is None:
        sys.exit(f"Cannot read reference: {args.reference}")
    if ref_mol.GetNumConformers() == 0:
        sys.exit("Reference molecule has no 3D coordinates.")
    ref_noh = ref_mol

    if args.warhead_smarts is not None and not ref_noh.HasSubstructMatch(Chem.MolFromSmarts(args.warhead_smarts)):
        print("WARNING: warhead SMARTS does not match the reference at all.")
    if args.e3_smarts is not None and not ref_noh.HasSubstructMatch(Chem.MolFromSmarts(args.e3_smarts)):
        print("WARNING: E3-ligand SMARTS does not match the reference at all.")

    # --- Load queries ---
    suppl = Chem.SDMolSupplier(args.query, removeHs=True)
    tasks = []
    for i, m in enumerate(suppl):
        if m is not None:
            name = m.GetProp('_Name') if m.HasProp('_Name') else f'mol_{i}'
            tasks.append((len(tasks), m, name))
    if not tasks:
        sys.exit(f"No valid molecules in {args.query}")

    nproc = max(1, min(args.nproc, len(tasks)))
    mode = "existing 3D (rigid MCS align)" if args.use_existing_3d else f"{args.num_confs} conformers"
    print(f"Reference : {ref_noh.GetNumAtoms()} heavy atoms")
    print(f"Queries   : {len(tasks)} molecules")
    print(f"Mode      : {mode}")
    print(f"Workers   : {nproc} process(es)")
    print(f"Timeout   : {f'{args.timeout:g} s per molecule' if args.timeout > 0 else 'none'}")

    args_dict = vars(args).copy()
    init_args = (args.reference, args.warhead_smarts, args.e3_smarts, args_dict)

    n_written = 0
    n_done = 0
    n_timeout = 0
    total = len(tasks)

    with open(args.output, 'w') as fh:
        if args.timeout > 0:
            # One killable child process per molecule, <= nproc in flight.
            n_written, n_timeout = run_timed(tasks, nproc, args.timeout, init_args, fh, total)
        elif nproc == 1:
            # Single-process path (no pool) -- handy for debugging / profiling.
            init_worker(*init_args)
            for index, name, log_text, blocks in map(process_one, tasks):
                n_done += 1
                n_written += write_result(fh, n_done, total, log_text, blocks)
        else:
            with Pool(processes=nproc, initializer=init_worker, initargs=init_args) as pool:
                # imap keeps input order; chunksize=1 balances long/short molecules.
                for index, name, log_text, blocks in pool.imap(process_one, tasks, chunksize=1):
                    n_done += 1
                    # Re-number the leading "[i]" tag to a "[done/total]" progress tag.
                    n_written += write_result(fh, n_done, total, log_text, blocks)

    summary = f"\nDone. {n_written} record(s) from {total} molecule(s) -> {args.output}"
    if n_timeout:
        summary += f"  ({n_timeout} molecule(s) timed out after {args.timeout:g} s)"
    print(summary)


if __name__ == '__main__':
    main()
