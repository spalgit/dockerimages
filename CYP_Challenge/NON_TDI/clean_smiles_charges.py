"""
Apply the four charge/bond-order corrections to the SMILES column of the CYP
challenge CSV files and write the result back as a new `SMILES_clean` column.

The correction logic is NOT re-implemented here: the function definitions are
pulled verbatim out of the existing SDF scripts

    neutralize_sulfonamide_N.py   R-SO2-N(-)-R'  -> R-SO2-NH-R'
    neutralize_nplus.py           pyridinium [nH+] -> n   (quaternary left alone)
    neutralize_morpholine_N.py    morpholine [NH+] -> N   (quaternary left alone)
    neutralize_iminium_N.py       amidine  C=[NH+] -> C=N  (only the sp2 N; a basic
                                  site elsewhere keeps its charge)
    fix_amide.py                  N=C([O-])      -> NC(=O)
    fix_phenol.py                 c[O-]          -> cOH

and applied in that same order.  Only the `def` blocks are exec'd, so the
file-list loops at the bottom of those scripts never run.

Molecules are carried as RDKit mols with explicit hydrogens (the equivalent of
the SDF round-trip the original scripts do) and written back out as clean,
canonical, implicit-H SMILES.

Usage:
    python clean_smiles_charges.py
"""
import ast
import sys
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent

SOURCE_SCRIPTS = [
    "neutralize_sulfonamide_N.py",
    "neutralize_nplus.py",
    "neutralize_morpholine_N.py",
    "neutralize_iminium_N.py",
    "fix_amide.py",
    "fix_phenol.py",
]

FILES = sorted(HERE.glob("cyp-challenge-TRAIN_direct_inhibition_with_single_shot_LABELLED*.csv")) + [
    HERE / "cyp-challenge-TEST-BLINDED_ligprepped.csv"
]

# Column layout of the CSVs:
#   SMILES             the cleaned structures this script produces (the column
#                      to model on)
#   SMILES_ligprepped  the LigPrep output that is read as input
#   SMILES_raw         the un-prepped source structure, used as the fallback
SRC_COL = "SMILES_ligprepped"
RAW_COL = "SMILES_raw"
OUT_COL = "SMILES"

# LigPrep artifacts: structures where the prepped SMILES has wrong bond orders /
# charges rather than a legitimate ionization state.  These cannot be repaired by
# a local fix (the azolium cases are keto/enol tautomer errors), so the affected
# compounds fall back to the un-prepped SMILES_raw.
#
#   azolium : N-substituted aromatic n+ in a triazole/tetrazole/triazinone that
#             is neutral in the raw structure  ([nD3+])
#   iminol  : an amide written as its iminol tautomer, C(-OH)=N- .  Charge-agnostic:
#             neutralize_iminium_N may already have stripped the + by this point.
#   nH_plus : a pyridinium the [n+] fix had to skip
#   iminium : a protonated amidine / guanidine / iminium,  C=[NH+]-  .  Reverting
#             also repairs the tautomer and the odd dearomatized kekulization
#             LigPrep emits for some of these (e.g. C2=c3ccccc3=[NH+]).
#   imine_taut: an amine tautomerised to an exocyclic imine, Ar-NH-C(=N)- written
#             as Ar-N=c(-[nH])-, which dearomatizes the ring.  (Genuine quinoid
#             dyes such as methylene blue are charged in the raw structure and so
#             are never flagged.)
#   exo_imine: an amidine/guanidine left as its exocyclic primary imine tautomer,
#             N-C(=NH)-N, where the raw structure puts the double bond in the ring.
#             Only reachable after neutralize_iminium_N has run.
ARTIFACT_PATTERNS = {
    "azolium": Chem.MolFromSmarts("[n+;D3]"),
    "iminol": Chem.MolFromSmarts("[OX2H1][CX3]=[N]"),
    "nH_plus": Chem.MolFromSmarts("[n+;H1]"),
    "iminium": Chem.MolFromSmarts("[N+;!a]=[#6]"),
    "imine_taut": Chem.MolFromSmarts("[c;R]=[C,N;!R]"),
    "exo_imine": Chem.MolFromSmarts("[NX2H1]=[CX3][NX3]"),
}


def find_artifacts(clean_smi, raw_smi):
    """Return the list of artifact classes present in the cleaned structure but
    absent from a neutral raw structure.  An empty list means 'keep the cleaned
    structure'.  A charged raw structure means the cation is real (e.g. a
    cetylpyridinium or a berberine-type alkaloid), so nothing is flagged."""
    mol = Chem.MolFromSmiles(clean_smi) if clean_smi else None
    raw = Chem.MolFromSmiles(str(raw_smi)) if isinstance(raw_smi, str) else None
    if mol is None or raw is None:
        return []
    if Chem.GetFormalCharge(raw) != 0:
        return []
    if mol.GetNumHeavyAtoms() != raw.GetNumHeavyAtoms():
        return []                     # not the same skeleton - do not substitute
    return [
        name for name, patt in ARTIFACT_PATTERNS.items()
        if mol.HasSubstructMatch(patt) and not raw.HasSubstructMatch(patt)
    ]


def load_fixers():
    """Exec only the module-level imports/assignments/function defs of each
    source script, so we reuse their exact code without triggering their
    SDF file loops.  Each script gets its OWN namespace - they all define a
    module-level `PATT`, which would otherwise clobber one another."""
    fx = {}
    for script in SOURCE_SCRIPTS:
        tree = ast.parse((HERE / script).read_text())
        keep = [
            node for node in tree.body
            if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))
            or (isinstance(node, ast.Assign)
                and not (isinstance(node.targets[0], ast.Name)
                         and node.targets[0].id == "FILES"))
        ]
        ns = {}
        exec(compile(ast.Module(body=keep, type_ignores=[]), script, "exec"), ns)
        name = Path(script).stem            # function name == file stem
        if name not in ns:
            sys.exit(f"{script}: no function named {name}()")
        fx[name] = ns[name]
    return fx


FX = load_fixers()


def clean_smiles(smi):
    """Return (clean_smiles, tally_dict, skipped_steps).

    clean_smiles is None if the molecule could not be parsed or the corrected
    molecule could not be sanitized."""
    tally = {"sulfonamide": 0, "nplus": 0, "morpholine": 0, "iminium": 0,
             "amide": 0, "phenol": 0}
    skipped = []
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None, tally, skipped

    # Round-trip through a mol block so the molecule behaves exactly like one
    # read from an SDF.  Atoms parsed from SMILES as [N-] / [O-] carry
    # NoImplicit=True, so simply zeroing their charge would leave a radical
    # instead of an N-H / O-H; the mol-block parser recomputes implicit Hs.
    mol = Chem.MolFromMolBlock(Chem.MolToMolBlock(mol), removeHs=False)
    if mol is None:
        return None, tally, skipped
    # explicit Hs, as in the ligprepped SDFs: neutralize_nplus looks for
    # explicit H neighbours
    mol = Chem.AddHs(mol)

    # Each step is applied defensively: if a correction makes a particular
    # molecule un-sanitizable, that step is skipped and the molecule carries on
    # from its pre-step state rather than aborting the whole file.
    def step(mol, fn_name, key):
        try:
            result = FX[fn_name](mol)
        except Exception:
            skipped.append(fn_name)
            return mol
        new_mol, n = result[0], result[1]
        tally[key] = int(n)
        return new_mol

    mol = step(mol, "neutralize_sulfonamide_N", "sulfonamide")
    mol = step(mol, "neutralize_nplus", "nplus")
    mol = step(mol, "neutralize_morpholine_N", "morpholine")
    mol = step(mol, "neutralize_iminium_N", "iminium")
    mol = step(mol, "fix_amide", "amide")
    mol = step(mol, "fix_phenol", "phenol")

    # clean up: drop explicit Hs, re-sanitize, canonicalise
    try:
        mol = Chem.RemoveHs(mol)
        Chem.SanitizeMol(mol)
        out = Chem.MolToSmiles(mol)
    except Exception:
        return None, tally, skipped
    if Chem.MolFromSmiles(out) is None:
        return None, tally, skipped
    return out, tally, skipped


def main():
    for path in FILES:
        # round_trip: the default C parser perturbs the last ULP of some
        # floats, which would rewrite unrelated label columns
        df = pd.read_csv(path, float_precision="round_trip")

        # Files that predate the rename still carry the LigPrep structures in a
        # plain "SMILES" column - migrate them so a re-run reads the LigPrep
        # structures, never the cleaned ones (this keeps the script idempotent).
        if SRC_COL not in df.columns:
            if "SMILES" in df.columns and OUT_COL != "SMILES":
                df = df.rename(columns={"SMILES": SRC_COL})
            elif "SMILES" in df.columns and "SMILES_clean" in df.columns:
                df = df.rename(columns={"SMILES": SRC_COL, "SMILES_clean": OUT_COL})
            elif "SMILES" in df.columns:
                df = df.rename(columns={"SMILES": SRC_COL})
            else:
                sys.exit(f"{path.name}: no '{SRC_COL}' column")

        totals = {"sulfonamide": 0, "nplus": 0, "morpholine": 0, "iminium": 0,
                  "amide": 0, "phenol": 0}
        cleaned, n_failed, n_changed, n_skipped = [], 0, 0, 0
        reverted = []

        raw_series = df[RAW_COL] if RAW_COL in df.columns else None

        for i, smi in enumerate(df[SRC_COL].astype(str)):
            out, tally, skipped = clean_smiles(smi)
            if skipped:
                n_skipped += 1
            if out is None:
                n_failed += 1
                cleaned.append(smi)          # keep the original if unparseable
                continue

            # LigPrep-artifact fallback: re-derive from SMILES_raw
            if raw_series is not None:
                bad = find_artifacts(out, raw_series.iloc[i])
                if bad:
                    raw_out, raw_tally, _ = clean_smiles(str(raw_series.iloc[i]))
                    if raw_out is not None:
                        reverted.append((df["Molecule_Name"].iloc[i], ",".join(bad)))
                        out, tally = raw_out, raw_tally

            for k, v in tally.items():
                totals[k] += v
            if out != Chem.CanonSmiles(smi):
                n_changed += 1
            cleaned.append(out)

        df[OUT_COL] = cleaned

        # keep SMILES first, then the provenance columns
        rest = [c for c in df.columns
                if c not in ("Molecule_Name", OUT_COL, SRC_COL, RAW_COL)]
        df = df[["Molecule_Name", OUT_COL, SRC_COL, RAW_COL] + rest]
        df.to_csv(path, index=False)

        print(f"{path.name}")
        print(f"  rows={len(df)}  changed={n_changed}  unparseable={n_failed}  "
              f"molecules with a skipped step={n_skipped}")
        print(f"  sulfonamide N(-)={totals['sulfonamide']}  [n+]H={totals['nplus']}  "
              f"morpholine [NH+]={totals['morpholine']}  "
              f"iminium C=[NH+]={totals['iminium']}  "
              f"amide N=C([O-])={totals['amide']}  phenol c[O-]={totals['phenol']}")
        print(f"  reverted to {RAW_COL} (LigPrep artifacts)={len(reverted)}")
        for name, why in reverted:
            print(f"      {name}  [{why}]")


if __name__ == "__main__":
    main()
