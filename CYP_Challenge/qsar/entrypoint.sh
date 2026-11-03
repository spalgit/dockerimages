#!/usr/bin/env bash
# Subcommand dispatcher for the cyp-qsar image.
#
# The image carries no project code, so every subcommand except `selftest`
# needs the repository mounted at /repo. `selftest` deliberately does not:
# it checks the IMAGE, and must be runnable before anything is mounted.
set -euo pipefail

usage() {
    cat <<'EOF'
cyp-qsar — conventional QSAR models for the CYP-Challenge activity track.

  selftest              check the image itself (no mount needed)
  check                 check the mounted repo: caches, split pairing  [/repo]
  features [args]       build PaDEL / RDKit / Mordred matrices         [/repo]
  arm <family> [args]   train one family: lgbm | xu | oct2             [/repo]
  report [args]         per-family comparison table                    [/repo]
  python [args]         raw interpreter
  shell                 interactive bash

Anything else is executed verbatim, so `docker run … cyp-qsar:1.0 ls -la` works.
EOF
}

# ─── image-level selftest ──────────────────────────────────────────────────
# Asserts the three things that have actually broken before: a missing JRE
# (PaDEL fails at run time, not build time), a mordred version whose
# descriptor set has drifted, and mordred's parallel path dying on the
# forkserver — the bug that pinned python to 3.12 in env.yml.
selftest() {
    echo "── java ─────────────────────────────────────────────"
    java -version 2>&1 | head -2

    echo "── python stack ─────────────────────────────────────"
    python - <<'PY'
import importlib, sys
print(f"  python     {sys.version.split()[0]}")
mods = ["numpy", "pandas", "scipy", "sklearn", "lightgbm", "optuna",
        "rdkit", "pyarrow", "joblib", "mordred", "padelpy"]
missing = []
for m in mods:
    try:
        mod = importlib.import_module(m)
        print(f"  {m:10s} {getattr(mod, '__version__', 'n/a')}")
    except Exception as e:
        missing.append(m)
        print(f"  {m:10s} MISSING ({type(e).__name__})")
if missing:
    sys.exit(f"FAIL: missing modules: {', '.join(missing)}")
PY

    echo "── mordred ──────────────────────────────────────────"
    python - <<'PY'
import sys
from rdkit import Chem
from mordred import Calculator, descriptors

calc = Calculator(descriptors, ignore_3D=True)
n = len(calc.descriptors)
print(f"  descriptors: {n}")
if n != 1613:
    sys.exit(f"FAIL: expected 1613 mordred 2D descriptors, got {n} — "
             "the version drifted and every cached matrix would change width")

# The forkserver bug is silent: it falls back to serial and just runs slow.
# Assert the parallel path actually returns, so a regression is loud.
mols = [Chem.MolFromSmiles(s) for s in
        ("CCO", "c1ccccc1C(=O)NC2CCNCC2", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C")]
df = calc.pandas(mols, quiet=True, nproc=2)
if df.shape != (3, 1613):
    sys.exit(f"FAIL: parallel mordred returned {df.shape}, expected (3, 1613)")
print(f"  parallel path (nproc=2): ok, shape {df.shape}")
PY

    echo "── padel ────────────────────────────────────────────"
    python - <<'PY'
import os, sys, tempfile
from padelpy import padeldescriptor

# One cheap fingerprint type end to end: this is what proves the jar can
# reach a working JVM. The other eight differ only in the XML flag.
with tempfile.TemporaryDirectory() as d:
    smi, out = os.path.join(d, "in.smi"), os.path.join(d, "out.csv")
    with open(smi, "w") as fh:
        fh.write("CCO\tm1\nc1ccccc1\tm2\n")
    padeldescriptor(mol_dir=smi, d_file=out, fingerprints=True,
                    d_2d=False, d_3d=False, retainorder=True,
                    removesalt=False, detectaromaticity=True,
                    threads=1, maxruntime=20000, sp_timeout=300)
    with open(out) as fh:
        header = fh.readline().rstrip("\n").split(",")
        rows = fh.read().strip().splitlines()
print(f"  default fingerprinter: {len(header) - 1} bits over {len(rows)} molecules")
if len(rows) != 2:
    sys.exit(f"FAIL: padel returned {len(rows)} rows, expected 2")
PY

    echo
    echo "SELFTEST PASSED"
}

need_repo() {
    [ -d /repo ] && [ -n "$(ls -A /repo 2>/dev/null)" ] || {
        echo "error: /repo is empty — mount the repository, e.g." >&2
        echo "  -v \"\$HOME/CYP-Challenge-Tutorial:/repo\"" >&2
        exit 2
    }
}

cmd="${1:-selftest}"; shift || true
case "$cmd" in
    selftest)        selftest ;;
    check)           need_repo; exec python /repo/qsar_selftest.py "$@" ;;
    features)        need_repo; exec python /repo/qsar_features.py "$@" ;;
    arm)
        need_repo
        fam="${1:-}"; shift || true
        case "$fam" in
            lgbm) exec python /repo/lgbm_cyp_pic50_4iso.py "$@" ;;
            xu)   exec python /repo/xu_grid_cyp_pic50.py "$@" ;;
            oct2) exec python /repo/mlp_oct2_cyp_pic50.py "$@" ;;
            *)    echo "arm: need a family — lgbm | xu | oct2" >&2; exit 2 ;;
        esac ;;
    report)          need_repo; exec python /repo/report_qsar_arms.py "$@" ;;
    python)          exec python "$@" ;;
    shell|bash)      exec bash "$@" ;;
    -h|--help|help)  usage ;;
    *)               exec "$cmd" "$@" ;;
esac
