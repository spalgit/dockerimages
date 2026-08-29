"""ChemProp + XGBoost MCC-optimised ensemble for the CYP TDI track.

One model **per isoform** (CYP2D6 and CYP3A4 are trained separately, with their own
hyperparameters, their own blend weight and their own operating point), each an
ensemble of two very different learners:

    arm A   ChemProp D-MPNN, multitask binary, Optuna-tuned
    arm B   XGBoost on descriptors + count fingerprints + MBI alerts,
            tuned over the number of trees
    blend   rank-average of the two, weight chosen on out-of-fold MCC

Everything downstream of the two arms — the blend weight, the decision threshold, the
choice of arm — is selected on **MCC**, which is the metric the track is scored on.

────────────────────────────────────────────────────────────────────────────────
WHY AN ENSEMBLE OF THESE TWO
────────────────────────────────────────────────────────────────────────────────
On the live TDI board the tree baseline beats our D-MPNN entry on both endpoints
(LGBM-baseline 0.2881 vs Cheminfo 0.2279), which is the opposite of the regression
board. The two families fail differently: the D-MPNN learns a smooth function of the
graph and generalises across scaffolds; the tree model splits directly on the
mechanism-based-inactivation motifs (methylenedioxyphenyl, furan, thiophene, terminal
acetylene, ...) that drive TDI and that a hashed graph representation dilutes. Rank-
averaging them recovers both behaviours, and the weight is not assumed — it is swept.

────────────────────────────────────────────────────────────────────────────────
HYPERPARAMETER OPTIMISATION
────────────────────────────────────────────────────────────────────────────────
*ChemProp* — Optuna TPE over the same space we used for PXR in QSARtuna
(``pxr_chemprop_optimize.json``): depth, message/FFN widths, FFN layers, dropout,
max_lr with init/final ratios, warmup, aggregation (+ its norm), batch-norm. Two knobs
are added that are specific to this track: the auxiliary-head weight, and whether the
auxiliary heads are used at all. Each trial is scored by out-of-fold MCC on a subset of
the scaffold folds (``--hpo-folds``), reporting per fold so unpromising trials prune.

*XGBoost* — the sweep is over the **number of trees**, up to four values
(``--xgb-trees``, default 200/400/800/1600 at learning rate 0.03). Every other tree
hyperparameter is held fixed, so the comparison is a clean capacity sweep and the
winner is picked on the same out-of-fold MCC as everything else.

────────────────────────────────────────────────────────────────────────────────
THE LABEL, AND THE TWO THINGS THAT COST MCC IF YOU IGNORE THEM
────────────────────────────────────────────────────────────────────────────────
1.  ``is_TDI`` is exactly a threshold on a continuous quantity — verified on 100% of
    rows carrying both surfaces by :func:`verify_label_rule`:

        is_TDI  <=>  max(pIC50_TDI_condition, 4.0) - max(pIC50_direct, 4.0) > log10(2)

    Both surfaces ship for all four isoforms, so the unscored isoforms are free
    auxiliary signal. ChemProp gets them as extra binary heads; XGBoost gets them as
    out-of-fold stacked predictions (``--no-stack`` turns that off).

2.  For CYP3A4, 1,249 of 3,584 labels are compounds with a TDI-condition fit but *no*
    direct fit, and every one is labelled False by convention. "Was a direct curve
    fitted at all" is therefore part of the label, and it gets its own head/feature.

3.  The blind set's prevalence is not the training prevalence. Solved from the public
    leaderboard (every entry reports accuracy/precision/recall on the same 750
    compounds) it is 0.0706 for CYP2D6 and 0.2916 for CYP3A4, against 0.216 and 0.213
    in training. All MCC numbers here are computed on an out-of-fold set reweighted to
    the blind prevalence (``--prevalence train`` disables that).

────────────────────────────────────────────────────────────────────────────────
USAGE
────────────────────────────────────────────────────────────────────────────────
    conda activate chemprop
    pip install xgboost optuna            # not in the chemprop env by default

    python tdi_ensemble_chemprop_xgb.py                      # both isoforms, full run
    python tdi_ensemble_chemprop_xgb.py --isoform CYP2D6     # one isoform
    python tdi_ensemble_chemprop_xgb.py --quick              # ~10 min smoke test
    python tdi_ensemble_chemprop_xgb.py --n-trials 40 --hpo-folds 3

    # offline VM — -u so the progress log is not block-buffered behind the redirect
    export CYP_DATA_DIR=/data/cyp-challenge
    nohup python -u tdi_ensemble_chemprop_xgb.py > tdi_ensemble.log 2>&1 &

Per-isoform runs are resumable in the sense that matters: each isoform writes its own
test scores, and the combined submission is rebuilt from whatever isoform files exist,
so ``--isoform CYP2D6`` today and ``--isoform CYP3A4`` tomorrow still produce a valid
two-column submission.

The decision threshold is expressed as a predicted-positive **rate** and read off the
out-of-fold MCC curve. ``--operating-point theory`` (default) takes the MCC optimum of
a bi-normal ranker at the measured AUC — smooth, and it degrades gracefully when the
ranking is weak; ``--operating-point argmax`` takes the empirical bootstrap-averaged
argmax instead. Both maximise MCC; see :func:`choose_operating_point` for when the
second one misbehaves.

Runtime at the defaults (20 Optuna trials x 3 HPO folds, then 5 CV folds and 3 ensemble
seeds) is roughly 3-6 h per isoform on one GPU — the Optuna search is ~70% of it, so
``--n-trials 10 --hpo-folds 2`` roughly quarters the total. ``--quick`` is ~10 min.

Outputs (``OUT_DIR``, default ``TDI/output/ensemble_chemprop_xgb/``):

    my_ensemble_tdi_submission.csv   the file to upload
    <ISO>/chemprop_hpo_trials.csv    every Optuna trial, its params and its MCC
    <ISO>/best_params.json           chosen ChemProp params, tree count, blend weight
    <ISO>/xgb_trees_sweep.csv        MCC/AUC vs number of trees
    <ISO>/blend_sweep.csv            MCC vs blend weight
    <ISO>/arm_comparison.csv         chemprop / xgb / blend, ranked by MCC
    <ISO>/threshold_sweep.csv        MCC vs predicted-positive rate, per arm
    <ISO>/oof_predictions.csv        out-of-fold scores for every arm
    <ISO>/test_scores.csv            blended test scores + the boolean call
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*does not have many workers.*")
warnings.filterwarnings("ignore", message=".*GPU available but not used.*")

PROJECT_ROOT = Path(__file__).resolve().parent
BANNER = "=" * 78


# ══════════════════════════════════════════════════════════════════════════════
# Constants that describe the assay and the challenge, not our modelling choices
# ══════════════════════════════════════════════════════════════════════════════

TDI_ISOFORMS = ["CYP2D6", "CYP3A4"]          # the only two scored
ALL_ISOFORMS = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]

FILE_TRAIN_TDI = "cyp-challenge-TRAIN_TDI.csv"
FILE_TEST = "cyp-challenge-TEST-BLINDED.csv"
HF_PREFIX = "hf://datasets/openadmet/cyp-challenge-train-test"

PIC50_FLOOR = 4.0                             # below this the assay is unquantifiable
TDI_SHIFT_THRESHOLD = float(np.log10(2))      # a >2-fold shift defines is_TDI
ACTIVITY_CUT = PIC50_FLOOR + TDI_SHIFT_THRESHOLD

#: Blind-set positive rates, solved from the public leaderboard. See the module
#: docstring; the same numbers are used by ``lgbm_cyp_tdi_classifier.py``.
TEST_PREVALENCE = {"CYP2D6": 0.0706, "CYP3A4": 0.2916}


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Config:
    """Every knob in one place, so the notebook and the CLI drive identical code."""

    # data / output
    data_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("CYP_DATA_DIR", PROJECT_ROOT / "data"))
    )
    out_dir: Path = PROJECT_ROOT / "TDI" / "output" / "ensemble_chemprop_xgb"
    isoforms: list[str] = field(default_factory=lambda: list(TDI_ISOFORMS))

    # splitting
    n_folds: int = 5
    scaffold_seed: int = 42

    # ── XGBoost ───────────────────────────────────────────────────────────────
    #: The hyperparameter we sweep: number of boosting rounds, at most four values.
    xgb_n_estimators_grid: list[int] = field(default_factory=lambda: [200, 400, 800, 1600])
    xgb_seeds: list[int] = field(default_factory=lambda: [42, 123, 456])
    xgb_params: dict = field(
        default_factory=lambda: dict(
            max_depth=6,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.3,
            min_child_weight=5.0,
            reg_lambda=5.0,
            tree_method="hist",
            eval_metric="logloss",
            n_jobs=-1,
        )
    )
    #: Out-of-fold predictions of the auxiliary pIC50 surfaces, added as XGB features.
    use_stack_features: bool = True
    stack_n_estimators: int = 400

    # ── ChemProp ──────────────────────────────────────────────────────────────
    n_trials: int = 20                        # Optuna trials
    hpo_folds: int = 3                        # folds scored per trial
    hpo_max_epochs: int = 40
    hpo_patience: int = 8
    max_epochs: int = 60
    patience: int = 12
    batch_size: int = 64
    num_workers: int = 0
    chemprop_seeds: list[int] = field(default_factory=lambda: [42, 123, 456])
    optuna_seed: int = 42
    use_descriptors: bool = True              # RDKit 2D block alongside the graph
    desc_clip: float = 10.0
    desc_abs_max: float = 1e10

    # ── Operating point / metric ──────────────────────────────────────────────
    prevalence: str = "blind"                 # "blind" | "train"
    operating_point: str = "theory"           # "theory" | "argmax"
    pos_rate_grid: np.ndarray = field(
        default_factory=lambda: np.arange(0.01, 0.601, 0.005)
    )
    blend_weight_grid: np.ndarray = field(
        default_factory=lambda: np.round(np.arange(0.0, 1.001, 0.05), 3)
    )
    n_bootstrap: int = 300                    # for the smoothed MCC-vs-rate curve
    min_positive_rate: float = 0.01

    # ── Misc ──────────────────────────────────────────────────────────────────
    cache_features: bool = True
    progress: bool = False
    fit_test: bool = True

    def prevalence_for(self, iso: str, y_train: np.ndarray) -> float:
        """Positive rate the MCC is evaluated at."""
        if self.prevalence == "train":
            return float(np.mean(y_train))
        return TEST_PREVALENCE[iso]

    def iso_dir(self, iso: str) -> Path:
        d = self.out_dir / iso
        d.mkdir(parents=True, exist_ok=True)
        return d


def quick_config(**overrides) -> Config:
    """Small everything — for a smoke test that exercises every code path."""
    cfg = Config(
        n_folds=3,
        n_trials=3,
        hpo_folds=2,
        hpo_max_epochs=8,
        hpo_patience=3,
        max_epochs=12,
        patience=4,
        chemprop_seeds=[42],
        xgb_seeds=[42],
        xgb_n_estimators_grid=[200, 400],
        n_bootstrap=50,
        use_stack_features=False,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# ══════════════════════════════════════════════════════════════════════════════
# Data
# ══════════════════════════════════════════════════════════════════════════════

def read_table(cfg: Config, filename: str) -> pd.DataFrame:
    local = cfg.data_dir / filename
    if local.exists():
        print(f"  {filename}  <- {local}")
        return pd.read_csv(local)
    url = f"{HF_PREFIX}/{filename}"
    print(f"  {filename}  <- {url}")
    return pd.read_csv(url)


def floored_shift(df: pd.DataFrame, iso: str) -> pd.Series:
    """max(TDI-condition pIC50, floor) - max(direct pIC50, floor).

    NaN wherever either surface is missing. This is the exact quantity ``is_TDI``
    thresholds, so a model of it is a model of the label.
    """
    a = np.maximum(df[f"{iso}_pIC50_TDI_condition"], PIC50_FLOOR)
    b = np.maximum(df[f"{iso}_pIC50_direct_inhibition"], PIC50_FLOOR)
    return a - b


def verify_label_rule(df: pd.DataFrame) -> None:
    """Confirm ``is_TDI`` is still the floored-shift rule before relying on it.

    A guard, not a formality: if a data release changes the rule, the auxiliary heads
    and stacked features below stop being the right auxiliaries and you want to know.
    """
    print("\nLabel definition check")
    for iso in TDI_ISOFORMS:
        lab = df[f"{iso}_is_TDI"]
        shift = floored_shift(df, iso)
        m = lab.notna() & shift.notna()
        agree = float(((shift[m] > TDI_SHIFT_THRESHOLD) == lab[m].astype(bool)).mean())
        assigned = lab.notna() & df[f"{iso}_pIC50_direct_inhibition"].isna()
        n_assigned = int(assigned.sum())
        print(
            f"  {iso}: {int(lab.notna().sum()):5d} labels, "
            f"{100 * lab[lab.notna()].mean():4.1f}% positive | "
            f"shift rule reproduces {agree:.4f} on {int(m.sum())} rows | "
            f"{n_assigned} with no direct fit "
            f"({100 * lab[assigned].mean() if n_assigned else 0:.1f}% positive)"
        )
        if agree < 0.999:
            raise ValueError(f"{iso}: floored-shift rule no longer reproduces the label")


# ══════════════════════════════════════════════════════════════════════════════
# Structures
# ══════════════════════════════════════════════════════════════════════════════

_LARGEST_FRAGMENT = rdMolStandardize.LargestFragmentChooser()
_UNCHARGER = rdMolStandardize.Uncharger()


def mol_from_smiles(smi: str):
    """Parse, keep the largest fragment, neutralise. Returns None on failure.

    Matches the preprocessing in the other CYP scripts so out-of-fold numbers stay
    comparable across model families.
    """
    if not isinstance(smi, str) or not smi.strip():
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        mol = _LARGEST_FRAGMENT.choose(mol)
        mol = _UNCHARGER.uncharge(mol)
    except Exception:
        return None
    return mol


def build_mols(smiles: list[str]) -> tuple[list, np.ndarray]:
    """Returns (mols for the parseable rows, boolean keep-mask over the input)."""
    mols, keep = [], np.zeros(len(smiles), dtype=bool)
    for i, smi in enumerate(smiles):
        mol = mol_from_smiles(smi)
        if mol is not None:
            mols.append(mol)
            keep[i] = True
    return mols, keep


# ══════════════════════════════════════════════════════════════════════════════
# Mechanism-based-inactivation alerts
# ══════════════════════════════════════════════════════════════════════════════
# Motifs repeatedly implicated in mechanism-based CYP inactivation. Hypothesis-driven
# features, not a curated regulatory set: the model decides which earn a split. Counts,
# not bits, so "two furans" differs from "one".

MBI_ALERTS = {
    "methylenedioxyphenyl": "c1cc2OCOc2cc1",
    "furan": "c1ccoc1",
    "thiophene": "c1ccsc1",
    "benzofuran": "c1ccc2occc2c1",
    "benzothiophene": "c1ccc2sccc2c1",
    "terminal_alkyne": "[CX2]#[CH1]",
    "internal_alkyne": "[CX2]#[CX2]",
    "terminal_alkene": "[CX3H2]=[CX3]",
    "aniline": "[NX3;H2,H1;!$(NC=O)]c",
    "tert_aniline": "[NX3;H0;!$(NC=O)](c)([#6])[#6]",
    "nitroaromatic": "[$([NX3](=O)=O),$([NX3+](=O)[O-])]c",
    "thiourea": "[NX3][CX3](=[SX1])[NX3]",
    "thioamide": "[NX3][CX3]=[SX1]",
    "hydrazine": "[NX3][NX3]",
    "hydrazone": "[NX3][NX2]=[CX3]",
    "cyclopropylamine": "[NX3]C1CC1",
    "cyclopropyl": "C1CC1",
    "epoxide": "[OX2r3]1[#6r3][#6r3]1",
    "azole_N": "[nX3;H0;$(n1cncc1),$(n1cnnc1),$(n1cnnn1)]",
    "imidazole": "c1cnc[nH0]1",
    "triazole": "c1nc[nH0]n1",
    "pyridine_N": "n1ccccc1",
    "quinone": "O=C1C=CC(=O)C=C1",
    "catechol": "c1cc(O)c(O)cc1",
    "phenol": "[OX2H]c",
    "michael_acceptor": "[CX3]=[CX3][CX3]=[OX1]",
    "acyl_halide": "[CX3](=O)[F,Cl,Br,I]",
    "alkyl_halide": "[CX4][Cl,Br,I]",
    "thiol": "[SX2H]",
    "sulfide": "[#6][SX2][#6]",
    "sulfoxide": "[#6][SX3](=O)[#6]",
    "tert_amine": "[NX3;H0;!$(N[!#6]);!$(NC=[O,S,N])]([CX4])([CX4])[CX4]",
    "sec_amine": "[NX3;H1;!$(N[!#6]);!$(NC=[O,S,N])]([CX4])[CX4]",
    "basic_N": "[NX3;H0,H1,H2;!$(NC=[O,S,N]);!$(N[a]);!$(N=*)]",
    "piperazine": "C1CNCCN1",
    "piperidine": "C1CCNCC1",
    "morpholine": "C1COCCN1",
    "n_methyl": "[NX3][CH3]",
    "n_dealkyl_site": "[NX3;!$(NC=[O,S,N])][CX4H2][#6]",
    "o_dealkyl_site": "[OX2]([CH3])c",
    "benzylic_CH": "[CX4;H1,H2]c",
    "furanone": "O=C1OC=CC1",
    "isocyanate": "[NX2]=[CX2]=[OX1]",
    "aromatic_amine_ortho_OMe": "[NX3;H2,H1]c1ccccc1OC",
}

_ALERT_PATTERNS = {k: Chem.MolFromSmarts(v) for k, v in MBI_ALERTS.items()}
_BAD_ALERTS = [k for k, v in _ALERT_PATTERNS.items() if v is None]
if _BAD_ALERTS:  # a typo in a SMARTS must not silently become a column of zeros
    raise ValueError(f"unparseable MBI alert SMARTS: {_BAD_ALERTS}")


# ══════════════════════════════════════════════════════════════════════════════
# Features
# ══════════════════════════════════════════════════════════════════════════════

MORGAN_BITS = 2048
FCFP_BITS = 1024

_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=MORGAN_BITS)
_FCFP = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=FCFP_BITS,
    atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen(),
)

_DESC_LIST = [(name, fn) for name, fn in Descriptors.descList]
_DESC_NAMES = [n for n, _ in _DESC_LIST]


def descriptor_block(mols: list) -> np.ndarray:
    """RDKit 2D descriptors, one descriptor at a time.

    Per-descriptor try/except matters: ``MolecularDescriptorCalculator`` aborts a whole
    row on the first exception, which silently empties the block when one unusual test
    compound trips it. Here a failure costs one cell.
    """
    rows = []
    for mol in mols:
        row = []
        for _, fn in _DESC_LIST:
            try:
                v = fn(mol)
                row.append(float(v) if v is not None else np.nan)
            except Exception:
                row.append(np.nan)
        rows.append(row)
    return np.asarray(rows, dtype=float)


def alert_counts(mol) -> list[float]:
    return [float(len(mol.GetSubstructMatches(p, uniquify=True))) for p in _ALERT_PATTERNS.values()]


def featurize_tabular(mols: list) -> tuple[np.ndarray, list[str]]:
    """Descriptors + Morgan counts + FCFP counts + MACCS + MBI alerts, for XGBoost.

    Count fingerprints rather than bit vectors: for TDI the *number* of bioactivatable
    groups matters, and a tree can split on a count directly.
    """
    desc = descriptor_block(mols)
    morgan, fcfp, maccs, alerts = [], [], [], []
    for mol in mols:
        morgan.append(_MORGAN.GetCountFingerprintAsNumPy(mol))
        fcfp.append(_FCFP.GetCountFingerprintAsNumPy(mol))
        maccs.append(np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.uint8))
        alerts.append(alert_counts(mol))

    blocks = [
        (desc, [f"desc_{n}" for n in _DESC_NAMES]),
        (np.asarray(morgan, dtype=float), [f"ecfp4_{i}" for i in range(MORGAN_BITS)]),
        (np.asarray(fcfp, dtype=float), [f"fcfp4_{i}" for i in range(FCFP_BITS)]),
        (np.asarray(maccs, dtype=float), [f"maccs_{i}" for i in range(167)]),
        (np.asarray(alerts, dtype=float), [f"alert_{k}" for k in MBI_ALERTS]),
    ]
    X = np.hstack([b for b, _ in blocks])
    names = [n for _, ns in blocks for n in ns]
    return X, names


def clean_columns(X_train: np.ndarray, cfg: Config) -> np.ndarray:
    """Keep columns finite, non-constant and sanely scaled **on the training block**.

    Selection on train only: letting the test set veto a column leaks, and one unusual
    test compound can otherwise empty the block. The magnitude test is what stops
    ``Ipc`` — finite but routinely 1e20 — from dominating split-gain arithmetic.

    Note the spread and magnitude tests are evaluated only on the fully-finite columns,
    with plain ``std``/``max`` rather than ``np.nanstd``/``np.nanmax``. That is not
    stylistic: numpy 2.2.x returns **zero** from ``nanstd``/``nanvar`` along axis 0 for
    wide 2-D arrays (reproducible at (400, 3500)), which would silently drop every
    feature. Reductions over a NaN-free block do not go down that path.
    """
    finite = np.all(np.isfinite(X_train), axis=0)
    block = X_train[:, finite]
    varied = np.zeros(X_train.shape[1], dtype=bool)
    sane = np.zeros(X_train.shape[1], dtype=bool)
    if block.size:
        varied[finite] = block.std(axis=0) > 0
        sane[finite] = np.abs(block).max(axis=0) < cfg.desc_abs_max
    keep = finite & varied & sane
    print(f"  features: {X_train.shape[1]} -> {int(keep.sum())} kept")
    return keep


def scale_descriptors(scaler: StandardScaler, X: np.ndarray, cfg: Config) -> np.ndarray:
    """Standardise and clip, for the ChemProp ``x_d`` block.

    Any non-finite cell surviving from an unusual test compound becomes 0 after
    centring, which is the column mean.
    """
    clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(np.nan_to_num(scaler.transform(clean)), -cfg.desc_clip, cfg.desc_clip)


# ══════════════════════════════════════════════════════════════════════════════
# Folds
# ══════════════════════════════════════════════════════════════════════════════

def murcko_scaffold_folds(mols: list, n_folds: int, seed: int) -> np.ndarray:
    """Bemis-Murcko scaffold K-fold assignment, one fold index per compound.

    Scaffold groups are shuffled, ordered largest-first, and each dropped into the
    currently smallest fold, so whole scaffolds stay together and a held-out fold
    contains chemotypes the training folds never saw. 75.5% of the blind set sits on
    scaffolds absent from training, so a random split would flatter every arm here.
    """
    groups: dict[str, list[int]] = {}
    for i, mol in enumerate(mols):
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        except Exception:
            scaffold = ""
        groups.setdefault(scaffold or f"__singleton_{i}", []).append(i)

    keys = list(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    keys.sort(key=lambda k: -len(groups[k]))

    folds = np.zeros(len(mols), dtype=int)
    sizes = [0] * n_folds
    for k in keys:
        f = int(np.argmin(sizes))
        for i in groups[k]:
            folds[i] = f
        sizes[f] += len(groups[k])
    print(f"  scaffold folds: {len(keys)} scaffolds -> sizes {sizes}")
    return folds


# ══════════════════════════════════════════════════════════════════════════════
# MCC, the metric everything is selected on
# ══════════════════════════════════════════════════════════════════════════════

def weighted_mcc(y: np.ndarray, pred: np.ndarray, w: np.ndarray) -> float:
    """MCC with per-sample weights, so the out-of-fold set can be reweighted."""
    tp = float((w * (pred == 1) * (y == 1)).sum())
    tn = float((w * (pred == 0) * (y == 0)).sum())
    fp = float((w * (pred == 1) * (y == 0)).sum())
    fn = float((w * (pred == 0) * (y == 1)).sum())
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return float((tp * tn - fp * fn) / denom) if denom > 0 else 0.0


def prevalence_weights(y: np.ndarray, p_test: float) -> np.ndarray:
    """Importance weights moving the out-of-fold set to the blind set's prevalence.

    Without this the MCC-optimal cut for CYP2D6 is chosen against a positive rate three
    times the one the submission will actually meet.
    """
    p_train = float(np.mean(y))
    if p_train <= 0 or p_train >= 1:
        return np.ones_like(y, dtype=float)
    return np.where(y == 1, p_test / p_train, (1 - p_test) / (1 - p_train))


def sweep_positive_rate(y: np.ndarray, score: np.ndarray, p_test: float, cfg: Config) -> pd.DataFrame:
    """Prevalence-corrected MCC across predicted-positive rates.

    Sweeping the *rate* rather than a raw score cut is what transfers to the test set:
    a full-data ensemble is sharper than the cross-validation models, so its score
    distribution shifts even when its ranking does not.
    """
    w = prevalence_weights(y, p_test)
    rows = []
    for q in cfg.pos_rate_grid:
        cut = float(np.quantile(score, 1 - q))
        pred = (score >= cut).astype(int)
        tp = float((w * (pred == 1) * (y == 1)).sum())
        pp = float((w * (pred == 1)).sum())
        ap = float((w * (y == 1)).sum())
        rows.append(
            dict(
                pos_rate=float(q),
                score_cut=cut,
                mcc=weighted_mcc(y, pred, w),
                precision=tp / pp if pp > 0 else 0.0,
                recall=tp / ap if ap > 0 else 0.0,
            )
        )
    return pd.DataFrame(rows)


def best_mcc(y: np.ndarray, score: np.ndarray, p_test: float, cfg: Config) -> tuple[float, float]:
    """Best achievable prevalence-corrected MCC over the rate grid.

    Optimistic by construction — the rate is chosen on the same data it is scored on —
    so it is reported for transparency but is only the *selection* criterion in
    ``--operating-point argmax`` mode. Returns ``(mcc, pos_rate)``.
    """
    if len(np.unique(y)) < 2:
        return 0.0, float(p_test)
    sweep = sweep_positive_rate(y, score, p_test, cfg)
    row = sweep.loc[sweep["mcc"].idxmax()]
    return float(row["mcc"]), float(row["pos_rate"])


def mcc_at_rule(y: np.ndarray, score: np.ndarray, p_test: float, cfg: Config) -> tuple[float, float]:
    """MCC under the configured rate rule — the one selection criterion in this script.

    The XGBoost tree count, the Optuna objective, the blend weight and the final arm
    choice all go through here, so a model is never selected under one definition of
    "best MCC" and then thresholded under another. Returns ``(mcc, pos_rate)``.

    Under ``theory`` the rate comes from the measured AUC rather than from the curve's
    own argmax, which also removes the selection bias in :func:`best_mcc` — the rate is
    not fitted to the same sample that scores it.
    """
    if len(np.unique(y)) < 2:
        return 0.0, float(p_test)
    if cfg.operating_point == "theory":
        auc = float(roc_auc_score(y, score))
        rate = max(theoretical_optimal_rate(auc, p_test), cfg.min_positive_rate)
        cut = float(np.quantile(score, 1 - rate))
        pred = (score >= cut).astype(int)
        return weighted_mcc(y, pred, prevalence_weights(y, p_test)), rate
    return best_mcc(y, score, p_test, cfg)


def theoretical_optimal_rate(auc: float, prevalence: float) -> float:
    """MCC-optimal predicted-positive rate for a bi-normal ranker of this AUC.

    Model the scores as ``N(d, 1)`` for positives and ``N(0, 1)`` for negatives, where
    ``AUC = Phi(d / sqrt(2))``, and read off the call rate that maximises MCC at the
    given prevalence. Smooth in AUC, so unlike an empirical argmax it cannot be
    captured by a sampling artefact — at AUC 0.85 and 7% prevalence it returns 0.08,
    near the prevalence, which is where every strong CYP2D6 entry sits; at AUC 0.66 it
    returns 0.25.
    """
    from scipy.stats import norm

    d = np.sqrt(2) * norm.ppf(np.clip(auc, 0.5001, 0.9999))
    best_q, best = prevalence, -np.inf
    for t in np.linspace(-6.0, 8.0, 2001):
        tp = prevalence * (1 - norm.cdf(t - d))
        fn = prevalence - tp
        fp = (1 - prevalence) * (1 - norm.cdf(t))
        tn = (1 - prevalence) - fp
        den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        if den <= 0:
            continue
        mcc = (tp * tn - fp * fn) / den
        if mcc > best:
            best, best_q = mcc, tp + fp
    return float(best_q)


def choose_operating_point(
    y: np.ndarray, score: np.ndarray, p_test: float, cfg: Config, seed: int = 0
) -> dict:
    """Pick the predicted-positive rate the submission will be cut at.

    Both modes maximise MCC; they differ in how much they trust a noisy curve.

    ``--operating-point theory`` (default) reads the rate off
    :func:`theoretical_optimal_rate` at the measured AUC — the MCC optimum of a
    bi-normal ranker of that quality at that prevalence. Smooth in AUC, so it cannot be
    captured by a sampling artefact.

    ``--operating-point argmax`` maximises MCC directly on a **bootstrap-averaged**
    rate curve. It is the literal empirical optimum, and it is the right choice when
    the curve has a clear peak. It is not the default because on CYP2D6 the curve is
    bimodal and flat: a smoke run here put the argmax at the grid floor, i.e. 8 calls
    on 750 compounds against ~53 true positives. The bootstrap IQR of the per-resample
    argmax is always reported — a wide IQR is the tell that the curve carries no
    reliable local structure and that the theory rate should be preferred.
    """
    w = prevalence_weights(y, p_test)
    rng = np.random.default_rng(seed)
    curves, argmaxes = [], []
    for _ in range(cfg.n_bootstrap):
        idx = rng.integers(0, len(y), len(y))
        yb, sb, wb = y[idx], score[idx], w[idx]
        curve = np.array(
            [
                weighted_mcc(yb, (sb >= np.quantile(sb, 1 - q)).astype(int), wb)
                for q in cfg.pos_rate_grid
            ]
        )
        curves.append(curve)
        argmaxes.append(cfg.pos_rate_grid[int(np.argmax(curve))])
    mean_curve = np.mean(curves, axis=0)
    iqr = (float(np.percentile(argmaxes, 25)), float(np.percentile(argmaxes, 75)))

    auc = float(roc_auc_score(y, score)) if len(np.unique(y)) > 1 else float("nan")
    theory_rate = theoretical_optimal_rate(auc, p_test) if np.isfinite(auc) else p_test

    if cfg.operating_point == "theory":
        rate = theory_rate
    else:
        rate = float(cfg.pos_rate_grid[int(np.argmax(mean_curve))])
    rate = max(rate, cfg.min_positive_rate)

    at = int(np.argmin(np.abs(cfg.pos_rate_grid - rate)))
    sweep = sweep_positive_rate(y, score, p_test, cfg)
    row = sweep.iloc[at]
    return dict(
        pos_rate=float(cfg.pos_rate_grid[at]),
        expected_mcc=float(mean_curve[at]),
        point_mcc=float(row["mcc"]),
        precision=float(row["precision"]),
        recall=float(row["recall"]),
        auc=auc,
        theory_rate=float(theory_rate),
        argmax_iqr_low=iqr[0],
        argmax_iqr_high=iqr[1],
        sweep=sweep,
    )


def rank_pct(v: np.ndarray) -> np.ndarray:
    """Percentile ranks in [0, 1]. Blending happens on ranks, never on raw scores:
    a D-MPNN probability and an XGBoost probability are not on the same scale."""
    return pd.Series(v).rank(pct=True).to_numpy()


# ══════════════════════════════════════════════════════════════════════════════
# XGBoost arm
# ══════════════════════════════════════════════════════════════════════════════

def _import_xgboost():
    try:
        import xgboost as xgb
    except ImportError as exc:                                   # pragma: no cover
        raise SystemExit(
            "xgboost is not installed in this environment.\n"
            "    conda activate chemprop && pip install xgboost optuna"
        ) from exc
    return xgb


def _xgb_runtime_params(cfg: Config, n_estimators: int) -> dict:
    """Tree params plus device placement.

    ``device`` only exists from XGBoost 2.0; on 1.x the GPU is selected through
    ``tree_method``, so the version check is not cosmetic.
    """
    xgb = _import_xgboost()
    params = dict(cfg.xgb_params, n_estimators=int(n_estimators))
    try:
        import torch

        gpu = bool(torch.cuda.is_available())
    except Exception:
        gpu = False
    major = int(str(xgb.__version__).split(".")[0])
    if major >= 2:
        params["device"] = "cuda" if gpu else "cpu"
    elif gpu:
        params["tree_method"] = "gpu_hist"
    return params


def xgb_fit_predict(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_pr: np.ndarray,
    n_estimators: int,
    cfg: Config,
    objective: str = "binary",
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    """Fit a seed-bagged XGBoost and return the mean prediction on ``X_pr``.

    Bagging over seeds costs |seeds|x the time and removes a few points of run-to-run
    jitter, which matters when tree counts are being compared at the third decimal.
    """
    xgb = _import_xgboost()
    params = _xgb_runtime_params(cfg, n_estimators)
    preds = []
    for seed in cfg.xgb_seeds:
        if objective == "binary":
            model = xgb.XGBClassifier(objective="binary:logistic", random_state=seed, **params)
            model.fit(X_tr, y_tr, sample_weight=sample_weight)
            preds.append(model.predict_proba(X_pr)[:, 1])
        else:
            model = xgb.XGBRegressor(objective="reg:squarederror", random_state=seed, **params)
            model.fit(X_tr, y_tr, sample_weight=sample_weight)
            preds.append(model.predict(X_pr))
    return np.mean(preds, axis=0)


def xgb_oof_and_test(
    X: np.ndarray,
    y: np.ndarray,
    folds: np.ndarray,
    X_test: np.ndarray,
    n_estimators: int,
    cfg: Config,
    objective: str = "binary",
    fit_test: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold predictions over labelled rows, plus a full-data test prediction.

    Rows whose target is NaN are excluded from training but still receive an
    out-of-fold prediction, so the result can be used as a dense stacking feature.
    """
    labelled = np.isfinite(y)
    oof = np.full(len(y), np.nan)
    for f in range(cfg.n_folds):
        tr = (folds != f) & labelled
        pr = folds == f
        if tr.sum() < 30 or pr.sum() == 0:
            continue
        oof[pr] = xgb_fit_predict(X[tr], y[tr], X[pr], n_estimators, cfg, objective)
    test = (
        xgb_fit_predict(X[labelled], y[labelled], X_test, n_estimators, cfg, objective)
        if fit_test and labelled.sum() >= 30
        else np.full(len(X_test), np.nan)
    )
    return oof, test


def build_stack_features(
    df: pd.DataFrame, X: np.ndarray, X_test: np.ndarray, folds: np.ndarray, cfg: Config
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Out-of-fold predictions of every auxiliary surface, for all four isoforms.

    Three families, each on all of ``ALL_ISOFORMS``:

    * ``shift``    — the floored TDI shift. CYP1A2 and CYP2C9 are not scored, but their
      shifts measure the same bioactivation propensity on ~2,700 extra compounds.
    * ``tdicond`` / ``direct`` — the two raw pIC50 surfaces. Potency is confounded with
      TDI (a compound must inhibit before a shift is measurable), so the classifier
      benefits from being told the potency it should be conditioning on.
    * ``fitted``   — whether a direct curve was fitted at all. Dense, never masked, and
      for CYP3A4 it *is* part of the label.

    Standard (non-nested) stacking: fold k's features come from models that never saw
    fold k, but folds != k were produced by models that did. The resulting optimism is
    small relative to the effects being compared and is identical for every arm, so the
    arm ranking stands.
    """
    print("\nStacking features (out-of-fold auxiliary surfaces)")
    cols_oof, cols_test, names = [], [], []
    n = cfg.stack_n_estimators

    for iso in ALL_ISOFORMS:
        targets = {
            f"{iso}_shift": floored_shift(df, iso).to_numpy(dtype=float),
            f"{iso}_tdicond": df[f"{iso}_pIC50_TDI_condition"].to_numpy(dtype=float),
            f"{iso}_direct": df[f"{iso}_pIC50_direct_inhibition"].to_numpy(dtype=float),
        }
        for name, y in targets.items():
            n_lab = int(np.isfinite(y).sum())
            if n_lab < 200:
                print(f"  {name:22s} skipped ({n_lab} labels)")
                continue
            oof, test = xgb_oof_and_test(X, y, folds, X_test, n, cfg, "regression", cfg.fit_test)
            cols_oof.append(oof)
            cols_test.append(test)
            names.append(f"stack_{name}")
            m = np.isfinite(y) & np.isfinite(oof)
            rho = np.corrcoef(y[m], oof[m])[0, 1] if m.sum() > 10 else np.nan
            print(f"  {name:22s} n={n_lab:5d}  OOF r={rho:.3f}")

        fitted = df[f"{iso}_pIC50_direct_inhibition"].notna().astype(float).to_numpy()
        oof, test = xgb_oof_and_test(X, fitted, folds, X_test, n, cfg, "binary", cfg.fit_test)
        cols_oof.append(oof)
        cols_test.append(test)
        names.append(f"stack_{iso}_fitted")
        print(f"  {iso}_fitted{'':13s} n={len(fitted):5d}  OOF AUC={roc_auc_score(fitted, oof):.3f}")

    return np.column_stack(cols_oof), np.column_stack(cols_test), names


def tune_xgb_trees(
    X: np.ndarray,
    y_masked: np.ndarray,
    labelled: np.ndarray,
    folds: np.ndarray,
    X_test: np.ndarray,
    p_test: float,
    cfg: Config,
) -> tuple[int, np.ndarray, np.ndarray, pd.DataFrame]:
    """Sweep the number of boosting rounds, select on out-of-fold MCC.

    The only XGBoost hyperparameter varied — at fixed learning rate the tree count *is*
    the capacity/overfitting dial, and on a 1.5-6k-row scaffold-split problem it is the
    one that moves the score. Up to four values (``--xgb-trees``).

    Returns ``(best_n_estimators, oof, test, sweep_table)``.
    """
    y_true = y_masked[labelled].astype(int)
    rows, best = [], None
    for n_est in cfg.xgb_n_estimators_grid:
        t0 = time.time()
        oof, test = xgb_oof_and_test(X, y_masked, folds, X_test, n_est, cfg, "binary", cfg.fit_test)
        auc = float(roc_auc_score(y_true, oof[labelled]))
        mcc, rate = mcc_at_rule(y_true, oof[labelled], p_test, cfg)
        mcc_argmax, _ = best_mcc(y_true, oof[labelled], p_test, cfg)
        rows.append(dict(n_estimators=int(n_est), auc=auc, mcc=mcc, pos_rate=rate,
                         mcc_argmax=mcc_argmax, seconds=round(time.time() - t0, 1)))
        print(f"  trees={n_est:5d}  AUC={auc:.4f}  MCC={mcc:.4f} @ rate={rate:.3f}  "
              f"({time.time() - t0:.0f}s)")
        if best is None or mcc > best[0]:
            best = (mcc, int(n_est), oof, test)
    print(f"  --> {best[1]} trees (OOF MCC {best[0]:.4f})")
    return best[1], best[2], best[3], pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# ChemProp arm
# ══════════════════════════════════════════════════════════════════════════════

def _import_chemprop():
    try:
        import torch
        from lightning import pytorch as pl
        from lightning.pytorch.callbacks import EarlyStopping

        from chemprop import data, featurizers, models, nn
    except ImportError as exc:                                   # pragma: no cover
        raise SystemExit(
            "chemprop (>=2.1) is not installed in this environment.\n"
            "    conda activate chemprop"
        ) from exc
    torch.set_float32_matmul_precision("medium")
    # A full run is ~1,500 Trainer.fit calls; Lightning's per-fit GPU/TPU banner would
    # bury the trial log under tens of thousands of lines.
    import logging

    for name in (
        "lightning.pytorch.utilities.rank_zero",
        "lightning.pytorch.accelerators.cuda",
        "lightning.pytorch.trainer.connectors.data_connector",
        "chemprop",
    ):
        logging.getLogger(name).setLevel(logging.WARNING)
    return torch, pl, EarlyStopping, data, featurizers, models, nn


def chemprop_targets(df: pd.DataFrame, iso: str, aux_weight: float) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Target matrix for one isoform's model: its ``is_TDI`` head plus auxiliaries.

    "One model per isoform" refers to the *scored* head — head 0 is the only one whose
    predictions leave this function's model. The auxiliary heads exist because the
    primary label is sparse (1,497 rows for CYP2D6) while the pIC50 surfaces that
    define it are dense across all four isoforms, and NaN targets are masked out of the
    loss, so each compound contributes only to the heads it actually has.

    ``<ISO>_direct_fitted`` is dense and never masked. It is what lets the encoder
    separate "too potent to fit" from "potent" — mandatory for CYP3A4, where 1,249
    compounds with no direct fit are all labelled negative. Do not drop it without
    re-measuring CYP3A4.
    """
    cols = [df[f"{iso}_is_TDI"].astype(float).to_numpy()]
    names = [f"{iso}_is_TDI"]

    if aux_weight > 0:
        for other in ALL_ISOFORMS:
            v = df[f"{other}_pIC50_TDI_condition"]
            cols.append(np.where(v.isna(), np.nan, (v > ACTIVITY_CUT).astype(float)))
            names.append(f"{other}_active_tdi")
        for other in ALL_ISOFORMS:
            v = df[f"{other}_pIC50_direct_inhibition"]
            cols.append(np.where(v.isna(), np.nan, (v > ACTIVITY_CUT).astype(float)))
            names.append(f"{other}_active_direct")
        for other in TDI_ISOFORMS:
            cols.append(df[f"{other}_pIC50_direct_inhibition"].notna().astype(float).to_numpy())
            names.append(f"{other}_direct_fitted")

    Y = np.column_stack(cols)
    weights = np.array([1.0] + [aux_weight] * (len(names) - 1))
    return Y, names, weights


def set_seed(seed: int) -> None:
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _best_state_tracker(pl):
    """Callback keeping the weights from the best val_loss epoch — an early-stopped fit
    otherwise ends ``patience`` epochs past its best."""

    class BestStateTracker(pl.Callback):
        def __init__(self) -> None:
            self.best = float("inf")
            self.state = None

        def on_validation_epoch_end(self, trainer, module) -> None:
            val = trainer.callback_metrics.get("val_loss")
            if val is None:
                return
            val = float(val)
            if val < self.best:
                self.best = val
                self.state = {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}

        def restore(self, module) -> None:
            if self.state is not None:
                module.load_state_dict(self.state)

    return BestStateTracker()


#: ChemProp search space, mirroring the PXR QSARtuna optimisation
#: (``pxr_chemprop_optimize.json``) plus the two TDI-specific auxiliary-head knobs.
def suggest_chemprop_params(trial) -> dict:
    params = dict(
        depth=trial.suggest_int("depth", 2, 5),
        message_hidden_dim=trial.suggest_categorical("message_hidden_dim", [300, 600]),
        ffn_hidden_dim=trial.suggest_categorical("ffn_hidden_dim", [300, 600, 1300]),
        ffn_n_layers=trial.suggest_int("ffn_n_layers", 1, 3),
        dropout=trial.suggest_float("dropout", 0.0, 0.35, step=0.05),
        max_lr=trial.suggest_float("max_lr", 1e-4, 1e-3, log=True),
        init_lr_ratio=trial.suggest_float("init_lr_ratio", 0.01, 0.5, log=True),
        final_lr_ratio=trial.suggest_float("final_lr_ratio", 0.001, 0.5, log=True),
        warmup_epochs=trial.suggest_int("warmup_epochs", 2, 8),
        aggregation=trial.suggest_categorical("aggregation", ["mean", "sum", "norm"]),
        batch_norm=trial.suggest_categorical("batch_norm", [True, False]),
        aux_weight=trial.suggest_categorical("aux_weight", [0.0, 0.15, 0.25, 0.5]),
    )
    if params["aggregation"] == "norm":
        params["aggregation_norm"] = trial.suggest_int("aggregation_norm", 50, 200)
    return params


DEFAULT_CHEMPROP_PARAMS = dict(
    depth=4,
    message_hidden_dim=300,
    ffn_hidden_dim=600,
    ffn_n_layers=2,
    dropout=0.15,
    max_lr=2e-4,
    init_lr_ratio=0.5,
    final_lr_ratio=0.05,
    warmup_epochs=2,
    aggregation="mean",
    batch_norm=True,
    aux_weight=0.25,
)


def build_mpnn(n_descriptors: int, task_weights: np.ndarray, params: dict):
    torch, pl, _EarlyStopping, _data, featurizers, models, nn = _import_chemprop()

    feat = featurizers.SimpleMoleculeMolGraphFeaturizer()
    mp = nn.BondMessagePassing(
        d_v=feat.atom_fdim,
        d_e=feat.bond_fdim,
        depth=params["depth"],
        d_h=params["message_hidden_dim"],
        dropout=params["dropout"],
    )
    if params["aggregation"] == "sum":
        agg = nn.SumAggregation()
    elif params["aggregation"] == "norm":
        agg = nn.NormAggregation(norm=float(params.get("aggregation_norm", 100)))
    else:
        agg = nn.MeanAggregation()

    ffn = nn.BinaryClassificationFFN(
        n_tasks=len(task_weights),
        input_dim=mp.output_dim + n_descriptors,
        hidden_dim=params["ffn_hidden_dim"],
        n_layers=params["ffn_n_layers"],
        dropout=params["dropout"],
        criterion=nn.metrics.BCELoss(task_weights=torch.tensor(task_weights, dtype=torch.float)),
    )
    max_lr = float(params["max_lr"])
    return models.MPNN(
        mp,
        agg,
        ffn,
        batch_norm=bool(params["batch_norm"]),
        metrics=[nn.metrics.BinaryAUROC()],
        warmup_epochs=int(params["warmup_epochs"]),
        init_lr=max_lr * float(params["init_lr_ratio"]),
        max_lr=max_lr,
        final_lr=max_lr * float(params["final_lr_ratio"]),
    )


def make_dataset(mols, Y, x_d):
    _torch, _pl, _ES, data, featurizers, _models, _nn = _import_chemprop()
    feat = featurizers.SimpleMoleculeMolGraphFeaturizer()
    points = [
        data.MoleculeDatapoint(
            mol=mol,
            y=Y[i].astype(float),
            x_d=None if x_d is None else x_d[i].astype(float),
        )
        for i, mol in enumerate(mols)
    ]
    return data.MoleculeDataset(points, feat)


def fit_mpnn(train_dset, val_dset, n_desc, task_weights, params, seed, cfg, max_epochs, patience):
    """Fit one multitask binary MPNN.

    There is deliberately no ``normalize_targets`` call: the targets are already 0/1
    and BCE needs them that way. That is the one thing you cannot copy from the
    regression scripts.
    """
    _torch, pl, EarlyStopping, data, _feat, _models, _nn = _import_chemprop()
    set_seed(seed)

    train_loader = data.build_dataloader(
        train_dset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, seed=seed
    )
    callbacks, val_loader, tracker = [], None, None
    if val_dset is not None:
        val_loader = data.build_dataloader(
            val_dset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=False
        )
        tracker = _best_state_tracker(pl)
        callbacks = [EarlyStopping(monitor="val_loss", patience=patience, mode="min"), tracker]

    mpnn = build_mpnn(n_desc, task_weights, params)
    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=cfg.progress,
        enable_model_summary=False,
        accelerator="auto",
        devices=1,
        max_epochs=max_epochs,
        callbacks=callbacks,
    )
    trainer.fit(mpnn, train_loader, val_loader)
    if tracker is not None:
        tracker.restore(mpnn)
    return mpnn, trainer


def predict_proba(mpnn, trainer, dset, cfg: Config) -> np.ndarray:
    """Positive-class probabilities. ``BinaryClassificationFFN`` applies the sigmoid in
    eval mode, so these come back already in [0, 1]."""
    import torch

    _torch, _pl, _ES, data, _feat, _models, _nn = _import_chemprop()
    loader = data.build_dataloader(
        dset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, shuffle=False
    )
    mpnn.eval()
    raw = trainer.predict(mpnn, loader)
    return torch.cat(raw).numpy().reshape(len(dset), -1)


def chemprop_cv(
    ctx: "Context",
    iso: str,
    params: dict,
    cfg: Config,
    folds_to_run: list[int],
    max_epochs: int,
    patience: int,
    on_fold=None,
) -> np.ndarray:
    """Cross-validated out-of-fold probabilities for the primary head.

    ``folds_to_run`` lets the Optuna objective score a subset of the folds; rows in
    folds that were not run stay NaN. ``on_fold(fold, oof)`` is called after each fold
    so a trial can report an intermediate value and be pruned.
    """
    Y, _names, task_weights = chemprop_targets(ctx.train_df, iso, params.get("aux_weight", 0.25))
    keep = np.isfinite(Y).any(axis=1)
    oof = np.full(len(ctx.train_mols), np.nan)

    for f in folds_to_run:
        tr = (ctx.folds != f) & keep
        # ``keep`` on the validation side too: a compound with no finite target anywhere
        # contributes an all-masked row, and a batch of those makes val_loss NaN, which
        # would silently break early stopping.
        va = (ctx.folds == f) & keep
        train_dset = make_dataset(
            [m for m, k in zip(ctx.train_mols, tr) if k], Y[tr],
            None if ctx.x_d_train is None else ctx.x_d_train[tr],
        )
        val_dset = make_dataset(
            [m for m, k in zip(ctx.train_mols, va) if k], Y[va],
            None if ctx.x_d_train is None else ctx.x_d_train[va],
        )
        mpnn, trainer = fit_mpnn(
            train_dset, val_dset, ctx.n_desc, task_weights, params,
            seed=cfg.scaffold_seed, cfg=cfg, max_epochs=max_epochs, patience=patience,
        )
        oof[va] = predict_proba(mpnn, trainer, val_dset, cfg)[:, 0]
        if on_fold is not None:
            on_fold(f, oof)
    return oof


def chemprop_hpo(ctx: "Context", iso: str, p_test: float, cfg: Config) -> tuple[dict, pd.DataFrame]:
    """Optuna TPE search over the ChemProp space, scored on out-of-fold MCC.

    The objective is the metric the track is scored on, not AUC — specifically
    :func:`mcc_at_rule`, the same criterion that later picks the tree count, the blend
    weight and the threshold. MCC is noisier than AUC, so both are recorded per trial in
    ``chemprop_hpo_trials.csv``: if the top trials disagree wildly on MCC while their
    AUCs are flat, the search has found noise and you should prefer the AUC ordering by
    hand.

    Trials report after every fold and are pruned against the running median, which is
    what makes 20 trials affordable: a bad trial dies after one fold.
    """
    try:
        import optuna
    except ImportError as exc:                                   # pragma: no cover
        raise SystemExit(
            "optuna is not installed in this environment.\n"
            "    conda activate chemprop && pip install xgboost optuna"
        ) from exc

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    y_full = ctx.train_df[f"{iso}_is_TDI"]
    labelled = y_full.notna().to_numpy()
    y_true = y_full[labelled].astype(int).to_numpy()
    folds_to_run = list(range(min(cfg.hpo_folds, cfg.n_folds)))
    hpo_mask = labelled & np.isin(ctx.folds, folds_to_run)
    y_hpo = y_full[hpo_mask].astype(int).to_numpy()

    def objective(trial) -> float:
        params = suggest_chemprop_params(trial)

        def on_fold(f, oof):
            done = hpo_mask & np.isfinite(oof)
            if done.sum() < 50 or len(np.unique(y_full[done].astype(int))) < 2:
                return
            partial, _ = mcc_at_rule(
                y_full[done].astype(int).to_numpy(), oof[done], p_test, cfg
            )
            trial.report(partial, step=f)
            if trial.should_prune():
                raise optuna.TrialPruned()

        oof = chemprop_cv(
            ctx, iso, params, cfg, folds_to_run,
            max_epochs=cfg.hpo_max_epochs, patience=cfg.hpo_patience, on_fold=on_fold,
        )
        m = hpo_mask & np.isfinite(oof)
        mcc, rate = mcc_at_rule(y_full[m].astype(int).to_numpy(), oof[m], p_test, cfg)
        auc = (
            float(roc_auc_score(y_full[m].astype(int), oof[m]))
            if len(np.unique(y_full[m].astype(int))) > 1
            else float("nan")
        )
        trial.set_user_attr("auc", auc)
        trial.set_user_attr("pos_rate", rate)
        trial.set_user_attr("n_scored", int(m.sum()))
        print(f"    trial {trial.number:3d}  MCC={mcc:.4f}  AUC={auc:.4f}  {params}")
        return mcc

    sampler = optuna.samplers.TPESampler(seed=cfg.optuna_seed, n_startup_trials=5)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    print(f"\n  Optuna: {cfg.n_trials} trials x {len(folds_to_run)} folds "
          f"({int(hpo_mask.sum())} of {len(y_true)} labels scored per trial)")
    study.optimize(objective, n_trials=cfg.n_trials, gc_after_trial=True)

    rows = []
    for t in study.trials:
        rows.append(
            dict(
                number=t.number,
                state=str(t.state).split(".")[-1],
                mcc=t.value if t.value is not None else np.nan,
                auc=t.user_attrs.get("auc", np.nan),
                pos_rate=t.user_attrs.get("pos_rate", np.nan),
                **t.params,
            )
        )
    trials = pd.DataFrame(rows)

    best = dict(DEFAULT_CHEMPROP_PARAMS)
    best.update(study.best_params)
    print(f"  --> best trial {study.best_trial.number}: MCC={study.best_value:.4f}")
    print(f"      {best}")
    return best, trials


# ══════════════════════════════════════════════════════════════════════════════
# Shared context
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Context:
    """Everything both arms share: molecules, features, folds. Built once, reused by
    every isoform, so the two isoform models are trained on identical splits."""

    train_df: pd.DataFrame
    test_df: pd.DataFrame
    train_mols: list
    test_mols: list
    folds: np.ndarray
    X: np.ndarray                 # tabular features for XGBoost (stacked if enabled)
    X_test: np.ndarray
    feature_names: list[str]
    x_d_train: np.ndarray | None  # scaled RDKit block for ChemProp
    x_d_test: np.ndarray | None
    n_desc: int


def _feature_cache_path(cfg: Config, smiles: list[str], tag: str) -> Path:
    import hashlib

    h = hashlib.md5(("|".join(smiles)).encode()).hexdigest()[:12]
    cache = cfg.out_dir / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    return cache / f"{tag}_{h}.npz"


def _featurize_cached(cfg: Config, mols: list, smiles: list[str], tag: str) -> tuple[np.ndarray, list[str]]:
    path = _feature_cache_path(cfg, smiles, tag)
    if cfg.cache_features and path.exists():
        blob = np.load(path, allow_pickle=True)
        print(f"  {tag}: features from cache {path.name}")
        return blob["X"], list(blob["names"])
    X, names = featurize_tabular(mols)
    if cfg.cache_features:
        np.savez_compressed(path, X=X, names=np.array(names, dtype=object))
    return X, names


def prepare_context(cfg: Config) -> Context:
    """Load, standardise, featurise and split once for both isoforms and both arms."""
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    print(BANNER)
    print("Data")
    df = read_table(cfg, FILE_TRAIN_TDI)
    test_df = read_table(cfg, FILE_TEST)
    print(f"  train {df.shape}   test {test_df.shape}")
    verify_label_rule(df)

    print("\nStructures")
    mols, keep = build_mols(df["SMILES"].tolist())
    if (~keep).any():
        print(f"  dropped {int((~keep).sum())} unparseable training SMILES")
        df = df[keep].reset_index(drop=True)
    test_mols, test_keep = build_mols(test_df["SMILES"].tolist())
    if (~test_keep).any():
        raise ValueError(f"{int((~test_keep).sum())} test SMILES failed to parse — cannot submit")
    test_df = test_df[test_keep].reset_index(drop=True)

    print("\nFeatures (tabular, for XGBoost)")
    X_all, names = _featurize_cached(cfg, mols, df["SMILES"].tolist(), "train")
    X_test_all, _ = _featurize_cached(cfg, test_mols, test_df["SMILES"].tolist(), "test")
    keep_cols = clean_columns(X_all, cfg)
    X = np.nan_to_num(X_all[:, keep_cols], nan=0.0)
    X_test = np.nan_to_num(X_test_all[:, keep_cols], nan=0.0)
    names = [n for n, k in zip(names, keep_cols) if k]

    if cfg.use_descriptors:
        print("\nFeatures (RDKit 2D block, for ChemProp)")
        d_train = descriptor_block(mols)
        d_test = descriptor_block(test_mols)
        d_keep = clean_columns(d_train, cfg)
        d_train, d_test = d_train[:, d_keep], d_test[:, d_keep]
        scaler = StandardScaler().fit(d_train)
        x_d_train = scale_descriptors(scaler, d_train, cfg)
        x_d_test = scale_descriptors(scaler, d_test, cfg)
        n_desc = x_d_train.shape[1]
    else:
        x_d_train = x_d_test = None
        n_desc = 0

    print("\nFolds")
    folds = murcko_scaffold_folds(mols, cfg.n_folds, cfg.scaffold_seed)

    ctx = Context(
        train_df=df,
        test_df=test_df,
        train_mols=mols,
        test_mols=test_mols,
        folds=folds,
        X=X,
        X_test=X_test,
        feature_names=names,
        x_d_train=x_d_train,
        x_d_test=x_d_test,
        n_desc=n_desc,
    )

    if cfg.use_stack_features:
        S, S_test, stack_names = build_stack_features(df, X, X_test, folds, cfg)
        ctx.X = np.hstack([X, np.nan_to_num(S, nan=0.0)])
        ctx.X_test = np.hstack([X_test, np.nan_to_num(S_test, nan=0.0)])
        ctx.feature_names = names + stack_names
        print(f"  XGBoost feature matrix: {ctx.X.shape}")

    return ctx


# ══════════════════════════════════════════════════════════════════════════════
# One isoform, end to end
# ══════════════════════════════════════════════════════════════════════════════

def run_isoform(ctx: Context, iso: str, cfg: Config) -> dict:
    """Tune, cross-validate, blend and threshold a single isoform.

    Returns a dict with the OOF frame, the test scores, the chosen hyperparameters and
    the operating point; everything is also written under ``<out_dir>/<iso>/``.
    """
    print(f"\n{BANNER}\n{iso}\n{BANNER}")
    out = cfg.iso_dir(iso)

    lab = ctx.train_df[f"{iso}_is_TDI"]
    labelled = lab.notna().to_numpy()
    y = lab.fillna(0).astype(int).to_numpy()
    y_true = y[labelled]
    y_masked = np.where(labelled, y.astype(float), np.nan)
    p_test = cfg.prevalence_for(iso, y_true)
    print(f"  {int(labelled.sum())} labels | train prevalence {y_true.mean():.4f} | "
          f"MCC evaluated at prevalence {p_test:.4f} ({cfg.prevalence})")

    # ── Arm B: XGBoost, swept over the number of trees ────────────────────────
    print(f"\n  XGBoost — sweeping {len(cfg.xgb_n_estimators_grid)} tree counts "
          f"({cfg.xgb_n_estimators_grid}) on {ctx.X.shape[1]} features")
    best_trees, xgb_oof, xgb_test, trees_sweep = tune_xgb_trees(
        ctx.X, y_masked, labelled, ctx.folds, ctx.X_test, p_test, cfg
    )
    trees_sweep.to_csv(out / "xgb_trees_sweep.csv", index=False)

    # ── Arm A: ChemProp, Optuna-tuned then cross-validated on all folds ───────
    print("\n  ChemProp — hyperparameter search")
    t0 = time.time()
    best_params, trials = chemprop_hpo(ctx, iso, p_test, cfg)
    trials.to_csv(out / "chemprop_hpo_trials.csv", index=False)
    print(f"  search took {(time.time() - t0) / 60:.1f} min")

    print(f"\n  ChemProp — {cfg.n_folds}-fold CV at the chosen hyperparameters")
    cp_oof = chemprop_cv(
        ctx, iso, best_params, cfg, list(range(cfg.n_folds)),
        max_epochs=cfg.max_epochs, patience=cfg.patience,
    )

    cp_test = np.full(len(ctx.test_mols), np.nan)
    if cfg.fit_test:
        print(f"  ChemProp — final ensemble on all data, seeds {cfg.chemprop_seeds}")
        Y, _names, task_weights = chemprop_targets(
            ctx.train_df, iso, best_params.get("aux_weight", 0.25)
        )
        keep = np.isfinite(Y).any(axis=1)
        full_dset = make_dataset(
            [m for m, k in zip(ctx.train_mols, keep) if k], Y[keep],
            None if ctx.x_d_train is None else ctx.x_d_train[keep],
        )
        test_dset = make_dataset(
            ctx.test_mols, np.full((len(ctx.test_mols), Y.shape[1]), np.nan), ctx.x_d_test
        )
        per_seed = []
        for seed in cfg.chemprop_seeds:
            mpnn, trainer = fit_mpnn(
                full_dset, None, ctx.n_desc, task_weights, best_params,
                seed=seed, cfg=cfg, max_epochs=cfg.max_epochs, patience=cfg.patience,
            )
            per_seed.append(predict_proba(mpnn, trainer, test_dset, cfg)[:, 0])
            print(f"    seed {seed} done")
        cp_test = np.mean(per_seed, axis=0)

    # ── Blend: rank-average, weight chosen on out-of-fold MCC ─────────────────
    # Ranks, not probabilities: the two arms are calibrated differently, and only the
    # ordering is comparable. w = 1 is pure ChemProp, w = 0 pure XGBoost.
    cp_rank = rank_pct(cp_oof[labelled])
    xgb_rank = rank_pct(xgb_oof[labelled])
    blend_rows = []
    for w in cfg.blend_weight_grid:
        score = w * cp_rank + (1 - w) * xgb_rank
        mcc, rate = mcc_at_rule(y_true, score, p_test, cfg)
        mcc_argmax, _ = best_mcc(y_true, score, p_test, cfg)
        auc = float(roc_auc_score(y_true, score))
        blend_rows.append(
            dict(weight_chemprop=float(w), mcc=mcc, auc=auc, pos_rate=rate, mcc_argmax=mcc_argmax)
        )
    blend = pd.DataFrame(blend_rows)
    blend.to_csv(out / "blend_sweep.csv", index=False)
    best_w = float(blend.loc[blend["mcc"].idxmax(), "weight_chemprop"])
    print(f"\n  Blend weight sweep -> w_chemprop={best_w:.2f} "
          f"(MCC {blend['mcc'].max():.4f}; pure ChemProp {blend.iloc[-1]['mcc']:.4f}, "
          f"pure XGBoost {blend.iloc[0]['mcc']:.4f})")

    # The blended OOF score is built from ranks over the *labelled* rows — the same
    # population the weight was swept on — so the arm comparison below scores exactly
    # the thing that was selected. Test ranks are taken over the test set, which is its
    # own population.
    blend_oof = np.full(len(cp_oof), np.nan)
    blend_oof[labelled] = best_w * cp_rank + (1 - best_w) * xgb_rank
    arms = {
        "chemprop": (cp_oof, cp_test),
        "xgboost": (xgb_oof, xgb_test),
        "blend": (
            blend_oof,
            best_w * rank_pct(cp_test) + (1 - best_w) * rank_pct(xgb_test),
        ),
    }

    # ── Arm comparison and operating point ────────────────────────────────────
    print(f"\n  Arm comparison ({iso}) — MCC at prevalence {p_test:.4f}")
    arm_rows, sweeps, ops = [], [], {}
    for tag, (oof_scores, _test_scores) in arms.items():
        op = choose_operating_point(y_true, oof_scores[labelled], p_test, cfg)
        sweeps.append(op.pop("sweep").assign(isoform=iso, arm=tag))
        ops[tag] = op
        arm_rows.append(dict(isoform=iso, arm=tag, **op))
        print(
            f"    {tag:9s} AUC={op['auc']:.4f}  MCC={op['expected_mcc']:.4f} "
            f"@ rate={op['pos_rate']:.3f}  P={op['precision']:.3f} R={op['recall']:.3f}  "
            f"(theory rate {op['theory_rate']:.3f}, argmax IQR "
            f"[{op['argmax_iqr_low']:.2f},{op['argmax_iqr_high']:.2f}])"
        )
    arm_table = pd.DataFrame(arm_rows)
    arm_table.to_csv(out / "arm_comparison.csv", index=False)
    pd.concat(sweeps).to_csv(out / "threshold_sweep.csv", index=False)

    # The blend is only kept if it actually beats both parents out of fold — an
    # ensemble that loses to one of its arms is not worth the extra variance.
    chosen = str(arm_table.loc[arm_table["expected_mcc"].idxmax(), "arm"])
    print(f"  --> chose arm '{chosen}'")
    op = ops[chosen]

    if op["argmax_iqr_high"] - op["argmax_iqr_low"] > 0.20:
        other = "theory" if cfg.operating_point == "argmax" else "argmax"
        print(
            f"  WARNING {iso}: the bootstrap argmax IQR is wide "
            f"[{op['argmax_iqr_low']:.2f},{op['argmax_iqr_high']:.2f}] — the MCC-vs-rate "
            f"curve carries little local structure, so the call rate is weakly "
            f"determined. Look at threshold_sweep.csv and compare "
            f"--operating-point {other}."
        )

    # ── Out-of-fold and test artefacts ────────────────────────────────────────
    oof_frame = pd.DataFrame(
        {
            "Molecule_Name": ctx.train_df["Molecule_Name"].to_numpy(),
            "fold": ctx.folds,
            f"{iso}_true": np.where(labelled, y, np.nan),
        }
    )
    for tag, (oof_scores, _t) in arms.items():
        oof_frame[f"{iso}_{tag}"] = oof_scores
    oof_frame.to_csv(out / "oof_predictions.csv", index=False)

    calls = None
    if cfg.fit_test:
        test_score = arms[chosen][1]
        cut = float(np.quantile(test_score, 1 - op["pos_rate"]))
        calls = (test_score >= cut).astype(bool)
        test_frame = pd.DataFrame(
            {
                "SMILES": ctx.test_df["SMILES"].to_numpy(),
                "Molecule_Name": ctx.test_df["Molecule_Name"].to_numpy(),
                f"{iso}_chemprop": cp_test,
                f"{iso}_xgboost": xgb_test,
                f"{iso}_blend": arms["blend"][1],
                f"{iso}_score": test_score,
                f"{iso}_is_TDI": calls,
            }
        )
        test_frame.to_csv(out / "test_scores.csv", index=False)
        print(
            f"  test: calling {int(calls.sum())}/{len(calls)} positive ({calls.mean():.4f}) "
            f"against an expected {p_test:.4f} ({int(round(p_test * len(calls)))} compounds)"
        )

    summary = dict(
        isoform=iso,
        n_labels=int(labelled.sum()),
        train_prevalence=float(y_true.mean()),
        eval_prevalence=float(p_test),
        arm=chosen,
        blend_weight_chemprop=best_w,
        xgb_n_estimators=int(best_trees),
        auc=float(op["auc"]),
        mcc=float(op["expected_mcc"]),
        pos_rate=float(op["pos_rate"]),
        precision=float(op["precision"]),
        recall=float(op["recall"]),
        chemprop_params=best_params,
    )
    (out / "best_params.json").write_text(json.dumps(summary, indent=2, default=str))

    return dict(
        summary=summary,
        oof=oof_frame,
        arms=arm_table,
        blend_sweep=blend,
        trees_sweep=trees_sweep,
        hpo_trials=trials,
        calls=calls,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Submission
# ══════════════════════════════════════════════════════════════════════════════

def write_submission(cfg: Config, ctx: Context | None = None) -> Path | None:
    """Assemble the two-column submission from whatever per-isoform runs exist.

    Reads ``<iso>/test_scores.csv`` for both isoforms, so a run that did CYP2D6 today
    and CYP3A4 tomorrow still produces one valid file.
    """
    frames = {}
    for iso in TDI_ISOFORMS:
        path = cfg.out_dir / iso / "test_scores.csv"
        if path.exists():
            frames[iso] = pd.read_csv(path)
        else:
            print(f"  {iso}: no test_scores.csv yet — run --isoform {iso}")
    if not frames:
        return None

    base = next(iter(frames.values()))[["SMILES", "Molecule_Name"]].copy()
    for iso, frame in frames.items():
        base = base.merge(frame[["Molecule_Name", f"{iso}_is_TDI"]], on="Molecule_Name", how="left")
        base[f"{iso}_is_TDI"] = base[f"{iso}_is_TDI"].astype(bool)

    path = cfg.out_dir / "my_ensemble_tdi_submission.csv"
    base.to_csv(path, index=False)
    print(f"\n  submission -> {path}  ({len(base)} rows)")

    if len(frames) < len(TDI_ISOFORMS):
        print("  NOTE: only one isoform present — this file is not yet submittable.")
        return path

    try:
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        from validation.tdi_validation import validate_tdi_submission

        expected = set(ctx.test_df["Molecule_Name"]) if ctx is not None else None
        ok, errors = validate_tdi_submission(path, expected_ids=expected)
        print("  validation:", "PASS" if ok else "FAIL")
        for msg in errors or []:
            print(f"    - {msg}")
    except Exception as exc:                                     # noqa: BLE001
        print(f"  validation skipped ({exc})")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--isoform", choices=[*TDI_ISOFORMS, "both"], default="both")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--folds", type=int, default=None, help="scaffold CV folds")
    p.add_argument("--n-trials", type=int, default=None, help="Optuna trials for ChemProp")
    p.add_argument("--hpo-folds", type=int, default=None, help="folds scored per Optuna trial")
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument(
        "--xgb-trees", type=int, nargs="+", default=None,
        help="up to 4 values for n_estimators (default 200 400 800 1600)",
    )
    p.add_argument("--chemprop-seeds", type=int, nargs="+", default=None)
    p.add_argument("--prevalence", choices=["blind", "train"], default=None)
    p.add_argument(
        "--operating-point", choices=["theory", "argmax"], default=None,
        help="how the MCC-optimal call rate is read off the OOF curve (default: theory)",
    )
    p.add_argument("--no-stack", action="store_true", help="drop the stacked auxiliary features")
    p.add_argument("--no-descriptors", action="store_true", help="ChemProp on the graph only")
    p.add_argument("--no-test", action="store_true", help="out-of-fold only, no submission")
    p.add_argument("--quick", action="store_true", help="tiny run that exercises every path")
    p.add_argument("--progress", action="store_true")
    return p.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> Config:
    cfg = quick_config() if args.quick else Config()
    if args.data_dir:
        cfg.data_dir = args.data_dir
    if args.out_dir:
        cfg.out_dir = args.out_dir
    if args.folds:
        cfg.n_folds = args.folds
    if args.n_trials is not None:
        cfg.n_trials = args.n_trials
    if args.hpo_folds is not None:
        cfg.hpo_folds = args.hpo_folds
    if args.max_epochs is not None:
        cfg.max_epochs = args.max_epochs
    if args.xgb_trees:
        if len(args.xgb_trees) > 4:
            raise SystemExit("--xgb-trees takes at most 4 values")
        cfg.xgb_n_estimators_grid = list(args.xgb_trees)
    if args.chemprop_seeds:
        cfg.chemprop_seeds = list(args.chemprop_seeds)
    if args.prevalence:
        cfg.prevalence = args.prevalence
    if args.operating_point:
        cfg.operating_point = args.operating_point
    if args.no_stack:
        cfg.use_stack_features = False
    if args.no_descriptors:
        cfg.use_descriptors = False
    if args.no_test:
        cfg.fit_test = False
    cfg.progress = args.progress
    cfg.isoforms = list(TDI_ISOFORMS) if args.isoform == "both" else [args.isoform]
    return cfg


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = config_from_args(args)

    t0 = time.time()
    ctx = prepare_context(cfg)

    summaries = []
    for iso in cfg.isoforms:
        result = run_isoform(ctx, iso, cfg)
        summaries.append(result["summary"])

    print(f"\n{BANNER}\nSummary")
    table = pd.DataFrame(summaries).drop(columns=["chemprop_params"])
    print(table.to_string(index=False))
    table.to_csv(cfg.out_dir / "summary.csv", index=False)
    if len(summaries) == len(TDI_ISOFORMS):
        print(f"\n  MA-MCC (out of fold, at the evaluation prevalence) = "
              f"{table['mcc'].mean():.4f}")
        print("  for reference — live board: Cheminfo 0.2279 (#22), "
              "LGBM-baseline 0.2881 (#10), leader 0.3654")

    if cfg.fit_test:
        write_submission(cfg, ctx)
    print(f"\n  total runtime {(time.time() - t0) / 60:.1f} min")
    print(f"  diagnostics -> {cfg.out_dir}/")


if __name__ == "__main__":
    main()
