"""
ChemProp multitask CYP direct-inhibition pIC50 — RDKit 2D descriptors + single-shot
log2fc auxiliary heads + left-censored loss + SVR-proxy model routed by predicted
potency + multi-seed ensemble.

OpenADMET CYP Inhibition Blind Challenge, Direct Inhibition (regression) track.
The design is the recommendation that falls out of the two analysis write-ups in this
repository, implemented end to end:

  * ``Analysis_of_compounds_with_DRC_percent_inhibition_And_Emax.md``
        §1  the single-dose screen is a usable *label* source, never a feature
        §2  half the CYP3A4 pIC50s are "IC50 > 50 uM" statements — left-censored
        §2.3 weight real labels by inverse credible-interval width
        §3  Emax needs no special handling; inverse-CI weighting already covers it
  * ``PXR_Challenge_Single_Shot_Data_Usage.md``
        §1  single-concentration readouts as *auxiliary multitask heads*
        §2  a *separate* proxy-label model, blended in where it predicts weak
        "Suggested next experiment", arms B and C

Two models, and the single-shot data is used in both of its roles.

  MODEL P (primary) — one MPNN, 8 heads:
        4 x <ISO>_pIC50_direct_inhibition   (primary, task weight 1.0)
        4 x <ISO>_log2fc_estimate           (auxiliary, task weight AUX_TASK_WEIGHT)
      The log2fc heads are dense (every screened compound, all four isoforms) where
      the pIC50 heads are sparse, so they carry the shared encoder. Real labels only —
      no pseudo-labels are mixed into this loss. That is deliberate: mixing them in is
      exactly where CYP2D6 blew up in §1.4 of the analysis doc (2,171 pseudo-inactives
      against 1,117 real labels, on the weakest calibration of the four isoforms).

  MODEL X (proxy) — the same architecture, 4 heads, trained *only* on compounds that
      were screened but never carried through to a dose-response curve, labelled with a
      fold-internal single-dose -> pIC50 calibration. This is matcha-croissant's
      "proxy-svr-labels" model, 1-D instead of 2-D because the CYP screen is a single
      point at 49.5 uM rather than four concentrations.

  ROUTING — final prediction per isoform is model P, except where model X predicts a
      weak compound, where the two are blended with a per-isoform weight chosen on
      out-of-fold predictions. The two label sets never share a loss, so the swamping
      failure mode is gone and the blend can shrink to nothing for an isoform where the
      proxy model is not trustworthy (alpha = 0 falls out of the OOF sweep by itself).

Two further things the label analysis asked for, which chemprop supports natively:

  * **Censored loss.** A fitted pIC50 below the top screening concentration
    (49.5 uM = pIC50 4.305) is an extrapolation past the data. Those rows are given
    ``lt_mask=True`` and the loss is ``BoundedMSE``, so predicting *lower* than the
    reported value is free — the "tube" loss §2.3 asks for, and the shape ST-RAE
    itself rewards. Proxy labels below the same threshold are censored the same way,
    so a weak pseudo-label is a soft upper bound rather than a point target.
    Note what that does *not* do: an upper bound still pulls a prediction down, so a
    non-hit whose true pIC50 is above the bound is still mislabelled. Censoring is not
    the CYP2D6 safeguard — see ``build_proxy_labels`` and ``choose_blend_alphas``.
  * **Inverse-CI weighting.** chemprop's per-datapoint ``weight`` is a scalar across
    all tasks, so the per-task inverse-CI weights are averaged over the isoforms a
    compound actually has measured. Coarser than ``cyp_label_utils.ci_weights``, but
    the same idea and it costs nothing.

Scoring is the challenge's own ST-RAE (imported from ``evaluation/`` when this script
sits in the repo, otherwise the identical local copy below), on Bemis-Murcko scaffold
folds — the split that ranks modelling choices honestly even though it flatters nobody.

Data (CSV, from ``LOCAL_DATA_DIR`` if present, else straight from Hugging Face):
    cyp-challenge-TRAIN_inhibition.csv           4,905 compounds, DRC pIC50 + CI + std
    cyp-challenge-single-concentration-TRAIN.csv 17,504 rows = 4,376 compounds x 4 iso
    cyp-challenge-TEST-BLINDED.csv               750 compounds, Molecule_Name + SMILES

Usage:
    conda activate chemprop            # chemprop >= 2.1 (developed against 2.2.x)
    python chemprop_cyp_pic50_rdkit2d_log2fc_aux_proxy_blend_ensemble.py

    # offline VM: download the three CSVs once, then
    export CYP_DATA_DIR=/data/cyp-challenge

Runtime is dominated by the fold loops. On one modern GPU expect roughly:
    grid search   len(PARAM_GRID combos) x N_FOLDS  fits
    OOF stage     N_FOLDS x 2                       fits
    final         len(PRIMARY_SEEDS) + len(PROXY_SEEDS) fits
Set RUN_GRID_SEARCH = False to skip straight to DEFAULT_PARAMS; CV results are cached
to CV_RESULTS_PATH and reloaded if the file already exists, so a killed run resumes.
"""

from __future__ import annotations

import itertools
import os
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightning import pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from chemprop import data, featurizers, models, nn

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore", message=".*does not have many workers.*")
warnings.filterwarnings("ignore", message=".*GPU available but not used.*")
torch.set_float32_matmul_precision("medium")

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

# ── Data ───────────────────────────────────────────────────────────────────────
LOCAL_DATA_DIR = Path(os.environ.get("CYP_DATA_DIR", PROJECT_ROOT / "data"))
HF_PREFIX = "hf://datasets/openadmet/cyp-challenge-train-test"

FILE_TRAIN_DRC = "cyp-challenge-TRAIN_inhibition.csv"
FILE_TRAIN_SINGLE = "cyp-challenge-single-concentration-TRAIN.csv"
FILE_TEST = "cyp-challenge-TEST-BLINDED.csv"

# ── Outputs ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path(os.environ.get("CYP_OUTPUT_DIR", PROJECT_ROOT / "outputs" / "chemprop"))
CV_RESULTS_PATH = OUTPUT_DIR / "cv_grid_results.csv"
CALIBRATION_PATH = OUTPUT_DIR / "single_dose_calibration.csv"
OOF_PRED_PATH = OUTPUT_DIR / "oof_predictions.csv"
OOF_SCORE_PATH = OUTPUT_DIR / "oof_scores.csv"
BLEND_PATH = OUTPUT_DIR / "blend_alpha_sweep.csv"
ENSEMBLE_DIR = OUTPUT_DIR / "models"
SUBMISSION_PATH = OUTPUT_DIR / "my_chemprop_activity_submission.csv"
TEST_DETAIL_PATH = OUTPUT_DIR / "test_predictions_detail.csv"
KEPT_DESCS_PATH = OUTPUT_DIR / "kept_descriptors.txt"

# ── Endpoints ──────────────────────────────────────────────────────────────────
ISOFORMS = ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")
TARGET_COLS = [f"{iso}_pIC50_direct_inhibition" for iso in ISOFORMS]
LOG2FC_COLS = [f"{iso}_log2fc_estimate" for iso in ISOFORMS]
HIT_COLS = [f"{iso}_is_hit" for iso in ISOFORMS]
N_ISO = len(ISOFORMS)

# ── Assay constants (challenge definitions) ────────────────────────────────────
TOP_CONC_PIC50 = -float(np.log10(49.5e-6))  # 4.305 — top screening concentration
HIT_LOG2FC = -1.0
HIT_FDR = 0.05

# ── Label handling ─────────────────────────────────────────────────────────────
AUX_TASK_WEIGHT = 0.2  # weight of each log2fc head relative to a pIC50 head
USE_CENSORED_LOSS = True  # BoundedMSE + lt_mask on pIC50 < TOP_CONC_PIC50
USE_CI_WEIGHTS = True  # per-compound inverse credible-interval weighting
CI_WEIGHT_FLOOR = 1.0  # keeps a zero-width band from dominating
PSEUDO_PIC50_CLIP = (2.0, 8.0)  # calibrated pseudo-labels are clipped to this range
PRED_CLIP = (2.5, 9.0)  # final predictions clipped to a physically sane range

# ── Single-dose -> pIC50 calibration ───────────────────────────────────────────
# "linear" reproduces cyp_label_utils.fit_log2fc_calibration; "svr" is arm C of the
# PXR write-up (1-input RBF SVR, the 1-D analogue of matcha-croissant's 2-input fit);
# "auto" picks per isoform by 5-fold CV MAE on the paired points of the training fold.
CALIBRATION_KIND = "auto"
SVR_PARAMS = dict(kernel="rbf", C=10.0, gamma="scale", epsilon=0.1)

# ── Proxy model ────────────────────────────────────────────────────────────────
TRAIN_PROXY_MODEL = True
# Restrict proxy labels to isoforms where "non-hit => inactive" holds (the >= 0.75 gate
# from §1.5 of the analysis doc). Off by default: routing plus censoring already
# contains the risk, and switching it on discards most of the CYP2D6 proxy set.
PROXY_NONHIT_EVIDENCE_GATE = None  # e.g. 0.75 to switch the gate on
PROXY_MIN_LABELS = 200  # skip an isoform's proxy labels below this count

# ── Potency routing ────────────────────────────────────────────────────────────
BLEND_THRESHOLD = TOP_CONC_PIC50  # matcha-croissant used pEC50 4.5; ours is 4.305
BLEND_RAMP = 0.5  # log units over which the proxy weight ramps in
BLEND_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)  # swept per isoform on OOF predictions

# ── Descriptor block ───────────────────────────────────────────────────────────
DESC_ABS_MAX = 1e10  # drop descriptor columns with absurd magnitudes (Ipc and friends)
DESC_CLIP = 10.0  # clip standardised descriptors, in standard deviations

# ── Structural filtering (matcha-croissant dropped ~30 such compounds) ─────────
FILTER_TO_TEST_ELEMENTS = True
FILTER_RING_SIZE_ABOVE = 0  # 0 = off; e.g. 12 drops macrocycles

# ── Cross-validation ───────────────────────────────────────────────────────────
N_FOLDS = 5
SCAFFOLD_SEED = 42
CV_MAX_EPOCHS = 50
CV_PATIENCE = 10
NUM_WORKERS = 0
BATCH_SIZE = 64

# ── Learning rates ─────────────────────────────────────────────────────────────
INIT_LR = 1e-4
MAX_LR = 2e-4
FINAL_LR = 1e-5

# ── Ensembles ──────────────────────────────────────────────────────────────────
PRIMARY_SEEDS = [42, 123, 456, 789, 1337]
PROXY_SEEDS = [42, 123, 456]

# ── Hyperparameters ────────────────────────────────────────────────────────────
RUN_GRID_SEARCH = True
PARAM_GRID = {
    "mp_depth": [3, 4],
    "mp_hidden_dim": [300, 600],
    "ffn_hidden_dim": [600],
    "ffn_n_layers": [2],
    "dropout": [0.1],
}
DEFAULT_PARAMS = dict(
    mp_depth=3, mp_hidden_dim=600, ffn_hidden_dim=600, ffn_n_layers=2, dropout=0.1
)

_BANNER = "═" * 78


# ══════════════════════════════════════════════════════════════════════════════
# Scoring — the challenge's ST-RAE
# ══════════════════════════════════════════════════════════════════════════════

def _st_rae_local(y_true, y_pred, lower, upper) -> float:
    """Soft-threshold relative absolute error.

    A prediction anywhere inside ``[lower, upper]`` costs zero; outside it, only the
    distance to the nearest bound counts. The naive constant predictor in the
    denominator goes through the same rule, so 1.0 still means "no better than
    predicting the mean". Identical to
    ``evaluation.custom_scoring_functions.rae_soft_threshold_absolute_error``.
    """
    err = np.clip(y_pred - upper, 0, None) + np.clip(lower - y_pred, 0, None)
    mu = np.mean(y_true)
    base = np.clip(mu - upper, 0, None) + np.clip(lower - mu, 0, None)
    denom = np.sum(base)
    return float(np.sum(err) / denom) if denom > 0 else float("nan")


try:  # prefer the repo's own scorer so we cannot drift from the leaderboard
    from evaluation.custom_scoring_functions import (  # noqa: E402
        rae_soft_threshold_absolute_error as _st_rae_repo,
    )

    def st_rae(y_true, y_pred, lower, upper) -> float:
        return float(
            _st_rae_repo(y_true, y_pred, y_true_upper=upper, y_true_lower=lower)
        )

    _SCORER_SOURCE = "evaluation/custom_scoring_functions.py"
except Exception:  # pragma: no cover - standalone VM without the repo package
    st_rae = _st_rae_local
    _SCORER_SOURCE = "local copy"


def score_per_isoform(truth: pd.DataFrame, preds: np.ndarray, label: str = "") -> dict:
    """ST-RAE / MAE / Spearman per isoform plus the macro average the board ranks on."""
    rows = []
    for j, iso in enumerate(ISOFORMS):
        col = f"{iso}_pIC50_direct_inhibition"
        y = truth[col].to_numpy(dtype=float)
        m = np.isfinite(y) & np.isfinite(preds[:, j])
        if m.sum() < 5:
            rows.append(dict(isoform=iso, n=int(m.sum()), st_rae=np.nan, mae=np.nan, spearman=np.nan))
            continue
        lo = truth[f"{col}_conf_low"].to_numpy(dtype=float)[m]
        hi = truth[f"{col}_conf_high"].to_numpy(dtype=float)[m]
        yv, pv = y[m], preds[m, j]
        lo = np.where(np.isfinite(lo), lo, yv)
        hi = np.where(np.isfinite(hi), hi, yv)
        rho, _ = stats.spearmanr(yv, pv)
        rows.append(
            dict(
                isoform=iso,
                n=int(m.sum()),
                st_rae=st_rae(yv, pv, lo, hi),
                mae=float(np.mean(np.abs(yv - pv))),
                spearman=float(rho),
            )
        )
    df = pd.DataFrame(rows)
    ma = {k: float(np.nanmean(df[k])) for k in ("st_rae", "mae", "spearman")}
    if label:
        line = "  ".join(f"{r.isoform} {r.st_rae:.3f}" for r in df.itertuples())
        print(f"  [{label}]  MA ST-RAE={ma['st_rae']:.4f}  |  {line}")
    return {"per_isoform": df, **{f"ma_{k}": v for k, v in ma.items()}}


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def read_table(filename: str) -> pd.DataFrame:
    """Read a challenge CSV from ``LOCAL_DATA_DIR`` if it is there, else from HF."""
    local = LOCAL_DATA_DIR / filename
    if local.exists():
        print(f"  {filename}  <- {local}")
        return pd.read_csv(local)
    url = f"{HF_PREFIX}/{filename}"
    print(f"  {filename}  <- {url}")
    return pd.read_csv(url)


def widen_single_concentration(single_df: pd.DataFrame) -> pd.DataFrame:
    """Long primary-screen table -> one row per compound.

    Same reshape as ``cyp_label_utils.widen_single_concentration``: columns
    ``<ISO>_log2fc_estimate`` and ``<ISO>_is_hit`` using the challenge's own hit
    definition (log2fc < -1 and FDR < 0.05).
    """
    df = single_df.copy()
    # cast to int before the pivot: pivot_table is happier aggregating numerics, and the
    # column is turned back into a bool below.
    df["is_hit"] = (
        (df.log2fc_estimate < HIT_LOG2FC) & (df.log2fc_fdr < HIT_FDR)
    ).astype(int)
    wide = df.pivot_table(
        index="Molecule_Name",
        columns="enzyme",
        values=["log2fc_estimate", "is_hit"],
        aggfunc="first",
    )
    wide.columns = [f"{enzyme}_{field}" for field, enzyme in wide.columns]
    wide = wide.reset_index()
    for col in HIT_COLS:
        if col in wide.columns:
            wide[col] = pd.to_numeric(wide[col], errors="coerce").fillna(0).astype(bool)
    for col in LOG2FC_COLS + HIT_COLS:
        if col not in wide.columns:
            wide[col] = np.nan if col in LOG2FC_COLS else False
    return wide


def build_compound_table(drc: pd.DataFrame, single: pd.DataFrame) -> pd.DataFrame:
    """One row per training compound: SMILES, DRC labels + bounds, screen readouts.

    An outer join, so compounds that were screened but never fitted are kept — they
    are the entire point of the proxy model. Rows that end up without a SMILES (the
    single-concentration file does not always carry structures) are dropped, with the
    count printed, because there is nothing to featurize.
    """
    wide = widen_single_concentration(single)
    if "SMILES" in single.columns:
        smiles_lookup = (
            single[["Molecule_Name", "SMILES"]].dropna().drop_duplicates("Molecule_Name")
        )
        wide = wide.merge(smiles_lookup, on="Molecule_Name", how="left")

    df = drc.merge(wide, on="Molecule_Name", how="outer", suffixes=("", "_screen"))
    if "SMILES_screen" in df.columns:
        df["SMILES"] = df["SMILES"].fillna(df["SMILES_screen"])
        df = df.drop(columns=["SMILES_screen"])

    for col in TARGET_COLS:
        for suffix in ("", "_conf_low", "_conf_high"):
            if col + suffix not in df.columns:
                df[col + suffix] = np.nan
    for col in HIT_COLS:
        df[col] = df[col].fillna(False).astype(bool)

    n_no_smiles = int(df["SMILES"].isna().sum())
    if n_no_smiles:
        print(f"  dropped {n_no_smiles} screened compounds with no SMILES available")
        df = df[df["SMILES"].notna()]
    return df.reset_index(drop=True)


def report_coverage(df: pd.DataFrame) -> None:
    rows = []
    for iso in ISOFORMS:
        pic50 = df[f"{iso}_pIC50_direct_inhibition"]
        log2fc = df[f"{iso}_log2fc_estimate"]
        is_hit = df[f"{iso}_is_hit"]
        censored = pic50 < TOP_CONC_PIC50
        rows.append(
            dict(
                isoform=iso,
                drc=int(pic50.notna().sum()),
                censored=int(censored.sum()),
                screened=int(log2fc.notna().sum()),
                hits=int(is_hit.sum()),
                hits_no_drc=int((is_hit & pic50.isna()).sum()),
                nonhits_no_drc=int((~is_hit & log2fc.notna() & pic50.isna()).sum()),
            )
        )
    print("\nLabel coverage")
    print(pd.DataFrame(rows).to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# Structures and descriptors
# ══════════════════════════════════════════════════════════════════════════════

_LARGEST_FRAGMENT = rdMolStandardize.LargestFragmentChooser()
_UNCHARGER = rdMolStandardize.Uncharger()


def mol_from_smiles(smi: str):
    """Parse, keep the largest fragment, neutralise. Returns None on failure."""
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


def structural_filter(mols: list, test_mols: list) -> np.ndarray:
    """Drop training compounds the test set cannot possibly contain chemistry for.

    matcha-croissant removed "~30 compounds with elements or high-level structures
    (macrocycles, long acyclic molecules) not found in either test set". The element
    whitelist is derived from the test set itself, so it needs no hand-maintained list.
    """
    keep = np.ones(len(mols), dtype=bool)
    if FILTER_TO_TEST_ELEMENTS:
        allowed = {a.GetAtomicNum() for m in test_mols for a in m.GetAtoms()}
        for i, mol in enumerate(mols):
            if any(a.GetAtomicNum() not in allowed for a in mol.GetAtoms()):
                keep[i] = False
    if FILTER_RING_SIZE_ABOVE:
        for i, mol in enumerate(mols):
            ring_sizes = [len(r) for r in mol.GetRingInfo().AtomRings()]
            if ring_sizes and max(ring_sizes) > FILTER_RING_SIZE_ABOVE:
                keep[i] = False
    if (~keep).any():
        print(f"  structural filter dropped {int((~keep).sum())} training compounds")
    return keep


_DESC_LIST = [(name, fn) for name, fn in Descriptors.descList]
ALL_DESC_NAMES = [name for name, _ in _DESC_LIST]


def compute_rdkit_descriptors(mols: list) -> np.ndarray:
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
    return np.array(rows, dtype=float)


def select_valid_columns(arr: np.ndarray) -> np.ndarray:
    """Finite everywhere in the training set, not constant, not absurdly scaled.

    The magnitude test is what keeps ``Ipc`` and friends — finite but routinely 1e20 —
    from swamping the standardiser and, through it, the descriptor block of the FFN
    input.
    """
    finite = np.all(np.isfinite(arr), axis=0)
    varied = np.var(arr, axis=0) > 0
    sane = np.max(np.abs(arr), axis=0) < DESC_ABS_MAX
    return finite & varied & sane


def scale_descriptors(scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    """Standardise and clip: a hold-out compound sitting 200 sd out on one descriptor
    should not dominate the concatenated input vector."""
    return np.clip(scaler.transform(X), -DESC_CLIP, DESC_CLIP)


# ══════════════════════════════════════════════════════════════════════════════
# Scaffold folds
# ══════════════════════════════════════════════════════════════════════════════

def murcko_scaffold_folds(mols: list, n_folds: int, seed: int) -> np.ndarray:
    """Bemis-Murcko scaffold K-fold assignment, one fold index per compound.

    Scaffold groups are shuffled, ordered largest-first, and each is dropped into the
    currently smallest fold. Whole scaffolds stay together, so a hold-out fold contains
    chemotypes the training folds never saw — pessimistic against the real test set
    (an analog expansion of screening hits), and exactly what makes it useful for
    ranking modelling choices.
    """
    rng = random.Random(seed)
    groups: dict[str, list[int]] = {}
    for i, mol in enumerate(mols):
        try:
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        except Exception:
            scaffold = ""
        groups.setdefault(scaffold or f"__singleton_{i}", []).append(i)

    keys = list(groups)
    rng.shuffle(keys)
    keys.sort(key=lambda k: len(groups[k]), reverse=True)

    folds = np.zeros(len(mols), dtype=int)
    sizes = [0] * n_folds
    for key in keys:
        f = int(np.argmin(sizes))
        for i in groups[key]:
            folds[i] = f
        sizes[f] += len(groups[key])
    print(f"  scaffold folds: {len(keys)} scaffolds -> sizes {sizes}")
    return folds


# ══════════════════════════════════════════════════════════════════════════════
# Single-dose -> pIC50 calibration
# ══════════════════════════════════════════════════════════════════════════════

def fit_calibration(train_df: pd.DataFrame, isoform: str) -> dict:
    """Learn ``log2fc_estimate -> pIC50`` on the compounds of this fold that have both.

    Fits the linear form used by ``cyp_label_utils`` and the 1-input RBF SVR of arm C,
    compares them by 5-fold CV MAE on the paired points, and returns the winner under
    ``CALIBRATION_KIND``. Fold-internal by construction — call it with the training
    fold only, never the whole set, or the pseudo-labels leak the hold-out.
    """
    y = train_df[f"{isoform}_pIC50_direct_inhibition"].to_numpy(dtype=float)
    x = train_df[f"{isoform}_log2fc_estimate"].to_numpy(dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    n = int(m.sum())
    if n < 50:
        raise ValueError(f"{isoform}: only {n} paired points, refusing to calibrate")
    x, y = x[m], y[m]

    slope, intercept, r_value, _, _ = stats.linregress(x, y)
    resid_sd = float(np.std(y - (slope * x + intercept)))
    rho, _ = stats.spearmanr(x, y)

    def _linear(v):
        return slope * v + intercept

    svr = make_pipeline(StandardScaler(), SVR(**SVR_PARAMS))
    svr.fit(x.reshape(-1, 1), y)

    def _svr(v):
        return svr.predict(np.asarray(v, dtype=float).reshape(-1, 1))

    # honest comparison of the two calibrations, on the paired points only
    kf = KFold(n_splits=5, shuffle=True, random_state=SCAFFOLD_SEED)
    mae_lin, mae_svr = [], []
    for tr, va in kf.split(x):
        s, b, *_ = stats.linregress(x[tr], y[tr])
        mae_lin.append(np.mean(np.abs(y[va] - (s * x[va] + b))))
        inner = make_pipeline(StandardScaler(), SVR(**SVR_PARAMS))
        inner.fit(x[tr].reshape(-1, 1), y[tr])
        mae_svr.append(np.mean(np.abs(y[va] - inner.predict(x[va].reshape(-1, 1)))))
    mae_lin, mae_svr = float(np.mean(mae_lin)), float(np.mean(mae_svr))

    if CALIBRATION_KIND == "linear":
        kind = "linear"
    elif CALIBRATION_KIND == "svr":
        kind = "svr"
    else:
        kind = "svr" if mae_svr < mae_lin else "linear"

    return dict(
        isoform=isoform,
        n=n,
        kind=kind,
        predict=_svr if kind == "svr" else _linear,
        slope=float(slope),
        intercept=float(intercept),
        r2=float(r_value**2),
        spearman=float(rho),
        resid_sd=resid_sd,
        cv_mae_linear=mae_lin,
        cv_mae_svr=mae_svr,
    )


def nonhit_inactivity_evidence(train_df: pd.DataFrame, isoform: str) -> tuple[float, float, int]:
    """How often "single-dose non-hit" really does mean "inactive in the DRC".

    Measured on non-hits that got a DRC anyway. 0.77/0.78 for CYP3A4/CYP1A2 but
    0.51/0.39 for CYP2C9/CYP2D6 — the reason a blanket inactive rule backfires.
    """
    pic50 = train_df[f"{isoform}_pIC50_direct_inhibition"]
    is_hit = train_df[f"{isoform}_is_hit"].astype(bool)
    vals = pic50[pic50.notna() & ~is_hit]
    if len(vals) < 30:
        return 0.0, TOP_CONC_PIC50, len(vals)
    return float((vals < TOP_CONC_PIC50).mean()), float(vals.median()), len(vals)


def fit_all_calibrations(train_df: pd.DataFrame, verbose: bool = True) -> dict:
    cals = {}
    for iso in ISOFORMS:
        cal = fit_calibration(train_df, iso)
        frac, median_pic50, n_nonhit = nonhit_inactivity_evidence(train_df, iso)
        cal.update(nonhit_frac_below_top=frac, nonhit_median_pIC50=median_pic50,
                   n_nonhit_with_drc=n_nonhit)
        cals[iso] = cal
    if verbose:
        cols = ["isoform", "n", "kind", "r2", "spearman", "resid_sd",
                "cv_mae_linear", "cv_mae_svr", "nonhit_frac_below_top"]
        table = pd.DataFrame([{k: c[k] for k in cols} for c in cals.values()])
        print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return cals


# ══════════════════════════════════════════════════════════════════════════════
# Label matrices
# ══════════════════════════════════════════════════════════════════════════════

def build_primary_labels(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Y (n x 8), lt_mask (n x 8), per-compound weights (n,) for the primary model.

    Columns 0-3 are the real DRC pIC50s (NaN where unmeasured — chemprop masks those
    out of the loss per task), columns 4-7 the single-dose log2fc readouts. No
    pseudo-labels: those live in the proxy model.
    """
    n = len(df)
    y = np.full((n, 2 * N_ISO), np.nan)
    lt = np.zeros((n, 2 * N_ISO), dtype=bool)
    w_parts, w_counts = np.zeros(n), np.zeros(n)

    for j, iso in enumerate(ISOFORMS):
        col = f"{iso}_pIC50_direct_inhibition"
        pic50 = df[col].to_numpy(dtype=float)
        y[:, j] = pic50
        y[:, N_ISO + j] = df[f"{iso}_log2fc_estimate"].to_numpy(dtype=float)

        observed = np.isfinite(pic50)
        if USE_CENSORED_LOSS:
            # below the top tested concentration the fit is an extrapolation: the honest
            # statement is "pIC50 is at most this", so predicting lower must be free.
            lt[:, j] = observed & (pic50 < TOP_CONC_PIC50)

        lo = df[f"{col}_conf_low"].to_numpy(dtype=float)
        hi = df[f"{col}_conf_high"].to_numpy(dtype=float)
        width = np.where(np.isfinite(hi) & np.isfinite(lo), hi - lo, 0.0)
        width = np.clip(width, 0.0, None)
        wj = 1.0 / (CI_WEIGHT_FLOOR + width)
        w_parts += np.where(observed, wj, 0.0)
        w_counts += observed

    if USE_CI_WEIGHTS:
        weights = np.divide(w_parts, w_counts, out=np.ones(n), where=w_counts > 0)
        weights = weights / weights.mean()
    else:
        weights = np.ones(n)
    return y, lt, weights


def build_proxy_labels(df: pd.DataFrame, calibrations: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Y (n x 4), lt_mask (n x 4) of calibrated pseudo-pIC50s for the proxy model.

    A compound gets a proxy label for an isoform when it was screened but never carried
    through to a DRC for that isoform — matcha-croissant's ">10,000 proxy pEC50
    measurements", here from the 1-D calibration.

    CYP2D6 is the isoform to watch. All 1,198 of its screening hits reached a DRC, so
    every compound labelled here is a *non-hit* — and only 39% of CYP2D6 non-hits that
    did get a DRC came back below the top tested concentration (median pIC50 4.35). The
    majority of this isoform's proxy labels are therefore too weak. Three things keep
    that contained, none of them the censoring:

    1. These labels never enter the primary model, so they cannot repeat the §1.4
       failure (2,171 pseudo-inactives against 1,117 real labels, ST-RAE 1.38 -> 2.28).
       The primary model's only exposure to these compounds is the raw, *measured*
       log2fc auxiliary head — a readout, never an inferred potency.
    2. Each label is that compound's own log2fc pushed through the calibration, not the
       single non-hit median §1.5 uses. A non-hit at log2fc -0.9 and one at -0.05 get
       different labels, so the spread the 61% live in survives.
    3. ``choose_blend_alphas`` decides out of fold how much of this model reaches the
       submission. An isoform whose proxy labels are this unreliable should select
       alpha = 0 and fall back entirely on the primary model — the sweep is the arbiter,
       and CYP2D6 is where it earns its keep.

    Setting PROXY_NONHIT_EVIDENCE_GATE = 0.75 refuses the labels up front instead,
    reproducing §1.5's validated rule: CYP1A2 and CYP3A4 keep theirs, CYP2C9 and CYP2D6
    lose theirs (CYP2D6 loses all of them, having no hits without a DRC).
    """
    n = len(df)
    y = np.full((n, N_ISO), np.nan)
    lt = np.zeros((n, N_ISO), dtype=bool)
    info = {}

    for j, iso in enumerate(ISOFORMS):
        cal = calibrations[iso]
        pic50 = df[f"{iso}_pIC50_direct_inhibition"].to_numpy(dtype=float)
        log2fc = df[f"{iso}_log2fc_estimate"].to_numpy(dtype=float)
        is_hit = df[f"{iso}_is_hit"].to_numpy(dtype=bool)

        m = ~np.isfinite(pic50) & np.isfinite(log2fc)
        if PROXY_NONHIT_EVIDENCE_GATE is not None:
            if cal["nonhit_frac_below_top"] < PROXY_NONHIT_EVIDENCE_GATE:
                m &= is_hit
        if m.sum() < PROXY_MIN_LABELS:
            info[iso] = dict(n=int(m.sum()), used=False)
            continue

        vals = np.clip(cal["predict"](log2fc[m]), *PSEUDO_PIC50_CLIP)
        y[m, j] = vals
        lt[m, j] = vals < TOP_CONC_PIC50
        info[iso] = dict(
            n=int(m.sum()),
            used=True,
            n_hits=int((m & is_hit).sum()),
            n_nonhits=int((m & ~is_hit).sum()),
            median=float(np.median(vals)),
            frac_censored=float(np.mean(vals < TOP_CONC_PIC50)),
        )
    return y, lt, info


# ══════════════════════════════════════════════════════════════════════════════
# Model
# ══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class BestStateTracker(pl.Callback):
    """Keep the best-validation weights in memory and hand them back after fit.

    chemprop trains without checkpointing here, so without this the model left at the
    end of an early-stopped fit is ``patience`` epochs past its best.
    """

    def __init__(self, monitor: str = "val_loss"):
        self.monitor = monitor
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self._state = None

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        value = trainer.callback_metrics.get(self.monitor)
        if value is None:
            return
        value = float(value)
        if value < self.best_val_loss:
            self.best_val_loss = value
            self.best_epoch = trainer.current_epoch
            self._state = {k: v.detach().cpu().clone() for k, v in pl_module.state_dict().items()}

    def restore(self, pl_module) -> None:
        if self._state is not None:
            pl_module.load_state_dict(self._state)


def task_weight_vector(n_tasks: int) -> list[float]:
    """1.0 for every pIC50 head, AUX_TASK_WEIGHT for every log2fc head."""
    if n_tasks == N_ISO:
        return [1.0] * N_ISO
    return [1.0] * N_ISO + [AUX_TASK_WEIGHT] * (n_tasks - N_ISO)


def build_mpnn(
    n_descriptors: int,
    n_tasks: int,
    target_scaler,
    mp_depth: int,
    mp_hidden_dim: int,
    ffn_hidden_dim: int,
    ffn_n_layers: int,
    dropout: float,
) -> models.MPNN:
    feat = featurizers.SimpleMoleculeMolGraphFeaturizer()
    mp = nn.BondMessagePassing(
        d_v=feat.atom_fdim, d_e=feat.bond_fdim, depth=mp_depth, d_h=mp_hidden_dim
    )
    agg = nn.MeanAggregation()
    weights = task_weight_vector(n_tasks)
    criterion = (
        nn.metrics.BoundedMSE(task_weights=weights)
        if USE_CENSORED_LOSS
        else nn.metrics.MSE(task_weights=weights)
    )
    ffn = nn.RegressionFFN(
        n_tasks=n_tasks,
        input_dim=mp.output_dim + n_descriptors,
        hidden_dim=ffn_hidden_dim,
        n_layers=ffn_n_layers,
        dropout=dropout,
        criterion=criterion,
        output_transform=nn.UnscaleTransform.from_standard_scaler(target_scaler),
    )
    return models.MPNN(
        mp,
        agg,
        ffn,
        batch_norm=True,
        metrics=[nn.metrics.RMSE(), nn.metrics.MAE()],
        init_lr=INIT_LR,
        max_lr=MAX_LR,
        final_lr=FINAL_LR,
    )


def make_datapoints(mols, y, weights, x_d, lt_mask=None) -> list:
    n_tasks = y.shape[1]
    lt_mask = np.zeros_like(y, dtype=bool) if lt_mask is None else lt_mask
    return [
        data.MoleculeDatapoint(
            mol=mol,
            y=y[i].astype(float),
            weight=float(weights[i]),
            x_d=x_d[i].astype(float),
            lt_mask=lt_mask[i].astype(bool),
            gt_mask=np.zeros(n_tasks, dtype=bool),
        )
        for i, mol in enumerate(mols)
    ]


def make_dataset(mols, y, weights, x_d, lt_mask=None) -> data.MoleculeDataset:
    feat = featurizers.SimpleMoleculeMolGraphFeaturizer()
    return data.MoleculeDataset(make_datapoints(mols, y, weights, x_d, lt_mask), feat)


def fit_mpnn(
    train_dset: data.MoleculeDataset,
    val_dset: data.MoleculeDataset | None,
    n_descriptors: int,
    params: dict,
    max_epochs: int,
    patience: int,
    seed: int,
    progress: bool = False,
) -> tuple[models.MPNN, pl.Trainer, BestStateTracker | None]:
    """Fit one MPNN. Targets are standardised per task, so the pIC50 and log2fc heads
    contribute comparably and ``task_weights`` means what it says."""
    set_seed(seed)
    target_scaler = train_dset.normalize_targets()
    train_loader = data.build_dataloader(
        train_dset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, seed=seed
    )

    callbacks, val_loader, tracker = [], None, None
    if val_dset is not None:
        val_dset.normalize_targets(target_scaler)
        val_loader = data.build_dataloader(
            val_dset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False
        )
        tracker = BestStateTracker()
        callbacks = [EarlyStopping(monitor="val_loss", patience=patience, mode="min"), tracker]

    mpnn = build_mpnn(n_descriptors, train_dset.Y.shape[1], target_scaler, **params)
    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=progress,
        enable_model_summary=False,
        accelerator="auto",
        devices=1,
        max_epochs=max_epochs,
        callbacks=callbacks,
    )
    trainer.fit(mpnn, train_loader, val_loader)
    if tracker is not None:
        tracker.restore(mpnn)
    return mpnn, trainer, tracker


def predict(mpnn: models.MPNN, trainer: pl.Trainer, dset: data.MoleculeDataset) -> np.ndarray:
    """Predictions in the original units — the model's UnscaleTransform switches on in
    eval mode, so no manual inverse transform is needed."""
    loader = data.build_dataloader(
        dset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False
    )
    mpnn.eval()
    raw = trainer.predict(mpnn, loader)
    return torch.cat(raw).numpy().reshape(len(dset), -1)


# ══════════════════════════════════════════════════════════════════════════════
# Potency routing
# ══════════════════════════════════════════════════════════════════════════════

def blend_predictions(primary: np.ndarray, proxy: np.ndarray, alpha: float) -> np.ndarray:
    """Fold the proxy model in where *it* says the compound is weak.

    ``alpha`` is the maximum share the proxy model can take, reached BLEND_RAMP log
    units below BLEND_THRESHOLD and ramping linearly to zero at the threshold. Routing
    on the proxy model's own prediction (not the primary's) is deliberate: it is the
    model that is trustworthy at the weak end, so it decides when it is on home ground.
    """
    if alpha <= 0:
        return primary.copy()
    share = alpha * np.clip((BLEND_THRESHOLD - proxy) / max(BLEND_RAMP, 1e-6), 0.0, 1.0)
    share = np.where(np.isfinite(proxy), share, 0.0)
    return (1.0 - share) * primary + share * np.nan_to_num(proxy)


def choose_blend_alphas(truth: pd.DataFrame, primary: np.ndarray, proxy: np.ndarray) -> dict:
    """Sweep alpha per isoform on out-of-fold predictions; keep whatever wins ST-RAE.

    Per isoform rather than globally because the isoforms disagree about everything
    else in this dataset, and alpha = 0 is always in the sweep — an isoform whose proxy
    model is useless simply switches the blend off.
    """
    rows, chosen = [], {}
    for j, iso in enumerate(ISOFORMS):
        col = f"{iso}_pIC50_direct_inhibition"
        y = truth[col].to_numpy(dtype=float)
        m = np.isfinite(y) & np.isfinite(primary[:, j])
        lo = truth[f"{col}_conf_low"].to_numpy(dtype=float)[m]
        hi = truth[f"{col}_conf_high"].to_numpy(dtype=float)[m]
        yv = y[m]
        lo = np.where(np.isfinite(lo), lo, yv)
        hi = np.where(np.isfinite(hi), hi, yv)

        best_alpha, best_score = 0.0, np.inf
        for alpha in BLEND_ALPHAS:
            blended = blend_predictions(primary[m, j], proxy[m, j], alpha)
            score = st_rae(yv, blended, lo, hi)
            rows.append(dict(isoform=iso, alpha=alpha, st_rae=score, n=int(m.sum())))
            if score < best_score:
                best_alpha, best_score = alpha, score
        chosen[iso] = best_alpha

    sweep = pd.DataFrame(rows)
    print(sweep.pivot(index="alpha", columns="isoform", values="st_rae")
          .to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"  chosen alpha per isoform: {chosen}")
    sweep.to_csv(BLEND_PATH, index=False)
    return chosen


def apply_blend(primary: np.ndarray, proxy: np.ndarray, alphas: dict) -> np.ndarray:
    out = primary.copy()
    for j, iso in enumerate(ISOFORMS):
        out[:, j] = blend_predictions(primary[:, j], proxy[:, j], alphas.get(iso, 0.0))
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Cross-validation stages
# ══════════════════════════════════════════════════════════════════════════════

def run_primary_fold(
    fold: int, folds: np.ndarray, mols: list, df: pd.DataFrame, x_d_raw: np.ndarray,
    params: dict, seed: int = SCAFFOLD_SEED,
) -> tuple[np.ndarray, float, int]:
    """Train the primary model on every fold but ``fold``; return its predictions for
    the held-out fold, the best validation loss, and the epoch it happened at."""
    tr, va = folds != fold, folds == fold
    y, lt, w = build_primary_labels(df)

    scaler = StandardScaler().fit(x_d_raw[tr])
    x_tr = scale_descriptors(scaler, x_d_raw[tr])
    x_va = scale_descriptors(scaler, x_d_raw[va])
    mols_tr = [m for m, keep in zip(mols, tr) if keep]
    mols_va = [m for m, keep in zip(mols, va) if keep]

    train_dset = make_dataset(mols_tr, y[tr], w[tr], x_tr, lt[tr])
    val_dset = make_dataset(mols_va, y[va], w[va], x_va, lt[va])

    mpnn, trainer, tracker = fit_mpnn(
        train_dset, val_dset, x_tr.shape[1], params,
        max_epochs=CV_MAX_EPOCHS, patience=CV_PATIENCE, seed=seed,
    )
    preds = predict(mpnn, trainer, val_dset)[:, :N_ISO]
    return preds, tracker.best_val_loss, tracker.best_epoch


def run_proxy_fold(
    fold: int, folds: np.ndarray, mols: list, df: pd.DataFrame, x_d_raw: np.ndarray,
    params: dict, seed: int = SCAFFOLD_SEED,
) -> np.ndarray:
    """Same fold, but trained only on this fold's calibrated proxy labels.

    The calibration is fitted inside the training fold, so the hold-out never
    contributes to its own pseudo-labels.
    """
    tr, va = folds != fold, folds == fold
    cals = fit_all_calibrations(df[tr], verbose=False)
    y_proxy, lt_proxy, _ = build_proxy_labels(df, cals)

    has_label = np.isfinite(y_proxy).any(axis=1)
    tr_proxy = tr & has_label
    if tr_proxy.sum() < PROXY_MIN_LABELS:
        return np.full((int(va.sum()), N_ISO), np.nan)

    scaler = StandardScaler().fit(x_d_raw[tr_proxy])
    x_tr = scale_descriptors(scaler, x_d_raw[tr_proxy])
    x_va = scale_descriptors(scaler, x_d_raw[va])
    mols_tr = [m for m, keep in zip(mols, tr_proxy) if keep]
    mols_va = [m for m, keep in zip(mols, va) if keep]

    train_dset = make_dataset(
        mols_tr, y_proxy[tr_proxy], np.ones(int(tr_proxy.sum())), x_tr, lt_proxy[tr_proxy]
    )
    val_dset = make_dataset(
        mols_va, y_proxy[va], np.ones(int(va.sum())), x_va, lt_proxy[va]
    )
    # Held-out proxy labels exist for only some compounds, so early stopping on them is
    # noisy; a short fixed budget is steadier and the proxy model is only ever asked for
    # a coarse read at the weak end.
    mpnn, trainer, _ = fit_mpnn(
        train_dset, None, x_tr.shape[1], params,
        max_epochs=CV_MAX_EPOCHS, patience=CV_PATIENCE, seed=seed,
    )
    return predict(mpnn, trainer, val_dset)[:, :N_ISO]


def grid_search(folds, mols, df, x_d_raw) -> dict:
    """Scaffold-CV grid search on the primary model, ranked by macro ST-RAE."""
    if CV_RESULTS_PATH.exists():
        print(f"\nCV results found at {CV_RESULTS_PATH} — skipping grid search.")
        df_cv = pd.read_csv(CV_RESULTS_PATH).sort_values("ma_st_rae").reset_index(drop=True)
        print(df_cv.to_string(index=False))
        return {k: df_cv.iloc[0][k] for k in DEFAULT_PARAMS}

    combos = [dict(zip(PARAM_GRID, c)) for c in itertools.product(*PARAM_GRID.values())]
    print(f"\n{len(combos)} hyperparameter combos x {N_FOLDS} scaffold folds")

    results = []
    for i, params in enumerate(combos, start=1):
        print(f"\n[{i}/{len(combos)}] {params}")
        oof = np.full((len(df), N_ISO), np.nan)
        losses, epochs = [], []
        for fold in range(N_FOLDS):
            preds, loss, epoch = run_primary_fold(fold, folds, mols, df, x_d_raw, params)
            oof[folds == fold] = preds
            losses.append(loss)
            epochs.append(epoch)
            print(f"  fold {fold}: val_loss={loss:.4f}  best_epoch={epoch}")
        scored = score_per_isoform(df, oof, label="OOF")
        results.append({
            **params,
            "ma_st_rae": scored["ma_st_rae"],
            "ma_mae": scored["ma_mae"],
            "ma_spearman": scored["ma_spearman"],
            "mean_val_loss": float(np.mean(losses)),
            "mean_best_epoch": int(np.mean(epochs)),
        })

    df_cv = pd.DataFrame(results).sort_values("ma_st_rae").reset_index(drop=True)
    df_cv.to_csv(CV_RESULTS_PATH, index=False)
    print(f"\nCV results saved to {CV_RESULTS_PATH}")
    print(df_cv.to_string(index=False))
    return {k: df_cv.iloc[0][k] for k in DEFAULT_PARAMS}


def coerce_params(params: dict) -> dict:
    return dict(
        mp_depth=int(params["mp_depth"]),
        mp_hidden_dim=int(params["mp_hidden_dim"]),
        ffn_hidden_dim=int(params["ffn_hidden_dim"]),
        ffn_n_layers=int(params["ffn_n_layers"]),
        dropout=float(params["dropout"]),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ENSEMBLE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"{_BANNER}\nCYP direct-inhibition — chemprop multitask + proxy blend")
    print(f"outputs -> {OUTPUT_DIR}   ST-RAE from {_SCORER_SOURCE}\n{_BANNER}")

    # ── Step 1 — data ─────────────────────────────────────────────────────────
    print("\nStep 1 — loading data")
    drc = read_table(FILE_TRAIN_DRC)
    single = read_table(FILE_TRAIN_SINGLE)
    test = read_table(FILE_TEST)
    df = build_compound_table(drc, single)
    print(f"  {len(df)} training compounds, {len(test)} blinded test compounds")
    report_coverage(df)

    # ── Step 2 — structures, filtering, descriptors ───────────────────────────
    print("\nStep 2 — structures and descriptors")
    test_mols, test_ok = build_mols(test["SMILES"].tolist())
    if (~test_ok).any():
        print(f"  WARNING: {int((~test_ok).sum())} test SMILES failed to parse; "
              "those rows will be filled with the per-isoform training median")

    mols, keep = build_mols(df["SMILES"].tolist())
    df = df[keep].reset_index(drop=True)
    struct_keep = structural_filter(mols, test_mols)
    mols = [m for m, k in zip(mols, struct_keep) if k]
    df = df[struct_keep].reset_index(drop=True)
    print(f"  {len(mols)} training molecules after parsing and filtering")

    x_all = compute_rdkit_descriptors(mols)
    col_mask = select_valid_columns(x_all)
    kept_names = [n for n, k in zip(ALL_DESC_NAMES, col_mask) if k]
    KEPT_DESCS_PATH.write_text("\n".join(kept_names))
    x_d_raw = x_all[:, col_mask]
    x_d_test_raw = np.nan_to_num(
        compute_rdkit_descriptors(test_mols)[:, col_mask], nan=0.0, posinf=0.0, neginf=0.0
    )
    n_descriptors = x_d_raw.shape[1]
    print(f"  kept {n_descriptors}/{len(col_mask)} RDKit descriptors -> {KEPT_DESCS_PATH}")

    # ── Step 3 — single-dose calibration ──────────────────────────────────────
    print("\nStep 3 — single-dose -> pIC50 calibration (full training set; the CV "
          "folds refit this internally)")
    calibrations = fit_all_calibrations(df)
    pd.DataFrame([
        {k: v for k, v in cal.items() if k != "predict"} for cal in calibrations.values()
    ]).to_csv(CALIBRATION_PATH, index=False)

    _, _, proxy_info = build_proxy_labels(df, calibrations)
    print("\n  proxy label counts")
    print(pd.DataFrame(proxy_info).T.to_string())

    # ── Step 4 — scaffold folds and hyperparameters ───────────────────────────
    print("\nStep 4 — scaffold folds")
    folds = murcko_scaffold_folds(mols, N_FOLDS, SCAFFOLD_SEED)

    if RUN_GRID_SEARCH:
        print(f"\n{_BANNER}\nStep 4b — grid search\n{_BANNER}")
        best_params = coerce_params(grid_search(folds, mols, df, x_d_raw))
    else:
        best_params = coerce_params(DEFAULT_PARAMS)
    print(f"\nHyperparameters: {best_params}")

    # ── Step 5 — out-of-fold arms and blend weights ───────────────────────────
    print(f"\n{_BANNER}\nStep 5 — out-of-fold comparison of the three arms\n{_BANNER}")
    oof_primary = np.full((len(df), N_ISO), np.nan)
    oof_proxy = np.full((len(df), N_ISO), np.nan)
    for fold in range(N_FOLDS):
        print(f"\n  fold {fold + 1}/{N_FOLDS}")
        preds, loss, epoch = run_primary_fold(fold, folds, mols, df, x_d_raw, best_params)
        oof_primary[folds == fold] = preds
        print(f"    primary: val_loss={loss:.4f}  best_epoch={epoch}")
        if TRAIN_PROXY_MODEL:
            oof_proxy[folds == fold] = run_proxy_fold(
                fold, folds, mols, df, x_d_raw, best_params
            )

    print("\nOut-of-fold scores")
    scored_primary = score_per_isoform(df, oof_primary, label="P: primary multitask")
    if TRAIN_PROXY_MODEL:
        score_per_isoform(df, oof_proxy, label="X: proxy labels only")
        print("\nBlend sweep (ST-RAE, lower is better)")
        alphas = choose_blend_alphas(df, oof_primary, oof_proxy)
        oof_blend = apply_blend(oof_primary, oof_proxy, alphas)
        scored_blend = score_per_isoform(df, oof_blend, label="P+X: routed blend")
    else:
        alphas = {iso: 0.0 for iso in ISOFORMS}
        oof_blend, scored_blend = oof_primary, scored_primary

    oof_frame = pd.DataFrame({"Molecule_Name": df["Molecule_Name"], "fold": folds})
    for j, iso in enumerate(ISOFORMS):
        oof_frame[f"{iso}_primary"] = oof_primary[:, j]
        oof_frame[f"{iso}_proxy"] = oof_proxy[:, j]
        oof_frame[f"{iso}_blend"] = oof_blend[:, j]
        oof_frame[f"{iso}_true"] = df[f"{iso}_pIC50_direct_inhibition"].to_numpy()
    oof_frame.to_csv(OOF_PRED_PATH, index=False)

    scores = pd.concat([
        scored_primary["per_isoform"].assign(arm="primary"),
        scored_blend["per_isoform"].assign(arm="blend"),
    ])
    scores.to_csv(OOF_SCORE_PATH, index=False)
    print(f"\n  OOF predictions -> {OOF_PRED_PATH}\n  OOF scores      -> {OOF_SCORE_PATH}")

    # ── Step 6 — refit on everything, ensembled ───────────────────────────────
    print(f"\n{_BANNER}\nStep 6 — final ensembles on all training data\n{_BANNER}")
    final_scaler = StandardScaler().fit(x_d_raw)
    x_d_train = scale_descriptors(final_scaler, x_d_raw)
    x_d_test = scale_descriptors(final_scaler, x_d_test_raw)
    final_epochs = max(int(CV_MAX_EPOCHS * 1.2), 60)

    y, lt, w = build_primary_labels(df)
    primary_test_preds = []
    for i, seed in enumerate(PRIMARY_SEEDS, start=1):
        print(f"\n  primary member {i}/{len(PRIMARY_SEEDS)}  seed={seed}")
        train_dset = make_dataset(mols, y, w, x_d_train, lt)
        mpnn, trainer, _ = fit_mpnn(
            train_dset, None, n_descriptors, best_params,
            max_epochs=final_epochs, patience=CV_PATIENCE, seed=seed, progress=True,
        )
        torch.save(mpnn, ENSEMBLE_DIR / f"primary_seed{seed}.pt")
        test_dset = make_dataset(
            test_mols, np.full((len(test_mols), y.shape[1]), np.nan),
            np.ones(len(test_mols)), x_d_test,
        )
        primary_test_preds.append(predict(mpnn, trainer, test_dset)[:, :N_ISO])

    primary_test = np.mean(primary_test_preds, axis=0)

    proxy_test = np.full_like(primary_test, np.nan)
    if TRAIN_PROXY_MODEL and any(a > 0 for a in alphas.values()):
        y_proxy, lt_proxy, _ = build_proxy_labels(df, calibrations)
        has_label = np.isfinite(y_proxy).any(axis=1)
        mols_proxy = [m for m, k in zip(mols, has_label) if k]
        proxy_preds = []
        for i, seed in enumerate(PROXY_SEEDS, start=1):
            print(f"\n  proxy member {i}/{len(PROXY_SEEDS)}  seed={seed}  "
                  f"({int(has_label.sum())} compounds)")
            train_dset = make_dataset(
                mols_proxy, y_proxy[has_label], np.ones(int(has_label.sum())),
                x_d_train[has_label], lt_proxy[has_label],
            )
            mpnn, trainer, _ = fit_mpnn(
                train_dset, None, n_descriptors, best_params,
                max_epochs=final_epochs, patience=CV_PATIENCE, seed=seed, progress=True,
            )
            torch.save(mpnn, ENSEMBLE_DIR / f"proxy_seed{seed}.pt")
            test_dset = make_dataset(
                test_mols, np.full((len(test_mols), N_ISO), np.nan),
                np.ones(len(test_mols)), x_d_test,
            )
            proxy_preds.append(predict(mpnn, trainer, test_dset)[:, :N_ISO])
        proxy_test = np.mean(proxy_preds, axis=0)
    else:
        print("\n  every isoform chose alpha = 0 — skipping the proxy ensemble")

    final_test = np.clip(apply_blend(primary_test, proxy_test, alphas), *PRED_CLIP)

    # ── Step 7 — submission ───────────────────────────────────────────────────
    print(f"\n{_BANNER}\nStep 7 — submission\n{_BANNER}")
    submission = pd.DataFrame({
        "SMILES": test["SMILES"].to_numpy(),
        "Molecule_Name": test["Molecule_Name"].to_numpy(),
    })
    medians = {iso: float(df[f"{iso}_pIC50_direct_inhibition"].median()) for iso in ISOFORMS}
    for j, iso in enumerate(ISOFORMS):
        col = np.full(len(test), medians[iso])
        col[test_ok] = final_test[:, j]
        submission[f"{iso}_pIC50_direct_inhibition"] = col

    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"  submission -> {SUBMISSION_PATH}  ({len(submission)} rows)")

    detail = submission.copy()
    for j, iso in enumerate(ISOFORMS):
        detail.loc[test_ok, f"{iso}_primary"] = primary_test[:, j]
        detail.loc[test_ok, f"{iso}_proxy"] = proxy_test[:, j]
        detail[f"{iso}_alpha"] = alphas[iso]
    detail.to_csv(TEST_DETAIL_PATH, index=False)
    print(f"  per-arm detail -> {TEST_DETAIL_PATH}")

    try:
        from validation.activity_validation import validate_activity_submission

        ok, errors = validate_activity_submission(
            SUBMISSION_PATH, expected_ids=set(test["Molecule_Name"])
        )
        print("  ✅ Activity submission file is valid." if ok
              else "  ❌ submission invalid:\n    " + "\n    ".join(errors))
    except ImportError:
        print("  (validation package not importable — skipped the format check)")

    print("\nSummary")
    print(f"  MA ST-RAE, out of fold : primary {scored_primary['ma_st_rae']:.4f}  "
          f"blended {scored_blend['ma_st_rae']:.4f}")
    print(f"  blend alphas           : {alphas}")
    print("  Scaffold-split numbers are pessimistic against an analog-expansion test "
          "set — use them to rank choices, not to predict the leaderboard.")


if __name__ == "__main__":
    main()
