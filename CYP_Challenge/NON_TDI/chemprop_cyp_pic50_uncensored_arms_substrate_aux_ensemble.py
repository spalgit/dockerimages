"""
ChemProp CYP direct-inhibition pIC50 — RDKit 2D descriptors + single-shot log2fc
auxiliary heads + **CYP substrate auxiliary heads (Ni et al. 2025)** + SVR-proxy model
routed by predicted potency + multi-seed ensemble. **Censoring off. CYP2D6 on its own.**

Built from ``chemprop_cyp_pic50_rdkit2d_log2fc_aux_proxy_blend_ensemble.py.baseline``
rather than from the current head of that script, on the finding that the censored loss
did not help the non-TDI task. Three deliberate differences from the ``.baseline`` it
started as, and nothing else:

  1. **Censoring off** (``USE_CENSORED_LOSS = False``). Everywhere, including the proxy
     model's pseudo-labels — see "What 'censoring off' covers" below, because the
     ``.baseline`` censored those through a separate code path that ignored the flag.
  2. **CYP2D6 is not in the multitask.** ``TASK_ARMS`` splits the model in two.
  3. **Substrate auxiliary heads** from Ni et al. 2025.

Everything the ``.baseline`` did that is *not* in that list is untouched: no
``CENSOR_NONHITS`` synthesised bounds, no ``EVAL_REWEIGHT``, the original grid and
proxy-blend machinery. (One exception, flagged loudly at FINAL_EPOCHS_LEGACY below: the
final-refit epoch budget is a bug fix, not a design change.)

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

  MODEL P (primary) — one MPNN **per task arm**, not one overall. For an arm covering
      isoforms ``arm`` (k of them) with substrate heads for ``sub`` (s of them):
        k x <ISO>_pIC50_direct_inhibition   (primary,   task weight 1.0)
        k x <ISO>_log2fc_estimate           (auxiliary, task weight AUX_TASK_WEIGHT)
        s x <ISO>_is_substrate              (auxiliary, task weight AUX_SUBSTRATE_TASK_WEIGHT)
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

CYP2D6 is not in the multitask
------------------------------
``TASK_ARMS = (("CYP1A2","CYP2C9","CYP3A4"), ("CYP2D6",))``. Two independent models;
predictions are merged back per isoform, so nothing downstream changes.

CYP2D6 is uncorrelated with the other three on co-measured compounds (Spearman 0.06,
-0.12, 0.06, versus 0.38-0.69 among 1A2/2C9/3A4), and 55% of its labelled compounds carry
no other isoform label at all — so a shared trunk has nothing to share. It showed: OOF
pred_sd/true_sd was 0.49 for 2D6 against 0.98 for 3A4, i.e. the head was collapsing toward
the mean inside a trunk optimised for a signal it does not have. Set
``TASK_ARMS = (ISOFORMS,)`` to recover the ``.baseline``'s single joint model.

The same logic governs which substrate heads an arm gets: **its own isoforms only**, so
the CYP2D6 arm gets a CYP2D6 substrate head and nothing else. Handing its trunk three
foreign substrate heads would quietly undo the split it exists to make.

The substrate heads
-------------------
Source: Ni, Y.-H. *et al.* "Curated CYP450 Interaction Dataset", *Scientific Data* **12**,
1427 (2025), doi:10.1038/s41597-025-05753-8; data doi:10.6084/m9.figshare.26630515,
CC BY 4.0. Loaded through ``cyp_substrate_aux.py``; full appraisal and the LightGBM
precedent in ``Articles/Ni2025_CYP450_substrate_dataset_notes.md``.

**A different kind of auxiliary task from the log2fc heads.** log2fc adds *columns* to
compounds we already have; the substrate labels add *rows*. Of 3,147 substrate compounds,
135 match a challenge training compound exactly and **zero** match anything in the blinded
test set. So matching compounds get their labels attached to the existing row — the only
place the network sees the two properties co-vary on one molecule — and the other 2,826
are appended as new datapoints with NaN in every pIC50 and log2fc column, which chemprop
masks out of the loss.

Substrate is not inhibition. Ni et al. label whether an isoform *metabolises* a compound;
the challenge asks how strongly it is *inhibited*. The two can be anti-correlated, so the
heads are auxiliary and nothing here lets a substrate label become a pIC50 value. The bet
is narrow: both properties hinge on fitting the same active site, so a trunk forced to
also separate substrates from non-substrates may hand the pIC50 heads a better
representation.

No target rescaling is needed — ``normalize_targets()`` standardises each head separately,
so a 0/1 head reaches the loss as +/-1 in the same units as everything else.

What "censoring off" covers
---------------------------
``USE_CENSORED_LOSS = False`` now means off *everywhere*:

  * real pIC50 values below the top screening concentration are point targets, not upper
    bounds — this is what the flag controlled in the ``.baseline``;
  * **the proxy model's pseudo-labels too.** The ``.baseline`` censored those in
    ``build_proxy_labels`` through a line that never consulted the flag, so setting it
    False there would have left roughly half the proxy labels quietly still censored.
    That line is now gated. This is the one behavioural difference you would not have
    predicted from the flag name, which is why it is called out here.

No ``CENSOR_NONHITS``: the ``.baseline`` predates it, and it is not ported. That matters
more than it sounds. Under the current head of the parent script, 58-71% of each pIC50
head is a bound rather than a measurement, and most of those bounds are *synthesised* from
screening non-hits (2,821 of CYP2D6's 4,313 "labels"). Here every pIC50 label is a real
DRC measurement: 1,412 / 1,285 / 1,492 / 2,333 for 1A2 / 2C9 / 2D6 / 3A4. Far fewer labels,
all of them real.

  * **Inverse-CI weighting** is kept. chemprop's per-datapoint ``weight`` is a scalar
    across all tasks, so the per-task inverse-CI weights are averaged over the isoforms a
    compound actually has measured. Coarser than ``cyp_label_utils.ci_weights``, but the
    same idea and it costs nothing.

Scoring is the challenge's own ST-RAE (imported from ``evaluation/`` when this script
sits in the repo, otherwise the identical local copy below), on Bemis-Murcko scaffold
folds — the split that ranks modelling choices honestly even though it flatters nobody.

Data (CSV, from ``LOCAL_DATA_DIR`` if present, else straight from Hugging Face):
    cyp-challenge-TRAIN_inhibition.csv           4,905 compounds, DRC pIC50 + CI + std
    cyp-challenge-single-concentration-TRAIN.csv 17,504 rows = 4,376 compounds x 4 iso
    cyp-challenge-TEST-BLINDED.csv               750 compounds, Molecule_Name + SMILES
    data/external/ni2025_cyp450/<ISO>_{training,testing}set.csv
                                                 3,147 compounds, substrate labels

Usage:
    conda activate chemprop            # chemprop >= 2.1 (developed against 2.2.x)
    python cyp_substrate_aux.py --download        # one-off, ~1.6 MB from Figshare
    python chemprop_cyp_pic50_uncensored_arms_substrate_aux_ensemble.py

    # offline VM: copy the CSVs over from a connected machine, then
    export CYP_DATA_DIR=/data/cyp-challenge
    export CYP_AUX_DIR=/data/ni2025_cyp450

Runtime is dominated by the fold loops. Note the arm split multiplies the primary fits by
len(TASK_ARMS) = 2. On one modern GPU expect roughly:
    grid search   len(PARAM_GRID combos) x N_FOLDS x len(TASK_ARMS) fits
    OOF stage     N_FOLDS x (len(TASK_ARMS) + 1)                    fits
    final         len(PRIMARY_SEEDS) x len(TASK_ARMS) + len(PROXY_SEEDS) fits
With the defaults below (grid search off) that is 15 + 13 = 28 fits.
CV results are cached to CV_RESULTS_PATH and reloaded if the file already exists, so a
killed grid search resumes.

The A/B for the substrate heads is one flag: AUX_SUBSTRATE = False leaves everything else
in place. Note that this is *not* a comparison against the .baseline, which also censored
and did not split CYP2D6 out — it isolates the substrate data alone.
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

from cyp_substrate_aux import load_substrate_labels  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

# ── Data ───────────────────────────────────────────────────────────────────────
LOCAL_DATA_DIR = Path(os.environ.get("CYP_DATA_DIR", PROJECT_ROOT / "data"))
HF_PREFIX = "hf://datasets/openadmet/cyp-challenge-train-test"

FILE_TRAIN_DRC = "cyp-challenge-TRAIN_inhibition.csv"
FILE_TRAIN_SINGLE = "cyp-challenge-single-concentration-TRAIN.csv"
FILE_TEST = "cyp-challenge-TEST-BLINDED.csv"

AUX_DATA_DIR = Path(
    os.environ.get("CYP_AUX_DIR", PROJECT_ROOT / "data" / "external" / "ni2025_cyp450")
)

# ── Outputs ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path(
    os.environ.get(
        "CYP_OUTPUT_DIR", PROJECT_ROOT / "outputs" / "chemprop_uncensored_substrate_aux"
    )
)
CV_RESULTS_PATH = OUTPUT_DIR / "cv_grid_results.csv"
CALIBRATION_PATH = OUTPUT_DIR / "single_dose_calibration.csv"
OOF_PRED_PATH = OUTPUT_DIR / "oof_predictions.csv"
OOF_SCORE_PATH = OUTPUT_DIR / "oof_scores.csv"
BLEND_PATH = OUTPUT_DIR / "blend_alpha_sweep.csv"
ENSEMBLE_DIR = OUTPUT_DIR / "models"
SUBMISSION_PATH = OUTPUT_DIR / "my_chemprop_uncensored_substrate_aux_submission.csv"
TEST_DETAIL_PATH = OUTPUT_DIR / "test_predictions_detail.csv"
KEPT_DESCS_PATH = OUTPUT_DIR / "kept_descriptors.txt"
AUX_REPORT_PATH = OUTPUT_DIR / "substrate_aux_report.csv"

# ── Endpoints ──────────────────────────────────────────────────────────────────
ISOFORMS = ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")
TARGET_COLS = [f"{iso}_pIC50_direct_inhibition" for iso in ISOFORMS]
LOG2FC_COLS = [f"{iso}_log2fc_estimate" for iso in ISOFORMS]
HIT_COLS = [f"{iso}_is_hit" for iso in ISOFORMS]
N_ISO = len(ISOFORMS)

SUBSTRATE_COL = "{iso}_is_substrate"

# ── Assay constants (challenge definitions) ────────────────────────────────────
TOP_CONC_PIC50 = -float(np.log10(49.5e-6))  # 4.305 — top screening concentration
HIT_LOG2FC = -1.0
HIT_FDR = 0.05

# ── Label handling ─────────────────────────────────────────────────────────────
AUX_TASK_WEIGHT = 0.2  # weight of each log2fc head relative to a pIC50 head
# OFF. The censored loss did not help the non-TDI task, which is the reason this script
# starts from the .baseline instead of the current head of the parent. False here means
# off everywhere, including build_proxy_labels — see "What 'censoring off' covers" in the
# module docstring. Flip to True to restore the .baseline's behaviour exactly.
USE_CENSORED_LOSS = False  # BoundedMSE + lt_mask on pIC50 < TOP_CONC_PIC50
USE_CI_WEIGHTS = True  # per-compound inverse credible-interval weighting
CI_WEIGHT_FLOOR = 1.0  # keeps a zero-width band from dominating
PSEUDO_PIC50_CLIP = (2.0, 8.0)  # calibrated pseudo-labels are clipped to this range
PRED_CLIP = (2.5, 9.0)  # final predictions clipped to a physically sane range

# ── Task arms — CYP2D6 is not in the multitask ─────────────────────────────────
# CYP2D6 is uncorrelated with the other three on co-measured compounds (Spearman 0.06,
# -0.12, 0.06, versus 0.38-0.69 among 1A2/2C9/3A4), and 55% of its labelled compounds
# carry no other isoform label at all — so multitask sharing has nothing to share. It
# shows: OOF pred_sd/true_sd is 0.49 for 2D6 against 0.98 for 3A4, i.e. the head is
# collapsing toward the mean inside a trunk optimised for a signal it does not have.
# Each arm is a separate model; predictions are merged back per isoform.
# Set to (ISOFORMS,) to recover the .baseline's single joint model.
TASK_ARMS = (("CYP1A2", "CYP2C9", "CYP3A4"), ("CYP2D6",))

# ── Substrate auxiliary heads (Ni et al. 2025) ─────────────────────────────────
# See the module docstring for the rationale and the leakage argument.
# False leaves the arm split and the uncensored loss in place and removes only the
# substrate data, which is the A/B for this dataset.
AUX_SUBSTRATE = True

# Isoforms whose substrate labels are loaded. Only those that also appear in a task arm
# become heads on that arm, unless listed in AUX_SUBSTRATE_EXTRA_ISOFORMS.
AUX_SUBSTRATE_ISOFORMS = ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")

# Non-scored isoforms added as substrate heads to *every* arm. ("CYP2C19", "CYP2E1") is
# the natural next experiment: more auxiliary signal, no scored endpoint, and CYP2C19 is
# close kin to CYP2C9. Off by default — it also lands foreign heads on the CYP2D6 arm,
# which is the arm that exists specifically to avoid foreign signal.
AUX_SUBSTRATE_EXTRA_ISOFORMS: tuple[str, ...] = ()

# Loss weight of one substrate head relative to a pIC50 head. Below AUX_TASK_WEIGHT
# because the log2fc heads are a *measured readout of the same assay*, whereas a
# substrate label is a different property on different compounds.
AUX_SUBSTRATE_TASK_WEIGHT = 0.1

# Per-datapoint weight of an appended substrate-only row, against 1.0 for a real DRC row.
# Compounds carrying both a pIC50 and a substrate label keep their normal inverse-CI
# weight — they are not appended rows, they are existing rows with extra columns filled.
AUX_SUBSTRATE_ROW_WEIGHT = 0.3

# Drop substrate rows sharing a scaffold (or exact structure) with the held-out fold.
# On by default. This makes CV pessimistic on purpose relative to the final refit, which
# keeps every row; off makes it dishonest. See the module docstring.
AUX_DROP_FOLD_LEAKAGE = True

# The same filter against the blinded test set, for the final refit. Off by default and it
# should stay off: substrate labels are public data containing no challenge answers, so a
# test compound appearing in training with a substrate label is ordinary transductive use.
# Provided only for a strictly inductive run.
AUX_DROP_TEST_LEAKAGE = False

# Keep substrate compounds inside the heavy-atom range the challenge spans. The Ni set is
# a *metabolism* dataset: it runs from 1 to 295 heavy atoms and its 99th percentile (119)
# is three times the largest test compound, because endogenous CoA thioesters and
# nucleotides are in there.
#
# The range is train UNION test (5-65), not test alone (12-39). Test alone is the tempting
# choice and it is wrong: it throws away 10 substrate compounds that *are* challenge
# training compounds — molecules the model is fitted on with a real pIC50, so in-domain by
# construction — plus 365 others the training set's own size range covers. Training
# compounds are never size-filtered, so filtering the auxiliary set more tightly than the
# primary set has no justification.
AUX_FIT_SIZE_RANGE = True

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
# The final refit has no validation split, so EarlyStopping and BestStateTracker never
# attach (fit_mpnn only builds callbacks when val_dset is not None). The .baseline then
# trained a flat max(CV_MAX_EPOCHS * 1.2, 60) = 60 epochs while CV was putting the optimum
# at 8-10 — a refit six times past the point the folds said to stop.
#
# This is the ONE place this script does not follow the .baseline, and it is a bug fix
# rather than a design change: the budget now comes from the folds, scaled up a little for
# the ~25% extra data a full-data refit sees. It matters more here than it did in the
# parent, because with censoring off there are far fewer labels to fit and correspondingly
# more room to overfit them. Set FINAL_EPOCHS_LEGACY = True to restore the flat 60 and get
# a like-for-like comparison against an old .baseline run.
FINAL_EPOCHS_LEGACY = False
FINAL_EPOCH_SCALE = 1.3
FINAL_EPOCHS_FLOOR = 8
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
# Off by default. The parent script's sweep over this same grid came back flat —
# mean_val_loss spanned 0.2275-0.2296 (<1%) and macro ST-RAE 0.734-0.755 across all four
# combos — so hyperparameters are not where the error is, and the arm split doubles the
# cost of re-discovering that tie (4 combos x 5 folds x 2 arms = 40 fits). Set True to
# sweep anyway; results cache to CV_RESULTS_PATH.
RUN_GRID_SEARCH = False
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
# Substrate auxiliary data (Ni et al. 2025)
# ══════════════════════════════════════════════════════════════════════════════

def scaffold_of(mol) -> str:
    """Bemis-Murcko scaffold SMILES; ``""`` for acyclic molecules and any failure.

    An empty scaffold is not an identity — every acyclic molecule shares it — so callers
    matching scaffolds between two sets must exclude ``""`` and fall back to exact
    structure. ``aux_overlap_mask`` does both.
    """
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:
        return ""


class SubstrateAux:
    """Substrate-labelled compounds, prepared exactly like the training compounds.

    Structures go through the same ``mol_from_smiles`` standardisation as the challenge
    set, and ``smiles`` / ``scaffolds`` are recomputed *from the standardised mol* — not
    carried over from the source file — so that matching against the training set compares
    like with like. Getting that wrong would silently break both the attach step and the
    leakage filter, in the direction of finding no overlap where overlap exists.
    """

    def __init__(self, mols, smiles, scaffolds, labels, x_d_raw, isoforms):
        self.mols = mols
        self.smiles = np.asarray(smiles)
        self.scaffolds = np.asarray(scaffolds)
        self.labels = labels.reset_index(drop=True)
        self.x_d_raw = x_d_raw
        self.isoforms = list(isoforms)

    def __len__(self) -> int:
        return len(self.mols)

    def subset(self, mask: np.ndarray) -> "SubstrateAux":
        mask = np.asarray(mask, dtype=bool)
        return SubstrateAux(
            [m for m, k in zip(self.mols, mask) if k],
            self.smiles[mask],
            self.scaffolds[mask],
            self.labels.loc[mask],
            self.x_d_raw[mask],
            self.isoforms,
        )


def load_aux_substrate(
    aux_dir: Path, isoforms, test_mols: list, train_mols: list, col_mask: np.ndarray
) -> SubstrateAux:
    """Load, standardise, filter and featurize the substrate compounds.

    Filtering mirrors what the challenge compounds get, and no more: the test-set element
    whitelist (which the training set was itself filtered to, in ``structural_filter``),
    and the heavy-atom range spanned by training and test together. A compound the model
    could never be asked about contributes nothing but gradient noise; a compound inside
    the challenge's own size range does not deserve to be held to a stricter standard than
    the training set it will sit beside.
    """
    wide = load_substrate_labels(aux_dir, list(isoforms), verbose=True)

    mols, keep = build_mols(wide["SMILES"].tolist())
    labels = wide.loc[keep].reset_index(drop=True)
    n_start = len(mols)

    allowed = {a.GetAtomicNum() for m in test_mols for a in m.GetAtoms()}
    sizes = [m.GetNumHeavyAtoms() for m in list(test_mols) + list(train_mols)]
    lo, hi = (min(sizes), max(sizes)) if AUX_FIT_SIZE_RANGE else (0, 10**9)

    ok = np.ones(len(mols), dtype=bool)
    n_element = n_size = 0
    for i, mol in enumerate(mols):
        if any(a.GetAtomicNum() not in allowed for a in mol.GetAtoms()):
            ok[i] = False
            n_element += 1
        elif not (lo <= mol.GetNumHeavyAtoms() <= hi):
            ok[i] = False
            n_size += 1
    mols = [m for m, k in zip(mols, ok) if k]
    labels = labels.loc[ok].reset_index(drop=True)

    x_aux = compute_rdkit_descriptors(mols)[:, col_mask]
    finite = np.all(np.isfinite(x_aux), axis=1)
    if not finite.all():
        mols = [m for m, k in zip(mols, finite) if k]
        labels = labels.loc[finite].reset_index(drop=True)
        x_aux = x_aux[finite]

    print(
        f"  substrate compounds: {n_start} parsed -> {len(mols)} kept "
        f"({n_element} off the test element whitelist, {n_size} outside the challenge "
        f"size range [{lo}, {hi}] heavy atoms, {int((~finite).sum())} with bad descriptors)"
    )
    return SubstrateAux(
        mols,
        [Chem.MolToSmiles(m) for m in mols],
        [scaffold_of(m) for m in mols],
        labels[[SUBSTRATE_COL.format(iso=i) for i in isoforms]],
        x_aux,
        isoforms,
    )


def attach_substrate_labels(
    df: pd.DataFrame, prim_smiles: np.ndarray, aux: SubstrateAux
) -> tuple[pd.DataFrame, SubstrateAux]:
    """Merge substrate labels onto matching training rows; return the rest as new rows.

    A compound in both sets must not become two datapoints — that would double its pull on
    the shared trunk and hide the one thing this dataset can uniquely teach, which is how
    the two properties co-vary on a single molecule. 135 of the 3,147 substrate compounds
    land here; the rest are appended.

    Matching is on canonical SMILES of the *standardised* molecule on both sides, so the
    largest-fragment and neutralisation steps cannot make the same compound look like two.
    """
    df = df.copy()
    cols = [SUBSTRATE_COL.format(iso=i) for i in aux.isoforms]
    for col in cols:
        df[col] = np.nan

    lookup: dict[str, int] = {}
    for i, smi in enumerate(aux.smiles):
        lookup.setdefault(smi, i)

    rows = np.array([lookup.get(s, -1) for s in prim_smiles])
    hit = rows >= 0
    if hit.any():
        matched = aux.labels.iloc[rows[hit]]
        for col in cols:
            df.loc[hit, col] = matched[col].to_numpy()

    attached = np.zeros(len(aux), dtype=bool)
    attached[rows[hit]] = True
    n_labels = int(df[cols].notna().to_numpy().sum())
    print(
        f"  attached to {int(hit.sum())} existing training compounds "
        f"({n_labels} substrate labels); {int((~attached).sum())} substrate compounds "
        f"will be appended as new rows"
    )
    return df, aux.subset(~attached)


def aux_overlap_mask(aux: SubstrateAux, holdout_mols: list) -> np.ndarray:
    """True where a substrate compound overlaps the hold-out by scaffold or structure.

    Matching on scaffold alone would miss acyclic molecules, whose scaffold is ``""``, and
    matching on ``""`` itself would drop every acyclic compound in the set. Exact
    canonical structure covers that case.
    """
    scaffolds = {scaffold_of(m) for m in holdout_mols}
    scaffolds.discard("")
    structures = {Chem.MolToSmiles(m) for m in holdout_mols}
    return np.isin(aux.scaffolds, list(scaffolds)) | np.isin(aux.smiles, list(structures))


def substrate_isoforms_for_arm(arm: tuple, aux: SubstrateAux | None) -> list[str]:
    """Which substrate heads this arm carries: its own isoforms, plus any extras.

    Restricting to the arm's own isoforms is what keeps the CYP2D6 arm honest — it exists
    because CYP2D6 shares no signal with the other three, so its trunk should not be
    fitting their substrate labels either.
    """
    if aux is None:
        return []
    wanted = list(arm) + [i for i in AUX_SUBSTRATE_EXTRA_ISOFORMS if i not in arm]
    return [i for i in wanted if i in aux.isoforms]


def build_aux_labels(aux: SubstrateAux, k: int, sub_isos: list[str]) -> np.ndarray:
    """(m x 2k+s) target block for appended substrate rows: NaN everywhere but the
    substrate heads, which chemprop masks out of the loss."""
    y = np.full((len(aux), 2 * k + len(sub_isos)), np.nan)
    for j, iso in enumerate(sub_isos):
        y[:, 2 * k + j] = aux.labels[SUBSTRATE_COL.format(iso=iso)].to_numpy(dtype=float)
    return y


def append_aux_rows(mols, y, w, x_d, lt, aux: SubstrateAux, scaler, k, sub_isos):
    """Concatenate the substrate rows onto one arm's training tensors.

    Descriptors are standardised with the scaler fitted on the *primary* training fold, so
    adding auxiliary compounds cannot shift the standardisation the primary rows are
    judged against.
    """
    if aux is None or len(aux) == 0 or not sub_isos:
        return mols, y, w, x_d, lt
    y_aux = build_aux_labels(aux, k, sub_isos)
    return (
        list(mols) + list(aux.mols),
        np.vstack([y, y_aux]),
        np.concatenate([w, np.full(len(aux), AUX_SUBSTRATE_ROW_WEIGHT)]),
        np.vstack([x_d, scale_descriptors(scaler, aux.x_d_raw)]),
        np.vstack([lt, np.zeros_like(y_aux, dtype=bool)]),
    )


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

def build_primary_labels(
    df: pd.DataFrame, arm: tuple = ISOFORMS, sub_isos: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Y (n x 2k+s), lt_mask, per-compound weights (n,) for one task arm.

    ``arm`` is the tuple of isoforms this model covers. Columns 0..k-1 are the real DRC
    pIC50s (NaN where unmeasured — chemprop masks those out of the loss per task),
    columns k..2k-1 the matching single-dose log2fc readouts, and columns 2k..2k+s-1 the
    substrate heads for ``sub_isos`` — mostly NaN here, since only the 134 training
    compounds that also appear in the Ni set carry a substrate label. The bulk of those
    heads' data arrives as appended rows via ``append_aux_rows``.

    No pseudo-labels: those live in the proxy model. No synthesised censoring bounds
    either — every pIC50 value in here is a real dose-response measurement.
    """
    n = len(df)
    k = len(arm)
    sub_isos = list(sub_isos or [])
    y = np.full((n, 2 * k + len(sub_isos)), np.nan)
    lt = np.zeros((n, 2 * k + len(sub_isos)), dtype=bool)
    w_parts, w_counts = np.zeros(n), np.zeros(n)

    for j, iso in enumerate(arm):
        col = f"{iso}_pIC50_direct_inhibition"
        pic50 = df[col].to_numpy(dtype=float)
        y[:, j] = pic50
        y[:, k + j] = df[f"{iso}_log2fc_estimate"].to_numpy(dtype=float)

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

    for j, iso in enumerate(sub_isos):
        col = SUBSTRATE_COL.format(iso=iso)
        if col in df.columns:
            y[:, 2 * k + j] = df[col].to_numpy(dtype=float)

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
        # The .baseline censored here unconditionally, ignoring USE_CENSORED_LOSS, so
        # turning the flag off still left roughly half the proxy labels as upper bounds.
        # Gated now, so "censoring off" means off in both models.
        if USE_CENSORED_LOSS:
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


def task_weight_vector(
    n_tasks: int, n_primary: int | None = None, n_substrate: int = 0
) -> list[float]:
    """1.0 per pIC50 head, AUX_TASK_WEIGHT per log2fc head, AUX_SUBSTRATE_TASK_WEIGHT
    per substrate head.

    ``n_primary`` is the number of pIC50 heads in this arm; with the arm split it is no
    longer always N_ISO. Defaults to half the non-substrate heads when the layout is the
    paired pIC50 + log2fc one, else to all of them.
    """
    n_paired = n_tasks - n_substrate
    if n_primary is None:
        n_primary = n_paired // 2 if n_paired % 2 == 0 else n_paired
    weights = [1.0] * n_primary + [AUX_TASK_WEIGHT] * (n_paired - n_primary)
    return weights + [AUX_SUBSTRATE_TASK_WEIGHT] * n_substrate


def build_mpnn(
    n_descriptors: int,
    n_tasks: int,
    target_scaler,
    mp_depth: int,
    mp_hidden_dim: int,
    ffn_hidden_dim: int,
    ffn_n_layers: int,
    dropout: float,
    n_primary: int | None = None,
    n_substrate: int = 0,
) -> models.MPNN:
    feat = featurizers.SimpleMoleculeMolGraphFeaturizer()
    mp = nn.BondMessagePassing(
        d_v=feat.atom_fdim, d_e=feat.bond_fdim, depth=mp_depth, d_h=mp_hidden_dim
    )
    agg = nn.MeanAggregation()
    weights = task_weight_vector(n_tasks, n_primary, n_substrate)
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
    n_primary: int | None = None,
    n_substrate: int = 0,
) -> tuple[models.MPNN, pl.Trainer, BestStateTracker | None]:
    """Fit one MPNN. Targets are standardised per task, so the pIC50, log2fc and 0/1
    substrate heads all contribute comparably and ``task_weights`` means what it says.

    That per-task standardisation is why the substrate heads need no manual rescaling
    here, where the LightGBM version of this experiment had to map 0/1 onto the pIC50
    scale by hand to keep L2 split gain comparable across stacked rows.
    """
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

    mpnn = build_mpnn(
        n_descriptors, train_dset.Y.shape[1], target_scaler,
        n_primary=n_primary, n_substrate=n_substrate, **params,
    )
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
    params: dict, seed: int = SCAFFOLD_SEED, arm: tuple = ISOFORMS,
    aux: SubstrateAux | None = None,
) -> tuple[np.ndarray, float, int]:
    """Train one arm on every fold but ``fold``; return its predictions for the held-out
    fold (columns follow ``arm``), the best validation loss, and the epoch it happened at.

    Appended substrate rows join the *training* dataset only. The validation fold stays
    exactly the set of challenge compounds assigned to it, so early stopping and the OOF
    score keep measuring the same population the leaderboard does. (Substrate labels
    attached to a challenge compound do reach the validation loss, at
    AUX_SUBSTRATE_TASK_WEIGHT — the same treatment the log2fc heads already get. Scoring
    is unaffected either way: ``predict`` slices out the pIC50 columns.)
    """
    tr, va = folds != fold, folds == fold
    sub_isos = substrate_isoforms_for_arm(arm, aux)
    y, lt, w = build_primary_labels(df, arm=arm, sub_isos=sub_isos)
    k = len(arm)

    scaler = StandardScaler().fit(x_d_raw[tr])
    x_tr = scale_descriptors(scaler, x_d_raw[tr])
    x_va = scale_descriptors(scaler, x_d_raw[va])
    mols_tr = [m for m, keep in zip(mols, tr) if keep]
    mols_va = [m for m, keep in zip(mols, va) if keep]

    y_tr, w_tr, lt_tr = y[tr], w[tr], lt[tr]
    if aux is not None and sub_isos:
        aux_fold = aux
        if AUX_DROP_FOLD_LEAKAGE:
            aux_fold = aux.subset(~aux_overlap_mask(aux, mols_va))
        mols_tr, y_tr, w_tr, x_tr, lt_tr = append_aux_rows(
            mols_tr, y_tr, w_tr, x_tr, lt_tr, aux_fold, scaler, k, sub_isos
        )

    train_dset = make_dataset(mols_tr, y_tr, w_tr, x_tr, lt_tr)
    val_dset = make_dataset(mols_va, y[va], w[va], x_va, lt[va])

    mpnn, trainer, tracker = fit_mpnn(
        train_dset, val_dset, x_tr.shape[1], params,
        max_epochs=CV_MAX_EPOCHS, patience=CV_PATIENCE, seed=seed,
        n_primary=k, n_substrate=len(sub_isos),
    )
    preds = predict(mpnn, trainer, val_dset)[:, :k]
    return preds, tracker.best_val_loss, tracker.best_epoch


def run_primary_fold_all_arms(
    fold: int, folds: np.ndarray, mols: list, df: pd.DataFrame, x_d_raw: np.ndarray,
    params: dict, seed: int = SCAFFOLD_SEED, aux: SubstrateAux | None = None,
) -> tuple[np.ndarray, float, int]:
    """Run every arm for one fold and merge the columns back into isoform order."""
    va = folds == fold
    out = np.full((int(va.sum()), N_ISO), np.nan)
    losses, epochs = [], []
    for arm in TASK_ARMS:
        preds, loss, epoch = run_primary_fold(
            fold, folds, mols, df, x_d_raw, params, seed=seed, arm=arm, aux=aux
        )
        for a, iso in enumerate(arm):
            out[:, ISOFORMS.index(iso)] = preds[:, a]
        losses.append(loss)
        epochs.append(epoch)
    return out, float(np.mean(losses)), int(np.ceil(np.mean(epochs)))


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


def grid_search(folds, mols, df, x_d_raw, aux: SubstrateAux | None = None) -> dict:
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
            preds, loss, epoch = run_primary_fold_all_arms(
                fold, folds, mols, df, x_d_raw, params, aux=aux
            )
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
    print(f"{_BANNER}\nCYP direct-inhibition — uncensored, arm-split, + substrate aux")
    print(f"outputs -> {OUTPUT_DIR}   ST-RAE from {_SCORER_SOURCE}")
    print(f"censored loss: {'ON' if USE_CENSORED_LOSS else 'OFF'}   task arms: {TASK_ARMS}")
    print(f"substrate auxiliary heads: {'ON' if AUX_SUBSTRATE else 'OFF'}")
    print(_BANNER)

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

    # ── Step 2b — substrate auxiliary data ────────────────────────────────────
    aux = None
    if AUX_SUBSTRATE:
        print("\nStep 2b — substrate auxiliary data (Ni et al. 2025)")
        try:
            aux = load_aux_substrate(
                AUX_DATA_DIR, AUX_SUBSTRATE_ISOFORMS, test_mols, mols, col_mask
            )
        except FileNotFoundError as exc:
            raise SystemExit(
                f"{exc}\n\nRun `python cyp_substrate_aux.py --download` first, or point "
                f"CYP_AUX_DIR at a copy of the CSVs. Set AUX_SUBSTRATE = False to run "
                f"the baseline without them."
            ) from exc

        prim_smiles = np.array([Chem.MolToSmiles(m) for m in mols])
        df, aux = attach_substrate_labels(df, prim_smiles, aux)

        test_smiles = {Chem.MolToSmiles(m) for m in test_mols}
        n_test_overlap = int(np.isin(aux.smiles, list(test_smiles)).sum())
        print(f"  substrate compounds also in the blinded test set: {n_test_overlap}")
        if AUX_DROP_TEST_LEAKAGE:
            aux = aux.subset(~aux_overlap_mask(aux, test_mols))
            print(f"  AUX_DROP_TEST_LEAKAGE on -> {len(aux)} substrate rows kept")

        rows = []
        for iso in aux.isoforms:
            col = SUBSTRATE_COL.format(iso=iso)
            appended = aux.labels[col]
            attached = df[col]
            rows.append(dict(
                isoform=iso,
                appended_rows=int(appended.notna().sum()),
                appended_substrate=int((appended == 1).sum()),
                attached_to_training=int(attached.notna().sum()),
                attached_substrate=int((attached == 1).sum()),
            ))
        aux_report = pd.DataFrame(rows)
        print("\n  substrate label coverage")
        print(aux_report.to_string(index=False))
        aux_report.to_csv(AUX_REPORT_PATH, index=False)
        for arm in TASK_ARMS:
            print(f"  arm {arm} -> substrate heads {substrate_isoforms_for_arm(arm, aux)}")

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
        best_params = coerce_params(grid_search(folds, mols, df, x_d_raw, aux=aux))
    else:
        best_params = coerce_params(DEFAULT_PARAMS)
    print(f"\nHyperparameters: {best_params}")
    print(f"Task arms: {TASK_ARMS}")

    # ── Step 5 — out-of-fold arms and blend weights ───────────────────────────
    print(f"\n{_BANNER}\nStep 5 — out-of-fold comparison of the three arms\n{_BANNER}")
    oof_primary = np.full((len(df), N_ISO), np.nan)
    oof_proxy = np.full((len(df), N_ISO), np.nan)
    fold_epochs = []
    for fold in range(N_FOLDS):
        print(f"\n  fold {fold + 1}/{N_FOLDS}")
        preds, loss, epoch = run_primary_fold_all_arms(
            fold, folds, mols, df, x_d_raw, best_params, aux=aux
        )
        oof_primary[folds == fold] = preds
        fold_epochs.append(epoch)
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
    # No validation split here, so no early stopping can fire: the budget has to come
    # from the folds. See FINAL_EPOCHS_LEGACY — the .baseline's flat 60 was six times the
    # CV optimum.
    if FINAL_EPOCHS_LEGACY:
        final_epochs = max(int(CV_MAX_EPOCHS * 1.2), 60)
        print(f"  final epochs: {final_epochs} (legacy fixed budget)")
    else:
        final_epochs = max(
            FINAL_EPOCHS_FLOOR, int(round(float(np.mean(fold_epochs)) * FINAL_EPOCH_SCALE))
        )
        print(f"  final epochs: {final_epochs}  "
              f"(mean CV best_epoch {np.mean(fold_epochs):.1f} x {FINAL_EPOCH_SCALE})")

    primary_test_preds = []
    for i, seed in enumerate(PRIMARY_SEEDS, start=1):
        print(f"\n  primary member {i}/{len(PRIMARY_SEEDS)}  seed={seed}")
        per_arm = np.full((len(test_mols), N_ISO), np.nan)
        for arm in TASK_ARMS:
            sub_isos = substrate_isoforms_for_arm(arm, aux)
            y, lt, w = build_primary_labels(df, arm=arm, sub_isos=sub_isos)
            k = len(arm)
            n_tasks = y.shape[1]
            # Every substrate row is kept here — unlike in CV there is no hold-out to
            # protect, and the labels carry no challenge answers.
            fit_mols, y_fit, w_fit, x_fit, lt_fit = append_aux_rows(
                mols, y, w, x_d_train, lt, aux, final_scaler, k, sub_isos
            )
            train_dset = make_dataset(fit_mols, y_fit, w_fit, x_fit, lt_fit)
            mpnn, trainer, _ = fit_mpnn(
                train_dset, None, n_descriptors, best_params,
                max_epochs=final_epochs, patience=CV_PATIENCE, seed=seed, progress=True,
                n_primary=k, n_substrate=len(sub_isos),
            )
            tag = "-".join(a.replace("CYP", "") for a in arm)
            torch.save(mpnn, ENSEMBLE_DIR / f"primary_{tag}_seed{seed}.pt")
            test_dset = make_dataset(
                test_mols, np.full((len(test_mols), n_tasks), np.nan),
                np.ones(len(test_mols)), x_d_test,
            )
            arm_pred = predict(mpnn, trainer, test_dset)[:, :k]
            for a, iso in enumerate(arm):
                per_arm[:, ISOFORMS.index(iso)] = arm_pred[:, a]
        primary_test_preds.append(per_arm)

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
    print(f"  censored loss          : {'ON' if USE_CENSORED_LOSS else 'OFF'}")
    print(f"  task arms              : {TASK_ARMS}")
    if AUX_SUBSTRATE:
        print(f"  substrate aux          : {len(aux)} appended rows, task weight "
              f"{AUX_SUBSTRATE_TASK_WEIGHT}, row weight {AUX_SUBSTRATE_ROW_WEIGHT}, "
              f"fold-leakage filter {'on' if AUX_DROP_FOLD_LEAKAGE else 'OFF'}")
        print("  To judge whether the substrate heads earned their place, rerun with "
              "AUX_SUBSTRATE = False and compare this MA ST-RAE. N_FOLDS folds under one "
              "SCAFFOLD_SEED share a split and give no error bar — vary the seed before "
              "believing a gap smaller than about 0.02.")
    else:
        print("  substrate aux          : OFF")
    print("  Scaffold-split numbers are pessimistic against an analog-expansion test "
          "set — use them to rank choices, not to predict the leaderboard.")


if __name__ == "__main__":
    main()
