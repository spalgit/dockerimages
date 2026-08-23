"""ChemProp multitask CYP time-dependent-inhibition (TDI) classifier.

OpenADMET CYP Inhibition Blind Challenge, TDI track (classification).
Companion to ``chemprop_cyp_pic50_rdkit2d_log2fc_aux_proxy_blend_ensemble.py``,
which handles the Direct Inhibition regression track.

The track asks for a boolean ``is_TDI`` per compound for CYP3A4 and CYP2D6 only.
Primary metric is MCC, macro-averaged over the two isoforms.

────────────────────────────────────────────────────────────────────────────────
WHAT THIS SCRIPT DOES DIFFERENTLY FROM THE TUTORIAL NOTEBOOK
────────────────────────────────────────────────────────────────────────────────

``notebooks/TDI_prediction.ipynb`` trains one LightGBM per isoform on RDKit
descriptors, splits randomly, and thresholds probabilities at 0.5. Three things
in that recipe cost a lot of MCC, and this script changes all three.

1.  **The 0.5 threshold is wrong.** The label is ~21% positive, so a calibrated
    classifier puts very few compounds over 0.5. Measured here on scaffold CV
    with the notebook's own model class:

        CYP3A4   MCC@0.50 = 0.302 (10.4% called positive)
                 MCC@0.11 = 0.365 (30.2% called positive)   +21%
        CYP2D6   MCC@0.50 = 0.017 ( 3.1% called positive)
                 MCC@0.06 = 0.087 (19.9% called positive)   +5x

    MCC is threshold-sensitive under class imbalance. We tune one threshold per
    isoform on out-of-fold predictions (``TUNE_THRESHOLDS``), and by default
    transfer it to the test set by *matching the predicted positive rate* rather
    than the raw cut, because the all-data ensemble is sharper than the CV models
    and its probabilities do not live on the same scale. See ``choose_operating_point``.

2.  **Random splits flatter the model.** The blind set is built by hit expansion
    and 75.5% of it sits on scaffolds absent from training. We use Bemis-Murcko
    scaffold folds, the same as the regression script.

3.  **The label carries far more signal than the two boolean columns.** ``is_TDI``
    is a deterministic function of two pIC50 surfaces — verified exactly, 100% of
    rows, in ``verify_label_rule()``:

        is_TDI  <=>  max(pIC50_TDI_condition, 4.0) - max(pIC50_direct, 4.0) > log10(2)

    and the dataset ships both surfaces for all four isoforms. So we train a
    multitask binary model over 12 heads: the 2 primary ``is_TDI`` labels plus 10
    auxiliary heads binarised off those pIC50 surfaces. The auxiliary heads are
    much denser than the primary ones and carry the shared encoder.

────────────────────────────────────────────────────────────────────────────────
THE CYP3A4 LABELLING CONVENTION — READ THIS BEFORE TRUSTING A CYP3A4 SCORE
────────────────────────────────────────────────────────────────────────────────

Of the 3,584 CYP3A4 ``is_TDI`` labels, **1,249 (34.9%) are assigned negatives**:
compounds with a TDI-condition pIC50 but *no fittable direct-inhibition curve*.
Every one of them is labelled ``False``, even though 84.5% would come out
positive if you substituted the pIC50 floor for the missing direct value.

Those compounds are not weak. Their TDI-condition pIC50 median is 5.40 against
4.63 for the compounds that did produce a direct curve — a failed direct fit at
CYP3A4 usually means the compound inhibited too hard to fit, not too weakly.

The consequence is that CYP3A4 direct potency relates to the label
**non-monotonically**:

    weak, fittable        -> mostly False   (direct pIC50 median 3.82)
    moderate, fittable    -> enriched True  (direct pIC50 median 4.66)
    too potent to fit     -> assigned False (by convention)

A model that learns "potent implies TDI" gets the third block exactly backwards.
That is why ``AUX_TASKS`` includes ``<ISO>_direct_fitted`` — a dense, never-masked
head that lets the encoder represent "will the direct curve fit at all",
separately from potency. Do not remove it without re-measuring CYP3A4.

CYP2D6 has no such block (4 rows) and the relationship there is mild and opposite:
TDI-positives have *lower* direct potency (4.50 vs 4.86).

────────────────────────────────────────────────────────────────────────────────
DATA
────────────────────────────────────────────────────────────────────────────────
From ``LOCAL_DATA_DIR`` if present, else straight from Hugging Face:

    cyp-challenge-TRAIN_TDI.csv        6,145 compounds
                                       CYP3A4_is_TDI  3,584 labels (21.3% positive)
                                       CYP2D6_is_TDI  1,497 labels (21.6% positive)
                                       only 259 compounds carry both
    cyp-challenge-TEST-BLINDED.csv       750 compounds

Note the two label sets barely overlap and agree only 61.8% where they do, so the
multitask coupling between the two primary heads is weak — the auxiliary heads,
not the other isoform, are what make this worth doing as one model.

────────────────────────────────────────────────────────────────────────────────
USAGE
────────────────────────────────────────────────────────────────────────────────
    conda activate chemprop          # chemprop >= 2.1 (developed against 2.2.3)
    python chemprop_cyp_tdi_classifier.py

    # offline VM: download the two CSVs once, then
    export CYP_DATA_DIR=/data/cyp-challenge
    python chemprop_cyp_tdi_classifier.py

    # quick pass, no ensemble
    python chemprop_cyp_tdi_classifier.py --seeds 42 --folds 3

Runtime is dominated by N_FOLDS + len(ENSEMBLE_SEEDS) fits. On one modern GPU
expect roughly 15-25 minutes at the defaults.

Outputs (OUTPUT_DIR):
    my_chemprop_tdi_submission.csv   the file to upload
    oof_predictions.csv              out-of-fold probabilities, both isoforms
    oof_scores.csv                   MCC/accuracy/precision/recall/F1 at the chosen cut
    threshold_sweep.csv              MCC vs threshold, per isoform — inspect this
    test_predictions_detail.csv      probabilities + per-seed spread before thresholding
"""

from __future__ import annotations

import argparse
import os
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
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from chemprop import data, featurizers, models, nn

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore", message=".*does not have many workers.*")
warnings.filterwarnings("ignore", message=".*GPU available but not used.*")
torch.set_float32_matmul_precision("medium")

PROJECT_ROOT = Path(__file__).resolve().parent


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

LOCAL_DATA_DIR = Path(os.environ.get("CYP_DATA_DIR", PROJECT_ROOT / "data"))
HF_PREFIX = "hf://datasets/openadmet/cyp-challenge-train-test"

FILE_TRAIN_TDI = "cyp-challenge-TRAIN_TDI.csv"
FILE_TEST = "cyp-challenge-TEST-BLINDED.csv"

OUTPUT_DIR = Path(os.environ.get("CYP_TDI_OUT", PROJECT_ROOT / "outputs" / "chemprop_tdi"))

# ── Endpoints ──────────────────────────────────────────────────────────────────
TDI_ISOFORMS = ["CYP3A4", "CYP2D6"]          # the only two scored
ALL_ISOFORMS = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
PRIMARY_TASKS = [f"{c}_is_TDI" for c in TDI_ISOFORMS]

# ── Assay constants (challenge definitions) ────────────────────────────────────
PIC50_FLOOR = 4.0                             # below this the assay is unquantifiable
TDI_SHIFT_THRESHOLD = float(np.log10(2))      # a >2-fold shift defines is_TDI
ACTIVITY_CUT = PIC50_FLOOR + TDI_SHIFT_THRESHOLD   # 4.301, "active" for the aux heads

# ── Auxiliary heads ────────────────────────────────────────────────────────────
# All binary, so they share the one BCE criterion. Weighted below 1.0 so they
# shape the encoder without outvoting the two heads we are actually scored on.
USE_AUX_TASKS = True
AUX_TASK_WEIGHT = 0.25

# ── Descriptor block ───────────────────────────────────────────────────────────
USE_DESCRIPTORS = True
DESC_CLIP = 10.0                              # clip standardised descriptors, in sd
DESC_ABS_MAX = 1e10                           # drop descriptors that are routinely 1e20

# ── Cross-validation ───────────────────────────────────────────────────────────
N_FOLDS = 5
SCAFFOLD_SEED = 42
MAX_EPOCHS = 60
PATIENCE = 12
BATCH_SIZE = 64
NUM_WORKERS = 0

# ── Learning rates ─────────────────────────────────────────────────────────────
INIT_LR = 1e-4
MAX_LR = 2e-4
FINAL_LR = 1e-5

# ── Ensemble ───────────────────────────────────────────────────────────────────
ENSEMBLE_SEEDS = [42, 123, 456, 789, 1337]

# ── Hyperparameters ────────────────────────────────────────────────────────────
PARAMS = dict(
    mp_depth=4,
    mp_hidden_dim=300,
    ffn_hidden_dim=600,
    ffn_n_layers=2,
    dropout=0.15,
)

# ── Operating point ────────────────────────────────────────────────────────────
TUNE_THRESHOLDS = True
THRESHOLD_GRID = np.arange(0.02, 0.951, 0.01)
# "rate"  — carry the OOF-optimal *predicted positive rate* over to the test set
#           (robust to the ensemble being sharper than the CV models)
# "prob"  — carry the raw OOF-optimal probability cut over
# Rate matching is the default for the same reason the regression track needed a
# calibration offset: what transfers between a CV model and a full-data ensemble
# is the shape of the ranking, not the absolute scale of the scores.
THRESHOLD_TRANSFER = "rate"
MIN_POSITIVE_RATE = 0.02                      # refuse to emit a degenerate submission
# The MCC optimum can legitimately sit above the label base rate, but a long way above
# usually means the ranking is weak and the sweep is chasing noise. Warn, do not clamp —
# threshold_sweep.csv is the thing to look at when this fires.
POS_RATE_WARN_MULTIPLE = 2.0


# ══════════════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════════════

def read_table(filename: str) -> pd.DataFrame:
    local = LOCAL_DATA_DIR / filename
    if local.exists():
        print(f"  {filename}  <- {local}")
        return pd.read_csv(local)
    url = f"{HF_PREFIX}/{filename}"
    print(f"  {filename}  <- {url}")
    return pd.read_csv(url)


def verify_label_rule(df: pd.DataFrame) -> None:
    """Confirm is_TDI is the floored-shift rule, and report the assigned-negative block.

    This is a guard, not a formality: if a future data release changes the rule, the
    auxiliary heads below stop being the right auxiliary heads and you want to know.
    """
    print("\nLabel definition check")
    for cyp in TDI_ISOFORMS:
        lab, act, dir_ = f"{cyp}_is_TDI", f"{cyp}_pIC50_TDI_condition", f"{cyp}_pIC50_direct_inhibition"
        labelled = df.dropna(subset=[lab])
        both = labelled.dropna(subset=[act, dir_])
        rule = (
            np.maximum(both[act], PIC50_FLOOR) - np.maximum(both[dir_], PIC50_FLOOR)
        ) > TDI_SHIFT_THRESHOLD
        agree = float((rule == both[lab].astype(bool)).mean())
        assigned = labelled[labelled[dir_].isna() & labelled[act].notna()]
        print(
            f"  {cyp}: {len(labelled):5d} labels, {100 * labelled[lab].mean():4.1f}% positive | "
            f"rule reproduces {agree:.3f} on {len(both)} rows with both surfaces"
        )
        if len(assigned):
            print(
                f"          {len(assigned):5d} assigned negatives (no direct fit), "
                f"{100 * len(assigned) / len(labelled):4.1f}% of the label set, "
                f"positive rate {100 * assigned[lab].mean():.1f}%"
            )
        if agree < 0.999:
            print(f"          WARNING: rule no longer exact for {cyp} — revisit AUX_TASKS")


def build_targets(df: pd.DataFrame) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Assemble the multitask binary target matrix.

    Returns ``(Y, task_names, task_weights)``. NaN entries are masked out of the loss
    by chemprop, so a compound contributes only to the heads it actually has.
    """
    cols, names = [], []

    for cyp in TDI_ISOFORMS:                                   # primary, weight 1.0
        cols.append(df[f"{cyp}_is_TDI"].astype(float).to_numpy())
        names.append(f"{cyp}_is_TDI")

    n_primary = len(names)

    if USE_AUX_TASKS:
        for cyp in ALL_ISOFORMS:                               # active under TDI preincubation
            v = df[f"{cyp}_pIC50_TDI_condition"]
            cols.append(np.where(v.isna(), np.nan, (v > ACTIVITY_CUT).astype(float)))
            names.append(f"{cyp}_active_tdi")
        for cyp in ALL_ISOFORMS:                               # active without preincubation
            v = df[f"{cyp}_pIC50_direct_inhibition"]
            cols.append(np.where(v.isna(), np.nan, (v > ACTIVITY_CUT).astype(float)))
            names.append(f"{cyp}_active_direct")
        for cyp in TDI_ISOFORMS:
            # Dense and never masked. This is the head that lets the encoder separate
            # "too potent to fit" from "potent", which is what the CYP3A4 assigned-
            # negative block demands. See the module docstring.
            cols.append(df[f"{cyp}_pIC50_direct_inhibition"].notna().astype(float).to_numpy())
            names.append(f"{cyp}_direct_fitted")

    Y = np.column_stack(cols)
    weights = np.array([1.0] * n_primary + [AUX_TASK_WEIGHT] * (len(names) - n_primary))
    return Y, names, weights


# ══════════════════════════════════════════════════════════════════════════════
# Molecules, descriptors, folds
# ══════════════════════════════════════════════════════════════════════════════

def build_mols(smiles: list[str]) -> tuple[list, np.ndarray]:
    mols, ok = [], []
    for smi in smiles:
        m = Chem.MolFromSmiles(smi)
        ok.append(m is not None)
        if m is not None:
            mols.append(m)
    ok = np.array(ok, dtype=bool)
    if not ok.all():
        print(f"  dropped {int((~ok).sum())} unparseable SMILES")
    return mols, ok


_DESC_NAMES = [d[0] for d in Descriptors._descList]


_DESC_LIST = list(Descriptors._descList)


def compute_rdkit_descriptors(mols: list) -> np.ndarray:
    """One descriptor at a time, so a single failing descriptor costs one cell rather
    than the whole row. ``MolecularDescriptorCalculator`` aborts a row on the first
    exception, which silently emptied the block when a test compound tripped it."""
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

    Selection is made on the training block only. Letting the test set veto a column
    would leak, and a single unusual test compound can otherwise empty the block.
    """
    finite = np.all(np.isfinite(arr), axis=0)
    varied = np.var(np.nan_to_num(arr), axis=0) > 0
    sane = np.nanmax(np.abs(arr), axis=0) < DESC_ABS_MAX
    return finite & varied & sane


def scale_descriptors(scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    """Standardise and clip. Any non-finite cell surviving from an unusual test
    compound becomes 0 after centring, which is the column mean."""
    return np.clip(np.nan_to_num(scaler.transform(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0))), -DESC_CLIP, DESC_CLIP)


def murcko_scaffold_folds(mols: list, n_folds: int, seed: int) -> np.ndarray:
    """Bemis-Murcko scaffold K-fold assignment, one fold index per compound.

    Scaffold groups are shuffled, ordered largest-first, and each is dropped into the
    currently smallest fold. Whole scaffolds stay together, so a held-out fold contains
    chemotypes the training folds never saw.
    """
    import random

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
# Model
# ══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class BestStateTracker(pl.Callback):
    """Keep the weights from the best val_loss epoch — an early-stopped fit otherwise
    ends `patience` epochs past its best."""

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


def build_mpnn(n_descriptors: int, task_weights: np.ndarray, params: dict) -> models.MPNN:
    feat = featurizers.SimpleMoleculeMolGraphFeaturizer()
    mp = nn.BondMessagePassing(
        d_v=feat.atom_fdim,
        d_e=feat.bond_fdim,
        depth=params["mp_depth"],
        d_h=params["mp_hidden_dim"],
    )
    agg = nn.MeanAggregation()
    ffn = nn.BinaryClassificationFFN(
        n_tasks=len(task_weights),
        input_dim=mp.output_dim + n_descriptors,
        hidden_dim=params["ffn_hidden_dim"],
        n_layers=params["ffn_n_layers"],
        dropout=params["dropout"],
        criterion=nn.metrics.BCELoss(task_weights=torch.tensor(task_weights, dtype=torch.float)),
    )
    return models.MPNN(
        mp,
        agg,
        ffn,
        batch_norm=True,
        metrics=[nn.metrics.BinaryAUROC()],
        init_lr=INIT_LR,
        max_lr=MAX_LR,
        final_lr=FINAL_LR,
    )


def make_dataset(mols, Y, x_d) -> data.MoleculeDataset:
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


def fit_mpnn(train_dset, val_dset, n_descriptors, task_weights, params, seed, progress=False):
    """Fit one multitask binary MPNN.

    Note there is deliberately no ``normalize_targets`` call here — the targets are
    already 0/1 and BCE needs them that way. That is the one thing you cannot copy
    over from the regression script.
    """
    set_seed(seed)
    train_loader = data.build_dataloader(
        train_dset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, seed=seed
    )

    callbacks, val_loader, tracker = [], None, None
    if val_dset is not None:
        val_loader = data.build_dataloader(
            val_dset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False
        )
        tracker = BestStateTracker()
        callbacks = [EarlyStopping(monitor="val_loss", patience=PATIENCE, mode="min"), tracker]

    mpnn = build_mpnn(n_descriptors, task_weights, params)
    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=progress,
        enable_model_summary=False,
        accelerator="auto",
        devices=1,
        max_epochs=MAX_EPOCHS,
        callbacks=callbacks,
    )
    trainer.fit(mpnn, train_loader, val_loader)
    if tracker is not None:
        tracker.restore(mpnn)
    return mpnn, trainer


def predict_proba(mpnn, trainer, dset) -> np.ndarray:
    """Positive-class probabilities. BinaryClassificationFFN applies the sigmoid in
    eval mode, so these come back already in [0, 1]."""
    loader = data.build_dataloader(
        dset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False
    )
    mpnn.eval()
    raw = trainer.predict(mpnn, loader)
    return torch.cat(raw).numpy().reshape(len(dset), -1)


# ══════════════════════════════════════════════════════════════════════════════
# Scoring and operating point
# ══════════════════════════════════════════════════════════════════════════════

def classification_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return dict(
        mcc=matthews_corrcoef(y_true, y_pred),
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
    )


def threshold_sweep(y_true: np.ndarray, proba: np.ndarray) -> pd.DataFrame:
    rows = []
    for t in THRESHOLD_GRID:
        pred = (proba >= t).astype(int)
        rows.append(dict(threshold=float(t), pos_rate=float(pred.mean()), **classification_scores(y_true, pred)))
    return pd.DataFrame(rows)


def choose_operating_point(y_true: np.ndarray, proba: np.ndarray) -> dict:
    """Pick the MCC-maximising cut on out-of-fold predictions.

    Returns both the probability cut and the predicted positive rate it implies.
    ``THRESHOLD_TRANSFER`` decides which of the two is carried to the test set; the
    rate is the safer one because the final ensemble is trained on more data than any
    CV model and its probabilities are correspondingly sharper.
    """
    sweep = threshold_sweep(y_true, proba)
    best = sweep.loc[sweep["mcc"].idxmax()]
    default = classification_scores(y_true, (proba >= 0.5).astype(int))
    return dict(
        threshold=float(best["threshold"]),
        pos_rate=float(best["pos_rate"]),
        mcc=float(best["mcc"]),
        mcc_at_half=float(default["mcc"]),
        auc=float(roc_auc_score(y_true, proba)) if len(np.unique(y_true)) > 1 else float("nan"),
        sweep=sweep,
    )


def apply_operating_point(proba: np.ndarray, op: dict) -> np.ndarray:
    """Turn test probabilities into booleans at the chosen operating point."""
    if THRESHOLD_TRANSFER == "rate":
        rate = max(op["pos_rate"], MIN_POSITIVE_RATE)
        cut = float(np.quantile(proba, 1.0 - rate))
        pred = proba >= cut
    else:
        pred = proba >= op["threshold"]
    if pred.mean() < MIN_POSITIVE_RATE:
        cut = float(np.quantile(proba, 1.0 - MIN_POSITIVE_RATE))
        pred = proba >= cut
        print(f"    positive rate floored to {MIN_POSITIVE_RATE:.1%}")
    return pred


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--folds", type=int, default=N_FOLDS)
    p.add_argument("--seeds", type=int, nargs="+", default=ENSEMBLE_SEEDS)
    p.add_argument("--no-aux", action="store_true", help="disable the auxiliary heads")
    p.add_argument("--no-descriptors", action="store_true", help="graph only, no RDKit block")
    p.add_argument("--transfer", choices=["rate", "prob"], default=THRESHOLD_TRANSFER)
    p.add_argument("--progress", action="store_true")
    return p.parse_args()


def main() -> None:
    global USE_AUX_TASKS, USE_DESCRIPTORS, THRESHOLD_TRANSFER
    args = parse_args()
    USE_AUX_TASKS = USE_AUX_TASKS and not args.no_aux
    USE_DESCRIPTORS = USE_DESCRIPTORS and not args.no_descriptors
    THRESHOLD_TRANSFER = args.transfer
    n_folds = args.folds

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1 — data ─────────────────────────────────────────────────────────
    print("Step 1 — data")
    train_df = read_table(FILE_TRAIN_TDI)
    test_df = read_table(FILE_TEST)
    print(f"  train {len(train_df)} compounds | test {len(test_df)} compounds")

    verify_label_rule(train_df)

    # Keep only compounds carrying at least one target we train on.
    Y_all, task_names, task_weights = build_targets(train_df)
    keep = np.isfinite(Y_all).any(axis=1)
    train_df = train_df.loc[keep].reset_index(drop=True)
    Y_all = Y_all[keep]
    print(f"\n  {len(task_names)} heads: {task_names}")
    print(f"  task weights: {task_weights.tolist()}")
    print(f"  training on {len(train_df)} compounds")
    for j, name in enumerate(task_names):
        col = Y_all[:, j]
        n = int(np.isfinite(col).sum())
        pos = float(np.nanmean(col)) if n else float("nan")
        print(f"    {name:28s} n={n:5d}  positive={100 * pos:5.1f}%")

    # ── Step 2 — molecules ────────────────────────────────────────────────────
    print("\nStep 2 — molecules")
    train_mols, train_ok = build_mols(train_df["SMILES"].tolist())
    test_mols, test_ok = build_mols(test_df["SMILES"].tolist())
    train_df = train_df.loc[train_ok].reset_index(drop=True)
    Y_all = Y_all[train_ok]
    test_df = test_df.loc[test_ok].reset_index(drop=True)

    # ── Step 3 — descriptors ──────────────────────────────────────────────────
    if USE_DESCRIPTORS:
        print("\nStep 3 — RDKit 2D descriptors")
        Xtr_raw = compute_rdkit_descriptors(train_mols)
        Xte_raw = compute_rdkit_descriptors(test_mols)
        cols = select_valid_columns(Xtr_raw)
        Xtr_raw, Xte_raw = Xtr_raw[:, cols], Xte_raw[:, cols]
        scaler = StandardScaler().fit(Xtr_raw)
        x_d_train = scale_descriptors(scaler, Xtr_raw)
        x_d_test = scale_descriptors(scaler, Xte_raw)
        n_desc = x_d_train.shape[1]
        print(f"  kept {n_desc}/{len(_DESC_LIST)} descriptors")
        (OUTPUT_DIR / "kept_descriptors.txt").write_text(
            "\n".join([n for n, k in zip(_DESC_NAMES, cols) if k])
        )
    else:
        print("\nStep 3 — descriptors disabled (graph only)")
        x_d_train = x_d_test = None
        n_desc = 0

    # ── Step 4 — scaffold folds ───────────────────────────────────────────────
    print("\nStep 4 — scaffold folds")
    folds = murcko_scaffold_folds(train_mols, n_folds, SCAFFOLD_SEED)

    # ── Step 5 — cross-validation for out-of-fold probabilities ───────────────
    print(f"\nStep 5 — {n_folds}-fold scaffold CV")
    oof = np.full((len(train_mols), len(task_names)), np.nan)
    for f in range(n_folds):
        tr, va = folds != f, folds == f
        print(f"  fold {f}: {int(tr.sum())} train / {int(va.sum())} val")
        train_dset = make_dataset(
            [m for m, k in zip(train_mols, tr) if k], Y_all[tr],
            None if x_d_train is None else x_d_train[tr],
        )
        val_dset = make_dataset(
            [m for m, k in zip(train_mols, va) if k], Y_all[va],
            None if x_d_train is None else x_d_train[va],
        )
        mpnn, trainer = fit_mpnn(
            train_dset, val_dset, n_desc, task_weights, PARAMS,
            seed=SCAFFOLD_SEED, progress=args.progress,
        )
        oof[va] = predict_proba(mpnn, trainer, val_dset)

    # ── Step 6 — operating point per isoform ──────────────────────────────────
    print("\nStep 6 — operating point (out of fold)")
    ops, score_rows, sweep_frames = {}, [], []
    for j, cyp in enumerate(TDI_ISOFORMS):
        mask = np.isfinite(Y_all[:, j])
        y_true = Y_all[mask, j].astype(int)
        proba = oof[mask, j]
        op = choose_operating_point(y_true, proba)
        ops[cyp] = op
        sweep = op.pop("sweep")
        sweep.insert(0, "isoform", cyp)
        sweep_frames.append(sweep)

        base_rate = float(y_true.mean())
        if op["pos_rate"] > POS_RATE_WARN_MULTIPLE * base_rate:
            print(
                f"  WARNING {cyp}: the MCC optimum calls {100 * op['pos_rate']:.1f}% positive "
                f"against a {100 * base_rate:.1f}% base rate. Weak ranking (AUC {op['auc']:.3f}) "
                f"lets MCC peak at an implausible cut — check threshold_sweep.csv and consider "
                f"a cut nearer the base rate."
            )
        pred = (proba >= op["threshold"]).astype(int)
        sc = classification_scores(y_true, pred)
        score_rows.append(dict(isoform=cyp, n=int(mask.sum()), base_rate=base_rate,
                               auc=op["auc"], threshold=op["threshold"],
                               pos_rate=op["pos_rate"], **sc))
        print(
            f"  {cyp}: n={int(mask.sum()):5d}  AUC={op['auc']:.3f}  "
            f"MCC@0.50={op['mcc_at_half']:.4f}  ->  MCC@{op['threshold']:.2f}={op['mcc']:.4f}  "
            f"(calls {100 * op['pos_rate']:.1f}% positive)"
        )

    scores = pd.DataFrame(score_rows)
    ma_mcc = float(scores["mcc"].mean())
    print(f"\n  MA-MCC (out of fold, tuned) = {ma_mcc:.4f}")
    print(f"  MA-MCC at the notebook's 0.5 = {np.mean([ops[c]['mcc_at_half'] for c in TDI_ISOFORMS]):.4f}")

    pd.concat(sweep_frames).to_csv(OUTPUT_DIR / "threshold_sweep.csv", index=False)
    scores.to_csv(OUTPUT_DIR / "oof_scores.csv", index=False)

    oof_frame = pd.DataFrame(oof, columns=[f"{n}_proba" for n in task_names])
    oof_frame.insert(0, "fold", folds)
    oof_frame.insert(0, "Molecule_Name", train_df["Molecule_Name"].to_numpy())
    for j, cyp in enumerate(TDI_ISOFORMS):
        oof_frame[f"{cyp}_true"] = Y_all[:, j]
    oof_frame.to_csv(OUTPUT_DIR / "oof_predictions.csv", index=False)

    # ── Step 7 — ensemble on all data ─────────────────────────────────────────
    print(f"\nStep 7 — final ensemble, {len(args.seeds)} seeds on all data")
    full_dset = make_dataset(train_mols, Y_all, x_d_train)
    test_dset = make_dataset(test_mols, np.full((len(test_mols), len(task_names)), np.nan), x_d_test)

    per_seed = []
    for seed in args.seeds:
        print(f"  seed {seed}")
        mpnn, trainer = fit_mpnn(
            full_dset, None, n_desc, task_weights, PARAMS, seed=seed, progress=args.progress
        )
        per_seed.append(predict_proba(mpnn, trainer, test_dset))
    test_proba = np.mean(per_seed, axis=0)
    seed_sd = np.std(per_seed, axis=0)

    # ── Step 8 — threshold and write the submission ───────────────────────────
    print("\nStep 8 — submission")
    submission = test_df[["SMILES", "Molecule_Name"]].copy()
    detail = submission.copy()
    for j, cyp in enumerate(TDI_ISOFORMS):
        proba = test_proba[:, j]
        pred = apply_operating_point(proba, ops[cyp])
        submission[f"{cyp}_is_TDI"] = pred.astype(bool)
        detail[f"{cyp}_proba"] = proba
        detail[f"{cyp}_seed_sd"] = seed_sd[:, j]
        detail[f"{cyp}_is_TDI"] = pred.astype(bool)
        print(
            f"  {cyp}: calls {100 * pred.mean():.1f}% positive on test "
            f"(out of fold optimum was {100 * ops[cyp]['pos_rate']:.1f}%), "
            f"mean proba {proba.mean():.3f}, mean seed sd {seed_sd[:, j].mean():.3f}"
        )

    sub_path = OUTPUT_DIR / "my_chemprop_tdi_submission.csv"
    submission.to_csv(sub_path, index=False)
    detail.to_csv(OUTPUT_DIR / "test_predictions_detail.csv", index=False)
    print(f"\n  wrote {sub_path}  ({len(submission)} rows)")

    # ── Step 9 — validate ─────────────────────────────────────────────────────
    try:
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        from validation.tdi_validation import validate_tdi_submission

        ok, errors = validate_tdi_submission(sub_path, expected_ids=set(test_df["Molecule_Name"]))
        print("  validation:", "PASS" if ok else "FAIL")
        for msg in errors or []:
            print(f"    - {msg}")
    except Exception as exc:                                    # noqa: BLE001
        print(f"  validation skipped ({exc})")

    print(
        "\nDone. Inspect threshold_sweep.csv before submitting — if the MCC curve is flat "
        "near the optimum the cut is safe, and if it is a narrow spike it is fitted to the "
        "CV folds and you should prefer a cut nearer the label base rate (~21%)."
    )


if __name__ == "__main__":
    main()
