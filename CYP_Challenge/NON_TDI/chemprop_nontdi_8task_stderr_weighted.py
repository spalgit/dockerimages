"""8-task CYP non-TDI ChemProp ensemble with PER-ISOFORM 1/std sample weighting.

Identical to chemprop_nontdi_8task_ensemble.py in every respect -- 8 heads, censored
loss, splits, trunks, descriptors, ensembling -- except that each molecule carries a
loss weight derived from its measurement precision.  The weighting is the only variable.

Why per-isoform, and not everywhere
-----------------------------------
Label imprecision is strongly ANTI-correlated with potency in this dataset
(Spearman(pIC50, std) = -0.89 / -0.90 / -0.56 / -0.93 for 1A2 / 2C9 / 2D6 / 3A4), so
1/std weighting shifts the effective training distribution toward potent compounds.
Whether that helps depends on which way each isoform's train/blind mismatch runs:

    isoform   train mu   blind mu   shift    weighting moves training...
    CYP1A2      4.96       4.32     -0.63    away from the blind set   -> OFF
    CYP2C9      4.58       4.89     +0.31    toward the blind set      -> ON
    CYP2D6      4.78       3.11     -1.67    far away                  -> OFF
    CYP3A4      4.10       4.74     +0.64    toward the blind set      -> ON

CYP3A4 is also the most heteroscedastic isoform (p90/p10 std ratio 21x, 29% of credible
bands wider than 1 log unit), so it has the most to gain.

This is NOT the multi-task variant.  Predicting std as an auxiliary head was tried on
PXR and hurt rank (see chemprop_pxr_pec50_rdkit2d_stderr_weight.py); the redundancy is
the reason -- only 10.7% of CYP3A4 std variance is not already a monotone function of
the pIC50 the model is predicting anyway.  std is metadata about measurement quality,
not a second objective.

Scalar-weight caveat
--------------------
chemprop's MoleculeDatapoint.weight is one float per MOLECULE, not per task -- the loss
forms a rank-1 product of per-sample and per-task weights -- so a per-isoform weight
cannot be expressed exactly.  It is nearly exact here because 73% of molecules carry a
single isoform label and only 720 of 4,905 (15%) mix a weighted with an unweighted
isoform.  Those 720 default to weight 1.0 (--mixed-policy neutral) so that no 1A2 or
2D6 label is ever scaled by a 2C9/3A4-derived weight.

    python chemprop_nontdi_8task_stderr_weighted.py --trunk scratch \
        --weighted-isoforms CYP2C9,CYP3A4 --weight-cap 3.0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

DATA_DIR = Path(os.environ.get(
    "CYP_DATA_DIR", Path.home() / "dockerimages" / "CYP_Challenge" / "NON_TDI"))
OUTPUT_DIR = Path(os.environ.get(
    "CYP_OUTPUT_DIR", Path.home() / "dockerimages" / "CYP_Challenge" / "NON_TDI"
    / "outputs_8task_ensemble"))

FILE_TRAIN = "cyp-challenge-TRAIN_direct_inhibition_with_single_shot_LABELLED.csv"
FILE_TEST = "cyp-challenge-TEST-BLINDED_ligprepped.csv"

SMILES_COL = "SMILES"          # the ligprepped structure, not SMILES_raw
ID_COL = "Molecule_Name"

ISOFORMS = ("CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4")
N_ISO = len(ISOFORMS)
TARGET_COLS = [f"{i}_pIC50_direct_inhibition" for i in ISOFORMS]

AUX_MODE = "log2fc"            # "log2fc" | "is_hit"
AUX_TASK_WEIGHT = 0.2          # weight of each auxiliary head vs a pIC50 head

# ── Ensemble ──────────────────────────────────────────────────────────────────
SPLIT_METHODS = ("random", "stratified", "scaffold")
SEEDS = (42, 123, 456, 789, 1337)
VAL_FRAC = 0.12                # held out per member, used for early stopping

# ── Architecture ──────────────────────────────────────────────────────────────
# FFN_HIDDEN_DIM cycles across ensemble members so the ensemble spans capacities
# rather than betting on one. 800 is included as requested.
FFN_HIDDEN_DIMS = (800, 600, 800, 400, 800)
MP_HIDDEN_DIM = 400            # scratch trunk only; CheMeleon's width is fixed at 2048
MP_DEPTH = 4                   # scratch trunk only; CheMeleon's depth is fixed at 6
FFN_N_LAYERS = 3
DROPOUT = 0.15
BATCH_SIZE = 128            # 4,905 compounds; 128 halves epoch time vs 64

MAX_EPOCHS = 400
PATIENCE = 50                  # early-stopping patience on val loss
INIT_LR, MAX_LR, FINAL_LR = 1e-4, 1e-3, 1e-4
WARMUP_EPOCHS = 5

# ── CheMeleon foundation trunk (--trunk chemeleon) ────────────────────────────
# A pretrained D-MPNN (d_h=2048, depth=6, 8.7M parameters) from
# https://doi.org/10.5281/zenodo.15426600. Replaces the randomly-initialised
# message-passing block; everything downstream (8 heads, censored loss, splits,
# ensembling) is identical, so the two runs are directly comparable.
CHEMELEON_CKPT = Path(os.environ.get(
    "CHEMELEON_CKPT", Path.home() / ".chemprop" / "chemeleon_mp.pt"))
CHEMELEON_URL = "https://zenodo.org/records/15460715/files/chemeleon_mp.pt"
# Finetuning a pretrained trunk wants a gentler schedule than training one from
# scratch -- at MAX_LR=1e-3 the first epochs overwrite the pretrained weights.
CHEMELEON_LRS = (5e-5, 3e-4, 5e-5)   # (init, max, final)
CHEMELEON_FREEZE = False       # True = use CheMeleon purely as a fixed featuriser

# ── Label handling ────────────────────────────────────────────────────────────
USE_RDKIT2D = True             # concatenate an RDKit 2-D descriptor block to the FFN
MAX_DESCRIPTORS = 200
TOP_CONC_PIC50 = -float(np.log10(49.5e-6))  # 4.305, the top screening concentration
USE_CENSORED_LOSS = True       # BoundedMSE: never penalise predicting *below* a
                               # left-censored pIC50 label
PRED_CLIP = (2.5, 9.0)

NUM_WORKERS = 0                # >0 can deadlock in containers; leave at 0

# ── 1/std sample weighting ────────────────────────────────────────────────────
WEIGHTED_ISOFORMS = ("CYP2C9", "CYP3A4")   # see the module docstring for why these two
STD_COL = "{iso}_pIC50_direct_inhibition_std"
WEIGHT_CAP = 3.0               # winsorise w to [1/cap, cap] AFTER normalising to mean 1.
                               # Uncapped 1/std spans ~21x on CYP3A4, which lets a handful
                               # of compounds dominate the gradient.
STD_FLOOR = 0.02               # guards 1/std against a near-zero std only. Deliberately
                               # below the 1st percentile of both weighted isoforms
                               # (2C9 0.039, 3A4 0.023) so that WEIGHT_CAP is the single
                               # knob controlling the weight range -- a floor of 0.05 put
                               # 25% of CYP3A4 on one identical maximum weight.
MIXED_POLICY = "neutral"       # molecules labelled on both a weighted and an unweighted
                               # isoform: "neutral" = 1.0, "weighted" = use the weight


# ══════════════════════════════════════════════════════════════════════════════
# Data
# ══════════════════════════════════════════════════════════════════════════════

def aux_columns(mode: str) -> list[str]:
    if mode == "log2fc":
        return [f"{i}_log2fc_estimate" for i in ISOFORMS]
    if mode == "is_hit":
        return [f"{i}_is_hit" for i in ISOFORMS]
    raise ValueError(f"unknown AUX_MODE {mode!r}")


def load_frames(data_dir: Path, aux_mode: str):
    train = pd.read_csv(data_dir / FILE_TRAIN)
    test = pd.read_csv(data_dir / FILE_TEST)

    missing = [c for c in [SMILES_COL, ID_COL] + TARGET_COLS if c not in train.columns]
    if missing:
        raise SystemExit(f"training file is missing columns: {missing}")
    aux = aux_columns(aux_mode)
    for c in aux:
        if c not in train.columns:
            raise SystemExit(f"training file is missing auxiliary column {c!r}")
    if SMILES_COL not in test.columns:
        raise SystemExit(f"test file has no {SMILES_COL!r} column")

    train = train[train[SMILES_COL].notna()].reset_index(drop=True)
    test = test[test[SMILES_COL].notna()].reset_index(drop=True)
    return train, test, aux


def build_targets(df: pd.DataFrame, aux_cols: list[str], aux_mode: str):
    """-> (y [n, 8], lt_mask [n, 8]).  NaN in y means 'no label, do not train on it'."""
    y_primary = df[TARGET_COLS].to_numpy(dtype=float)
    y_aux = df[aux_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    y = np.hstack([y_primary, y_aux])

    lt = np.zeros_like(y, dtype=bool)
    if USE_CENSORED_LOSS:
        # A pIC50 at or below the top screening concentration is a "<=" statement,
        # not a point estimate. lt_mask + BoundedMSE charges nothing for predicting
        # under it and only penalises calling an inactive compound potent.
        with np.errstate(invalid="ignore"):
            lt[:, :N_ISO] = np.nan_to_num(y_primary, nan=9.9) <= TOP_CONC_PIC50
    if aux_mode == "is_hit":
        # a 0/1 target regressed under MSE == Brier score; nothing is censored
        pass
    return y, lt


def build_sample_weights(df: pd.DataFrame, weighted_isoforms, cap, mixed_policy):
    """-> (w [n], report) — one loss weight per molecule, mean 1.0 over labelled rows.

    w = 1/std of the weighted isoforms the molecule is labelled on (geometric mean when
    it has more than one), normalised to mean 1.0 and winsorised to [1/cap, cap].
    Molecules with no weighted-isoform label, and — under "neutral" — molecules that also
    carry an unweighted-isoform label, get exactly 1.0.
    """
    n = len(df)
    w = np.ones(n, dtype=float)
    if not weighted_isoforms:
        return w, {"weighted_isoforms": [], "n_weighted": 0}

    unweighted = [i for i in ISOFORMS if i not in weighted_isoforms]
    lab_w = np.zeros(n, dtype=bool)
    lab_u = np.zeros(n, dtype=bool)
    for i in weighted_isoforms:
        lab_w |= df[f"{i}_pIC50_direct_inhibition"].notna().to_numpy()
    for i in unweighted:
        lab_u |= df[f"{i}_pIC50_direct_inhibition"].notna().to_numpy()

    # geometric mean of 1/std across the weighted isoforms this molecule is labelled on
    log_acc = np.zeros(n); cnt = np.zeros(n)
    for i in weighted_isoforms:
        std = pd.to_numeric(df.get(STD_COL.format(iso=i)), errors="coerce").to_numpy(float)
        lab = df[f"{i}_pIC50_direct_inhibition"].notna().to_numpy()
        ok = lab & np.isfinite(std)
        log_acc[ok] += np.log(1.0 / np.maximum(std[ok], STD_FLOOR))
        cnt[ok] += 1
    raw = np.where(cnt > 0, np.exp(log_acc / np.maximum(cnt, 1)), np.nan)

    apply = cnt > 0
    if mixed_policy == "neutral":
        apply &= ~lab_u                      # never scale an unweighted isoform's label
    if apply.sum() == 0:
        return w, {"weighted_isoforms": list(weighted_isoforms), "n_weighted": 0}

    v = raw[apply]
    v = v / v.mean()                          # preserve the effective learning rate
    v = np.clip(v, 1.0 / cap, cap)
    v = v / v.mean()                          # re-normalise after winsorising
    w[apply] = v
    return w, {"weighted_isoforms": list(weighted_isoforms),
               "n_weighted": int(apply.sum()), "n_neutral": int((~apply).sum()),
               "n_mixed_skipped": int((cnt > 0).sum() - apply.sum()),
               "w_min": float(v.min()), "w_max": float(v.max()),
               "w_p10": float(np.quantile(v, .1)), "w_p90": float(np.quantile(v, .9)),
               "cap": cap, "mixed_policy": mixed_policy}


def rdkit2d_block(mols, kept: list[str] | None):
    """RDKit 2-D descriptors as a dense float block. `kept` fixes the column set so
    train and test always produce identically-shaped, identically-ordered blocks."""
    from rdkit.Chem import Descriptors
    names = [n for n, _ in Descriptors.descList]
    rows = []
    for m in mols:
        try:
            vals = dict(zip(names, Descriptors.CalcMolDescriptors(m).values()))
        except Exception:
            vals = {}
        rows.append(vals)
    X = pd.DataFrame(rows, columns=names).apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    if kept is None:
        ok = X.notna().all(axis=0) & (X.std(axis=0) > 0) & (X.abs().max(axis=0) < 1e10)
        kept = list(X.columns[ok][:MAX_DESCRIPTORS])
    return X[kept].fillna(0.0).to_numpy(dtype=float), kept


# ══════════════════════════════════════════════════════════════════════════════
# Splitters
# ══════════════════════════════════════════════════════════════════════════════

def split_random(df, seed, val_frac):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n_val = max(1, int(round(val_frac * len(df))))
    return np.sort(idx[n_val:]), np.sort(idx[:n_val])


def _strata_key(df):
    """(label pattern) x (activity quartile) -- keeps label sparsity and potency
    mix identical between train and val."""
    labelled = df[TARGET_COLS].notna()
    pattern = labelled.astype(int).astype(str).agg("".join, axis=1)
    mean_pic50 = df[TARGET_COLS].mean(axis=1, skipna=True)
    try:
        band = pd.qcut(mean_pic50, 4, labels=False, duplicates="drop")
    except ValueError:
        band = pd.Series(np.zeros(len(df)), index=df.index)
    band = band.fillna(-1).astype(int).astype(str)
    return (pattern + "_" + band).to_numpy()


def split_stratified(df, seed, val_frac):
    from sklearn.model_selection import StratifiedShuffleSplit
    key = _strata_key(df)
    # StratifiedShuffleSplit needs >= 2 members per class; pool the rare ones
    vc = pd.Series(key).value_counts()
    rare = set(vc[vc < 2].index)
    key = np.array(["__rare__" if k in rare else k for k in key])
    if pd.Series(key).nunique() < 2:
        return split_random(df, seed, val_frac)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    tr, va = next(sss.split(np.zeros(len(df)), key))
    return np.sort(tr), np.sort(va)


def split_scaffold(df, seed, val_frac):
    """Bemis-Murcko. Whole scaffolds move together; scaffold groups are shuffled by
    `seed` and filled largest-first into train until the val quota is reached."""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    scaffs = {}
    for i, smi in enumerate(df[SMILES_COL]):
        m = Chem.MolFromSmiles(smi)
        s = MurckoScaffold.MurckoScaffoldSmiles(mol=m) if m else f"__bad__{i}"
        scaffs.setdefault(s, []).append(i)

    groups = list(scaffs.values())
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)      # big scaffolds into train first

    n_val_target = max(1, int(round(val_frac * len(df))))
    val: list[int] = []
    train: list[int] = []
    for g in groups:
        if len(val) < n_val_target and len(g) <= max(1, n_val_target - len(val)):
            val.extend(g)
        else:
            train.extend(g)
    if not val:                              # degenerate: fall back
        return split_random(df, seed, val_frac)
    return np.sort(np.array(train)), np.sort(np.array(val))


SPLITTERS = {"random": split_random,
             "stratified": split_stratified,
             "scaffold": split_scaffold}


# ══════════════════════════════════════════════════════════════════════════════
# Model
# ══════════════════════════════════════════════════════════════════════════════

def make_datapoints(mols, y, lt, x_d, w=None):
    from chemprop import data
    n = y.shape[1]
    return [data.MoleculeDatapoint(
        mol=m,
        y=y[i].astype(float),
        weight=1.0 if w is None else float(w[i]),
        x_d=None if x_d is None else x_d[i].astype(float),
        lt_mask=lt[i].astype(bool),
        gt_mask=np.zeros(n, dtype=bool),
    ) for i, m in enumerate(mols)]


def ensure_chemeleon() -> Path:
    """Return the CheMeleon checkpoint path, downloading it once if absent."""
    if CHEMELEON_CKPT.exists():
        return CHEMELEON_CKPT
    from urllib.request import urlretrieve
    CHEMELEON_CKPT.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading CheMeleon -> {CHEMELEON_CKPT}")
    urlretrieve(CHEMELEON_URL, CHEMELEON_CKPT)
    return CHEMELEON_CKPT


def build_message_passing(trunk: str):
    """The only difference between the two runs. CheMeleon carries its own
    hyper-parameters (d_v=72, d_e=14, d_h=2048, depth=6) which must match the
    featuriser exactly, so they are read from the checkpoint, never hard-coded."""
    import torch
    from chemprop import featurizers, nn

    feat = featurizers.SimpleMoleculeMolGraphFeaturizer()
    if trunk == "scratch":
        return nn.BondMessagePassing(d_v=feat.atom_fdim, d_e=feat.bond_fdim,
                                     depth=MP_DEPTH, d_h=MP_HIDDEN_DIM)

    ck = torch.load(ensure_chemeleon(), weights_only=True)
    hp = ck["hyper_parameters"]
    if (hp["d_v"], hp["d_e"]) != (feat.atom_fdim, feat.bond_fdim):
        raise SystemExit(
            f"CheMeleon expects d_v={hp['d_v']}, d_e={hp['d_e']} but this chemprop's "
            f"featuriser gives {feat.atom_fdim}, {feat.bond_fdim} — version mismatch")
    mp = nn.BondMessagePassing(**hp)
    mp.load_state_dict(ck["state_dict"])
    if CHEMELEON_FREEZE:
        mp.eval()
        mp.apply(lambda m: m.requires_grad_(False))
    return mp


def build_model(n_desc, target_scaler, ffn_hidden_dim, seed, trunk):
    import torch
    from chemprop import models, nn

    torch.manual_seed(seed)
    mp = build_message_passing(trunk)
    agg = nn.MeanAggregation()

    weights = torch.tensor([[1.0] * N_ISO + [AUX_TASK_WEIGHT] * N_ISO])
    criterion = (nn.metrics.BoundedMSE(task_weights=weights)
                 if USE_CENSORED_LOSS else nn.metrics.MSE(task_weights=weights))

    ffn = nn.RegressionFFN(
        n_tasks=2 * N_ISO,
        input_dim=mp.output_dim + n_desc,   # 2048 + n_desc under CheMeleon
        hidden_dim=ffn_hidden_dim,
        n_layers=FFN_N_LAYERS,
        dropout=DROPOUT,
        criterion=criterion,
        output_transform=nn.UnscaleTransform.from_standard_scaler(target_scaler),
    )
    init_lr, max_lr, final_lr = (
        CHEMELEON_LRS if trunk == "chemeleon" else (INIT_LR, MAX_LR, FINAL_LR))
    return models.MPNN(mp, agg, ffn, batch_norm=True,
                       metrics=[nn.metrics.RMSE(), nn.metrics.MAE()],
                       warmup_epochs=WARMUP_EPOCHS,
                       init_lr=init_lr, max_lr=max_lr, final_lr=final_lr)


def train_member(mols_tr, y_tr, lt_tr, xd_tr,
                 mols_va, y_va, lt_va, xd_va,
                 mols_te, xd_te, ffn_hidden_dim, seed, max_epochs, log_dir, trunk,
                 batch_size, w_tr=None, w_va=None):
    import lightning.pytorch as pl
    import torch
    from lightning.pytorch.callbacks import EarlyStopping
    from chemprop import data, featurizers

    # use Tensor Cores where the GPU has them; no measurable accuracy cost here
    torch.set_float32_matmul_precision("high")

    feat = featurizers.SimpleMoleculeMolGraphFeaturizer()
    # Weights are applied to TRAIN only. The val loss drives early stopping, and a
    # reweighted val loss would select checkpoints against a different objective than
    # the leaderboard's -- so validation stays unweighted on purpose.
    dset_tr = data.MoleculeDataset(
        make_datapoints(mols_tr, y_tr, lt_tr, xd_tr, w_tr), feat)
    dset_va = data.MoleculeDataset(make_datapoints(mols_va, y_va, lt_va, xd_va), feat)
    n_te = len(mols_te)
    dset_te = data.MoleculeDataset(
        make_datapoints(mols_te, np.full((n_te, 2 * N_ISO), np.nan),
                        np.zeros((n_te, 2 * N_ISO), bool), xd_te), feat)

    target_scaler = dset_tr.normalize_targets()
    dset_va.normalize_targets(target_scaler)

    n_desc = 0
    if xd_tr is not None:
        xd_scaler = dset_tr.normalize_inputs("X_d")
        dset_va.normalize_inputs("X_d", xd_scaler)
        dset_te.normalize_inputs("X_d", xd_scaler)
        n_desc = xd_tr.shape[1]

    # a batch of size 1 breaks batch-norm; drop_last only when it would occur
    drop_last = (len(dset_tr) % batch_size) == 1
    load_tr = data.build_dataloader(dset_tr, batch_size=batch_size, shuffle=True,
                                    num_workers=NUM_WORKERS, drop_last=drop_last)
    load_va = data.build_dataloader(dset_va, batch_size=batch_size, shuffle=False,
                                    num_workers=NUM_WORKERS)
    load_te = data.build_dataloader(dset_te, batch_size=batch_size, shuffle=False,
                                    num_workers=NUM_WORKERS)

    model = build_model(n_desc, target_scaler, ffn_hidden_dim, seed, trunk)
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        enable_checkpointing=False,
        default_root_dir=str(log_dir),
        callbacks=[EarlyStopping(monitor="val_loss", patience=PATIENCE, mode="min")],
        gradient_clip_val=5.0,
    )
    trainer.fit(model, load_tr, load_va)

    model.eval()
    with torch.no_grad():
        pred_va = np.vstack([p.numpy() for p in trainer.predict(model, load_va)])
        pred_te = np.vstack([p.numpy() for p in trainer.predict(model, load_te)])
    return pred_va, pred_te, int(trainer.current_epoch)


# ══════════════════════════════════════════════════════════════════════════════
# Scoring
# ══════════════════════════════════════════════════════════════════════════════

def score(y_true, y_pred):
    from scipy.stats import spearmanr
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() < 5:
        return dict(n=int(m.sum()), mae=np.nan, rmse=np.nan, r2=np.nan, spearman=np.nan)
    t, p = y_true[m], y_pred[m]
    ss_res = float(((t - p) ** 2).sum())
    ss_tot = float(((t - t.mean()) ** 2).sum())
    return dict(n=int(m.sum()),
                mae=float(np.abs(t - p).mean()),
                rmse=float(np.sqrt(((t - p) ** 2).mean())),
                r2=1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan,
                spearman=float(spearmanr(t, p).statistic))


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    ap.add_argument("--aux-mode", choices=("log2fc", "is_hit"), default=AUX_MODE)
    ap.add_argument("--trunk", choices=("scratch", "chemeleon"), default="scratch",
                    help="message-passing trunk: randomly initialised, or the "
                         "pretrained CheMeleon foundation model")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--splits", nargs="+", default=list(SPLIT_METHODS),
                    choices=list(SPLITTERS))
    ap.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--no-rdkit2d", action="store_true")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--max-hours", type=float, default=None,
                    help="wall-clock budget. No NEW member starts once this is "
                         "passed; the ensemble is built from whatever finished. "
                         "Re-run later to pick up the rest.")
    ap.add_argument("--weighted-isoforms", default=",".join(WEIGHTED_ISOFORMS),
                    help="comma-separated isoforms to apply 1/std weighting to, or "
                         "'none' to disable (gives the unweighted baseline)")
    ap.add_argument("--weight-cap", type=float, default=WEIGHT_CAP,
                    help="winsorise weights to [1/cap, cap] after normalising to mean 1")
    ap.add_argument("--mixed-policy", choices=("neutral", "weighted"),
                    default=MIXED_POLICY,
                    help="molecules labelled on both a weighted and an unweighted "
                         "isoform: neutral = weight 1.0 (default), weighted = apply it")
    ap.add_argument("--smoke", action="store_true",
                    help="3 epochs, 2 members, for a fast end-to-end check")
    args = ap.parse_args()

    if args.smoke:
        args.epochs = 3
        args.seeds = args.seeds[:1]
        args.splits = args.splits[:2]

    out = args.output_dir
    (out / "members").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"data   : {args.data_dir}")
    print(f"output : {out}")
    print(f"aux    : {args.aux_mode}   splits: {args.splits}   seeds: {args.seeds}")
    print(f"trunk  : {args.trunk}")
    if args.trunk == "chemeleon":
        ensure_chemeleon()

    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    train, test, aux_cols = load_frames(args.data_dir, args.aux_mode)
    mols_tr_all = [Chem.MolFromSmiles(s) for s in train[SMILES_COL]]
    mols_te = [Chem.MolFromSmiles(s) for s in test[SMILES_COL]]
    bad = [i for i, m in enumerate(mols_tr_all) if m is None]
    if bad:
        print(f"  dropping {len(bad)} unparseable training SMILES")
        keep = [i for i in range(len(train)) if i not in set(bad)]
        train = train.iloc[keep].reset_index(drop=True)
        mols_tr_all = [mols_tr_all[i] for i in keep]
    if any(m is None for m in mols_te):
        raise SystemExit("unparseable SMILES in the test file")

    y_all, lt_all = build_targets(train, aux_cols, args.aux_mode)

    wiso = () if args.weighted_isoforms.strip().lower() == "none" else tuple(
        x.strip() for x in args.weighted_isoforms.split(",") if x.strip())
    unknown = [i for i in wiso if i not in ISOFORMS]
    if unknown:
        raise SystemExit(f"--weighted-isoforms: unknown isoform(s) {unknown}")
    w_all, w_report = build_sample_weights(train, wiso, args.weight_cap,
                                           args.mixed_policy)
    if wiso:
        print(f"  1/std weighting on {list(wiso)}: {w_report['n_weighted']} molecules "
              f"weighted, {w_report['n_neutral']} at 1.0 "
              f"({w_report['n_mixed_skipped']} mixed-label skipped)")
        print(f"    w range [{w_report['w_min']:.3f}, {w_report['w_max']:.3f}]  "
              f"p10-p90 [{w_report['w_p10']:.3f}, {w_report['w_p90']:.3f}]  "
              f"cap {args.weight_cap}")
    else:
        w_all = None
        print("  1/std weighting: DISABLED (unweighted baseline)")

    print(f"  train {len(train)} compounds, test {len(test)}")
    for j, c in enumerate(TARGET_COLS + aux_cols):
        print(f"    task {j}: {c:38s} {int(np.isfinite(y_all[:, j]).sum()):5d} labels")

    xd_all = xd_te = None
    if USE_RDKIT2D and not args.no_rdkit2d:
        xd_all, kept = rdkit2d_block(mols_tr_all, None)
        xd_te, _ = rdkit2d_block(mols_te, kept)
        (out / "kept_descriptors.txt").write_text("\n".join(kept))
        print(f"  rdkit2d block: {xd_all.shape[1]} descriptors")

    members = [(sp, sd) for sp in args.splits for sd in args.seeds]
    print(f"\n{len(members)} ensemble members "
          f"({len(args.splits)} splits x {len(args.seeds)} seeds), "
          f"up to {args.epochs} epochs each\n")

    test_preds, oof_rows = [], []
    for k, (sp, sd) in enumerate(members, 1):
        tag = f"{sp}_seed{sd}"
        f_te = out / "members" / f"test_{tag}.npy"
        f_va = out / "members" / f"val_{tag}.csv"
        if f_te.exists() and f_va.exists():
            print(f"[{k}/{len(members)}] {tag}: cached, skipping")
            test_preds.append(np.load(f_te))
            oof_rows.append(pd.read_csv(f_va))
            continue

        if args.max_hours is not None and (time.time() - t0) / 3600 > args.max_hours:
            print(f"[{k}/{len(members)}] {tag}: skipped — {args.max_hours}h budget "
                  f"spent. Re-run to continue from here.")
            continue

        tr_idx, va_idx = SPLITTERS[sp](train, sd, VAL_FRAC)
        ffn_dim = FFN_HIDDEN_DIMS[k % len(FFN_HIDDEN_DIMS)]
        print(f"[{k}/{len(members)}] {tag}: train {len(tr_idx)} / val {len(va_idx)}, "
              f"ffn_hidden={ffn_dim} ... ", end="", flush=True)

        t1 = time.time()
        pred_va, pred_te, epochs_run = train_member(
            [mols_tr_all[i] for i in tr_idx], y_all[tr_idx], lt_all[tr_idx],
            None if xd_all is None else xd_all[tr_idx],
            [mols_tr_all[i] for i in va_idx], y_all[va_idx], lt_all[va_idx],
            None if xd_all is None else xd_all[va_idx],
            mols_te, xd_te, ffn_dim, sd, args.epochs, out / "logs", args.trunk,
            args.batch_size,
            w_tr=None if w_all is None else w_all[tr_idx])

        pred_te = np.clip(pred_te[:, :N_ISO], *PRED_CLIP)
        np.save(f_te, pred_te)
        va = pd.DataFrame({ID_COL: train.iloc[va_idx][ID_COL].to_numpy(),
                           "split": sp, "seed": sd})
        for j, iso in enumerate(ISOFORMS):
            va[f"{iso}_true"] = y_all[va_idx, j]
            va[f"{iso}_pred"] = np.clip(pred_va[:, j], *PRED_CLIP)
        va.to_csv(f_va, index=False)
        test_preds.append(pred_te)
        oof_rows.append(va)
        print(f"done in {time.time() - t1:.0f}s ({epochs_run} epochs)")

    if not test_preds:
        print("\nNo members completed — nothing to ensemble. "
              "Raise --max-hours or re-run.")
        return 1

    # ── out-of-fold scores, per split method ─────────────────────────────────
    oof = pd.concat(oof_rows, ignore_index=True)
    oof.to_csv(out / "oof_predictions.csv", index=False)
    rows = []
    for sp in sorted(oof["split"].unique()):
        d = oof[oof["split"] == sp]
        agg = d.groupby(ID_COL).mean(numeric_only=True)   # average repeated holdouts
        for iso in ISOFORMS:
            rows.append(dict(split=sp, isoform=iso,
                             **score(agg[f"{iso}_true"].to_numpy(),
                                     agg[f"{iso}_pred"].to_numpy())))
    scores = pd.DataFrame(rows)
    scores.to_csv(out / "oof_scores.csv", index=False)
    print("\nOut-of-fold (repeated holdout, averaged over seeds):")
    print(scores.to_string(index=False))
    print("\nMean Spearman by split method:")
    print(scores.groupby("split")["spearman"].mean().round(4).to_string())

    # ── ensemble submission ──────────────────────────────────────────────────
    P = np.mean(np.stack(test_preds), axis=0)
    sub = pd.DataFrame({ID_COL: test[ID_COL], SMILES_COL: test[SMILES_COL]})
    for j, iso in enumerate(ISOFORMS):
        sub[f"{iso}_pIC50_direct_inhibition"] = P[:, j]
    sub.to_csv(out / "submission_ensemble.csv", index=False)

    spread = np.std(np.stack(test_preds), axis=0).mean(axis=0)
    meta = dict(n_members=len(test_preds), members=[f"{s}_seed{d}" for s, d in members],
                trunk=args.trunk, aux_mode=args.aux_mode, epochs=args.epochs,
                batch_size=args.batch_size,
                ffn_hidden_dims=list(FFN_HIDDEN_DIMS), mp_hidden_dim=MP_HIDDEN_DIM,
                mp_depth=MP_DEPTH, ffn_n_layers=FFN_N_LAYERS, dropout=DROPOUT,
                smiles_column=SMILES_COL, rdkit2d=bool(xd_all is not None),
                censored_loss=USE_CENSORED_LOSS, aux_task_weight=AUX_TASK_WEIGHT,
                stderr_weighting=w_report,
                member_disagreement_sd={i: round(float(s), 4)
                                        for i, s in zip(ISOFORMS, spread)},
                runtime_s=round(time.time() - t0, 1))
    (out / "run_metadata.json").write_text(json.dumps(meta, indent=2))

    print(f"\npredicted pIC50 per isoform (mean +- sd across the {len(test)} test compounds):")
    for j, iso in enumerate(ISOFORMS):
        print(f"  {iso}: {P[:, j].mean():.3f} +- {P[:, j].std():.3f}   "
              f"member disagreement sd {spread[j]:.3f}")
    print(f"\nwrote {out / 'submission_ensemble.csv'}")
    print(f"total {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
