#!/usr/bin/env python
"""
CYP direct-inhibition (non-TDI) — 8-task ensemble on the pseudo-labelled corpus
═══════════════════════════════════════════════════════════════════════════════

`chemprop_nontdi_8task_ensemble.py` with the single-shot pseudo-labels switched on.
Architecture, splits, seeds, trunk, descriptors, ensembling and scoring are imported
unchanged from that module, so any difference in result is attributable to the data
and to the censoring rule below — nothing else.

Generate the data first (see `PSEUDOLABEL_METHOD_EXPLAINED.md` for how, and
`PSEUDOLABEL_RUN_HOWTO.md` for the run order):

    python cyp_singleshot_pseudolabels.py --tiers S

which writes `..._LABELLED_pseudo_tierS.csv` — CYP2D6 goes from 1,493 labels to 3,917
by filling 2,424 confirmed non-inhibitors at that isoform's own dose-response floor.

Why this cannot just be `--data-file` on the base script
────────────────────────────────────────────────────────
The base script marks **every** pIC50 at or below the top screening concentration as
left-censored:

    lt[:, :N_ISO] = y_primary <= TOP_CONC_PIC50          # 4.305

and trains those cells under `BoundedMSE`, which charges nothing for predicting below
the label. That is correct for a real dose-response reading that bottomed out. It is
catastrophic for the pseudo-labels: all 2,424 CYP2D6 fills sit at 2.168, so every one
of them would become "predict anything below 2.168, free of charge". With no gradient
under the bound, the cheapest way to satisfy 2,424 rows at once is a single constant,
and the CYP2D6 head collapses.

This is not hypothetical. §6.4 of `Cyp_NONTDI_submission_recent_results.md` records
`briford` feeding ~2,900 CYP2D6 non-inhibitors in as censored labels and watching their
predictions collapse to sigma = 0.07.

The pseudo-labels are deliberately **point** estimates — our best guess at the number
the dose-response pipeline would have printed, not a bound — so they must be trained
under ordinary MSE. `--censor drc` (the default) keeps censoring for real measurements
and disables it for pseudo rows. `--censor all` reproduces the base script's rule and
exists only to demonstrate the collapse; do not submit from it.

Sample weights
──────────────
chemprop 2.2.3's `MoleculeDatapoint.weight` is one scalar per molecule, not per task,
so a row's weight is the mean of `<ISO>_weight` over its labelled primary cells (real
measurements contribute 1.0). Under tier S every pseudo weight is already 1.0 — the
floor fill is as accurate as a measurement, MAE 0.168 — so weighting is a no-op there
and `--pseudo-weight` is the knob that matters for the tier SA/SAB follow-up, where
expected errors run 0.35-0.46.

Honest out-of-fold reporting
────────────────────────────
Pseudo-labels are a smooth function of `log2fc` and are easier to predict than real
pIC50s, so an OOF score pooled over both flatters the run. `oof_scores.csv` carries a
`source` column: **quote the `drc` rows.** §2 of the results doc records this hold-out
calling four architecture comparisons wrong, so treat even the `drc` number as an abort
gate rather than as evidence of gain — the leaderboard is the instrument.

Usage
─────
    python chemprop_nontdi_8task_pseudo.py                       # tier S, censor drc
    python chemprop_nontdi_8task_pseudo.py --smoke               # fast end-to-end check
    python chemprop_nontdi_8task_pseudo.py --train-file <name>   # a different variant
    python chemprop_nontdi_8task_pseudo.py --censor all          # reproduce the collapse
    python chemprop_nontdi_8task_pseudo.py --trunk chemeleon --max-hours 6

The training CSV must sit in `--data-dir` alongside the blinded test file.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import chemprop_nontdi_8task_ensemble as base
from chemprop_nontdi_8task_ensemble import (
    ID_COL, ISOFORMS, N_ISO, PRED_CLIP, SMILES_COL, TARGET_COLS,
    VAL_FRAC, aux_columns, rdkit2d_block,
)


def score(y_true, y_pred):
    """`base.score`, with correlation statistics suppressed on constant targets.

    Tier S pseudo-labels are all one value, so within a `source == 'pseudo'` group
    the true variance is ~0: R-squared divides by it and returns numbers like -1e30,
    and Spearman is undefined. MAE and RMSE remain meaningful and are what those rows
    should be judged on.
    """
    out = base.score(y_true, y_pred)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() >= 5 and float(np.std(y_true[m])) < 1e-6:
        out["r2"] = np.nan
        out["spearman"] = np.nan
    return out

FILE_TRAIN_PSEUDO = (
    "cyp-challenge-TRAIN_direct_inhibition_with_single_shot_LABELLED_pseudo_tierS.csv")

# Below this, a CYP2D6 predicted spread means the censored-loss collapse has happened.
# Current sigma_pred/sigma_train is 0.72 and training sd goes 0.916 -> 1.391, so the
# expectation is ~1.0. See PSEUDOLABEL_RUN_HOWTO.md section 5.
D6_SD_ABORT = 0.5


# ══════════════════════════════════════════════════════════════════════════════
# Splitters
# ══════════════════════════════════════════════════════════════════════════════

def split_stratified_safe(df, seed, val_frac):
    """`base.split_stratified` with two degenerate cases handled.

    The base version pools strata with fewer than two members into `__rare__`, but
    does not check `__rare__` itself: when exactly one stratum is a singleton, the
    pooled class also has one member and `StratifiedShuffleSplit` raises. Filling
    CYP2D6 changes the (label pattern x activity quartile) key enough to hit this --
    it is a latent bug in the base script, not a fault in the pseudo-labels, and the
    same input would break the unmodified script.

    Fixes: fold a singleton `__rare__` into the largest stratum, and fall back to a
    random split on any remaining ValueError (too many strata for the validation
    quota) rather than aborting a run mid-ensemble.
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    key = base._strata_key(df)
    vc = pd.Series(key).value_counts()
    rare = set(vc[vc < 2].index)
    key = np.array(["__rare__" if k in rare else k for k in key])

    vc = pd.Series(key).value_counts()
    if "__rare__" in vc.index and vc["__rare__"] < 2 and len(vc) > 1:
        key = np.where(key == "__rare__", vc.drop("__rare__").idxmax(), key)

    if pd.Series(key).nunique() < 2:
        return base.split_random(df, seed, val_frac)
    try:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_frac,
                                     random_state=seed)
        tr, va = next(sss.split(np.zeros(len(df)), key))
    except ValueError as exc:
        print(f"\n    stratified split unavailable ({exc}); using a random split",
              flush=True)
        return base.split_random(df, seed, val_frac)
    return np.sort(tr), np.sort(va)


SPLITTERS = {"random": base.split_random,
             "stratified": split_stratified_safe,
             "scaffold": base.split_scaffold}


# ══════════════════════════════════════════════════════════════════════════════
# Data — the substantive change
# ══════════════════════════════════════════════════════════════════════════════

def source_frame(train: pd.DataFrame) -> pd.DataFrame:
    """Per-isoform label provenance as 'drc' | 'pseudo' | '' (unlabelled).

    Falls back to all-'drc' when the columns are absent, so this script still runs
    against the original un-pseudo-labelled table and reduces to the base behaviour.
    """
    out = {}
    for iso in ISOFORMS:
        col = f"{iso}_label_source"
        labelled = train[f"{iso}_pIC50_direct_inhibition"].notna()
        if col in train.columns:
            s = train[col].fillna("").astype(str)
        else:
            s = pd.Series(np.where(labelled, "drc", ""), index=train.index)
        out[iso] = s.where(labelled, "")
    return pd.DataFrame(out, index=train.index)


def build_targets_pseudo(train: pd.DataFrame, aux_cols: list[str], aux_mode: str,
                         src: pd.DataFrame, censor: str):
    """As `base.build_targets`, then correct the censoring mask.

    censor='drc'  censor real dose-response labels at/below the top concentration,
                  never the pseudo-labels  (the correct rule -- see the module
                  docstring on why censoring a point label collapses the head)
    censor='all'  the base script's rule, for reproducing the failure
    censor='none' no censoring anywhere
    """
    prev = base.USE_CENSORED_LOSS
    base.USE_CENSORED_LOSS = censor != "none"
    try:
        y, lt = base.build_targets(train, aux_cols, aux_mode)
    finally:
        base.USE_CENSORED_LOSS = prev

    if censor == "drc":
        for j, iso in enumerate(ISOFORMS):
            lt[src[iso].to_numpy() == "pseudo", j] = False
    return y, lt


def row_weights(train: pd.DataFrame, src: pd.DataFrame,
                pseudo_weight: float) -> np.ndarray:
    """One scalar per molecule: mean of `<iso>_weight` over its labelled primaries.

    chemprop 2.2.3 carries a single weight per datapoint, not per task, so a row with
    a real CYP3A4 measurement and a pseudo CYP2D6 one gets the average of the two.
    Real measurements contribute 1.0. `pseudo_weight` scales the pseudo side only.
    """
    acc = np.zeros(len(train))
    cnt = np.zeros(len(train))
    for iso in ISOFORMS:
        s = src[iso].to_numpy()
        w = np.ones(len(train))
        wcol = f"{iso}_weight"
        if wcol in train.columns:
            w = train[wcol].fillna(1.0).to_numpy(dtype=float)
        w = np.where(s == "pseudo", w * pseudo_weight, 1.0)
        m = s != ""
        acc[m] += w[m]
        cnt[m] += 1
    return np.where(cnt > 0, acc / np.maximum(cnt, 1), 1.0)


def make_datapoints_weighted(mols, y, lt, x_d, w):
    """`base.make_datapoints` plus the per-molecule weight."""
    from chemprop import data
    n = y.shape[1]
    return [data.MoleculeDatapoint(
        mol=m,
        y=y[i].astype(float),
        weight=float(w[i]),
        x_d=None if x_d is None else x_d[i].astype(float),
        lt_mask=lt[i].astype(bool),
        gt_mask=np.zeros(n, dtype=bool),
    ) for i, m in enumerate(mols)]


def train_member(mols_tr, y_tr, lt_tr, xd_tr, w_tr,
                 mols_va, y_va, lt_va, xd_va, w_va,
                 mols_te, xd_te, ffn_hidden_dim, seed, max_epochs, log_dir, trunk,
                 batch_size, censor):
    """Mirrors `base.train_member`, threading the per-molecule weights through.

    Kept as an explicit copy rather than a monkeypatch so that any drift from the base
    script is visible in a diff. `base.build_model` reads `base.USE_CENSORED_LOSS` to
    choose BoundedMSE vs MSE, so that flag is set around the call.
    """
    import lightning.pytorch as pl
    import torch
    from lightning.pytorch.callbacks import EarlyStopping
    from chemprop import data, featurizers

    torch.set_float32_matmul_precision("high")

    feat = featurizers.SimpleMoleculeMolGraphFeaturizer()
    dset_tr = data.MoleculeDataset(
        make_datapoints_weighted(mols_tr, y_tr, lt_tr, xd_tr, w_tr), feat)
    dset_va = data.MoleculeDataset(
        make_datapoints_weighted(mols_va, y_va, lt_va, xd_va, w_va), feat)
    n_te = len(mols_te)
    dset_te = data.MoleculeDataset(
        make_datapoints_weighted(mols_te, np.full((n_te, 2 * N_ISO), np.nan),
                                 np.zeros((n_te, 2 * N_ISO), bool), xd_te,
                                 np.ones(n_te)), feat)

    target_scaler = dset_tr.normalize_targets()
    dset_va.normalize_targets(target_scaler)

    n_desc = 0
    if xd_tr is not None:
        xd_scaler = dset_tr.normalize_inputs("X_d")
        dset_va.normalize_inputs("X_d", xd_scaler)
        dset_te.normalize_inputs("X_d", xd_scaler)
        n_desc = xd_tr.shape[1]

    drop_last = (len(dset_tr) % batch_size) == 1
    load_tr = data.build_dataloader(dset_tr, batch_size=batch_size, shuffle=True,
                                    num_workers=base.NUM_WORKERS, drop_last=drop_last)
    load_va = data.build_dataloader(dset_va, batch_size=batch_size, shuffle=False,
                                    num_workers=base.NUM_WORKERS)
    load_te = data.build_dataloader(dset_te, batch_size=batch_size, shuffle=False,
                                    num_workers=base.NUM_WORKERS)

    prev = base.USE_CENSORED_LOSS
    base.USE_CENSORED_LOSS = censor != "none"
    try:
        model = base.build_model(n_desc, target_scaler, ffn_hidden_dim, seed, trunk)
    finally:
        base.USE_CENSORED_LOSS = prev

    trainer = pl.Trainer(
        max_epochs=max_epochs, accelerator="auto", devices=1,
        enable_progress_bar=False, enable_model_summary=False, logger=False,
        enable_checkpointing=False, default_root_dir=str(log_dir),
        callbacks=[EarlyStopping(monitor="val_loss", patience=base.PATIENCE,
                                 mode="min")],
        gradient_clip_val=5.0,
    )
    trainer.fit(model, load_tr, load_va)

    model.eval()
    with torch.no_grad():
        pred_va = np.vstack([p.numpy() for p in trainer.predict(model, load_va)])
        pred_te = np.vstack([p.numpy() for p in trainer.predict(model, load_te)])
    return pred_va, pred_te, int(trainer.current_epoch)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=base.DATA_DIR)
    ap.add_argument("--output-dir", type=Path,
                    default=base.OUTPUT_DIR.parent / "outputs_8task_pseudo")
    ap.add_argument("--train-file", default=FILE_TRAIN_PSEUDO,
                    help="pseudo-labelled training CSV inside --data-dir")
    ap.add_argument("--censor", choices=("drc", "all", "none"), default="drc",
                    help="'drc' censors real measurements only (correct); 'all' is "
                         "the base script's rule and collapses the CYP2D6 head; "
                         "'none' disables censoring everywhere")
    ap.add_argument("--pseudo-weight", type=float, default=1.0,
                    help="multiplier on the per-row weight of pseudo cells")
    ap.add_argument("--aux-mode", choices=("log2fc", "is_hit"), default=base.AUX_MODE)
    ap.add_argument("--trunk", choices=("scratch", "chemeleon"), default="scratch")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(base.SEEDS))
    ap.add_argument("--splits", nargs="+", default=list(base.SPLIT_METHODS),
                    choices=list(SPLITTERS))
    ap.add_argument("--epochs", type=int, default=base.MAX_EPOCHS)
    ap.add_argument("--no-rdkit2d", action="store_true")
    ap.add_argument("--batch-size", type=int, default=base.BATCH_SIZE)
    ap.add_argument("--max-hours", type=float, default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.epochs, args.seeds, args.splits = 3, args.seeds[:1], args.splits[:2]

    out = args.output_dir
    (out / "members").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"data   : {args.data_dir}")
    print(f"train  : {args.train_file}")
    print(f"output : {out}")
    print(f"censor : {args.censor}   pseudo-weight: {args.pseudo_weight}")
    print(f"aux    : {args.aux_mode}   splits: {args.splits}   seeds: {args.seeds}")
    print(f"trunk  : {args.trunk}")
    if args.trunk == "chemeleon":
        base.ensure_chemeleon()

    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")

    # base.load_frames hardcodes its filename, so read here and reuse its checks
    prev_file = base.FILE_TRAIN
    base.FILE_TRAIN = args.train_file
    try:
        train, test, aux_cols = base.load_frames(args.data_dir, args.aux_mode)
    finally:
        base.FILE_TRAIN = prev_file

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

    src = source_frame(train)
    n_pseudo = int((src == "pseudo").to_numpy().sum())
    if n_pseudo == 0:
        print("\n  WARNING: no '<iso>_label_source' == 'pseudo' cells found. This is "
              "the plain training table, so this script is equivalent to the base "
              "one. Did you mean to pass --train-file?\n")

    y_all, lt_all = build_targets_pseudo(train, aux_cols, args.aux_mode, src,
                                         args.censor)
    w_all = row_weights(train, src, args.pseudo_weight)

    print(f"  train {len(train)} compounds, test {len(test)}")
    print(f"  {'task':<42s} {'drc':>6s} {'pseudo':>7s} {'censored':>9s}  "
          f"{'mean':>6s} {'sd':>5s}")
    for j, iso in enumerate(ISOFORMS):
        col = f"{iso}_pIC50_direct_inhibition"
        v = train[col]
        print(f"  {col:<42s} {int((src[iso]=='drc').sum()):6d} "
              f"{int((src[iso]=='pseudo').sum()):7d} {int(lt_all[:, j].sum()):9d}  "
              f"{v.mean():6.3f} {v.std():5.3f}")
    for j, c in enumerate(aux_cols):
        print(f"  {c:<42s} {int(np.isfinite(y_all[:, N_ISO+j]).sum()):6d} "
              f"{'-':>7s} {'-':>9s}")
    if args.censor == "all" and n_pseudo:
        print("\n  !! --censor all marks pseudo-labels as left-censored. This is the "
              "\n     configuration that collapses the CYP2D6 head. Diagnosis only.\n")

    xd_all = xd_te = None
    if base.USE_RDKIT2D and not args.no_rdkit2d:
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
        ffn_dim = base.FFN_HIDDEN_DIMS[k % len(base.FFN_HIDDEN_DIMS)]
        print(f"[{k}/{len(members)}] {tag}: train {len(tr_idx)} / val {len(va_idx)}, "
              f"ffn_hidden={ffn_dim} ... ", end="", flush=True)

        t1 = time.time()
        pred_va, pred_te, epochs_run = train_member(
            [mols_tr_all[i] for i in tr_idx], y_all[tr_idx], lt_all[tr_idx],
            None if xd_all is None else xd_all[tr_idx], w_all[tr_idx],
            [mols_tr_all[i] for i in va_idx], y_all[va_idx], lt_all[va_idx],
            None if xd_all is None else xd_all[va_idx], w_all[va_idx],
            mols_te, xd_te, ffn_dim, sd, args.epochs, out / "logs", args.trunk,
            args.batch_size, args.censor)

        pred_te = np.clip(pred_te[:, :N_ISO], *PRED_CLIP)
        np.save(f_te, pred_te)
        va = pd.DataFrame({ID_COL: train.iloc[va_idx][ID_COL].to_numpy(),
                           "split": sp, "seed": sd})
        for j, iso in enumerate(ISOFORMS):
            va[f"{iso}_true"] = y_all[va_idx, j]
            va[f"{iso}_pred"] = np.clip(pred_va[:, j], *PRED_CLIP)
            va[f"{iso}_source"] = src[iso].to_numpy()[va_idx]
        va.to_csv(f_va, index=False)
        test_preds.append(pred_te)
        oof_rows.append(va)
        print(f"done in {time.time() - t1:.0f}s ({epochs_run} epochs)")

    if not test_preds:
        print("\nNo members completed — nothing to ensemble. "
              "Raise --max-hours or re-run.")
        return 1

    # ── out-of-fold, split by label provenance ───────────────────────────────
    oof = pd.concat(oof_rows, ignore_index=True)
    oof.to_csv(out / "oof_predictions.csv", index=False)
    rows = []
    for sp in sorted(oof["split"].unique()):
        d = oof[oof["split"] == sp]
        for iso in ISOFORMS:
            first = d.groupby(ID_COL)[f"{iso}_source"].first()
            agg = d.groupby(ID_COL).mean(numeric_only=True)
            agg = agg.assign(_src=first)
            for name, sel in (("drc", agg._src == "drc"),
                              ("pseudo", agg._src == "pseudo"),
                              ("all", agg._src != "")):
                g = agg[sel]
                if len(g) < 5:
                    continue
                rows.append(dict(split=sp, isoform=iso, source=name,
                                 **score(g[f"{iso}_true"].to_numpy(),
                                         g[f"{iso}_pred"].to_numpy())))
    scores = pd.DataFrame(rows)
    scores.to_csv(out / "oof_scores.csv", index=False)
    print("\nOut-of-fold by label provenance "
          "(quote `drc` -- pseudo rows are easier than real ones):")
    print(scores.to_string(index=False))
    drc = scores[scores.source == "drc"]
    if len(drc):
        print("\nMean Spearman on real dose-response labels, by split method:")
        print(drc.groupby("split")["spearman"].mean().round(4).to_string())

    # ── ensemble submission ──────────────────────────────────────────────────
    P = np.mean(np.stack(test_preds), axis=0)
    sub = pd.DataFrame({ID_COL: test[ID_COL], SMILES_COL: test[SMILES_COL]})
    for j, iso in enumerate(ISOFORMS):
        sub[f"{iso}_pIC50_direct_inhibition"] = P[:, j]
    sub.to_csv(out / "submission_ensemble_pseudo.csv", index=False)

    spread = np.std(np.stack(test_preds), axis=0).mean(axis=0)
    meta = dict(n_members=len(test_preds), members=[f"{s}_seed{d}" for s, d in members],
                train_file=args.train_file, censor=args.censor,
                pseudo_weight=args.pseudo_weight, n_pseudo_cells=n_pseudo,
                trunk=args.trunk, aux_mode=args.aux_mode, epochs=args.epochs,
                batch_size=args.batch_size,
                ffn_hidden_dims=list(base.FFN_HIDDEN_DIMS),
                mp_hidden_dim=base.MP_HIDDEN_DIM, mp_depth=base.MP_DEPTH,
                ffn_n_layers=base.FFN_N_LAYERS, dropout=base.DROPOUT,
                smiles_column=SMILES_COL, rdkit2d=bool(xd_all is not None),
                aux_task_weight=base.AUX_TASK_WEIGHT,
                pred_sd={i: round(float(P[:, j].std()), 4)
                         for j, i in enumerate(ISOFORMS)},
                member_disagreement_sd={i: round(float(s), 4)
                                        for i, s in zip(ISOFORMS, spread)},
                runtime_s=round(time.time() - t0, 1))
    (out / "run_metadata.json").write_text(json.dumps(meta, indent=2))

    # ── abort gate ───────────────────────────────────────────────────────────
    print(f"\npredicted pIC50 per isoform "
          f"(mean +- sd across the {len(test)} test compounds):")
    for j, iso in enumerate(ISOFORMS):
        print(f"  {iso}: {P[:, j].mean():.3f} +- {P[:, j].std():.3f}   "
              f"member disagreement sd {spread[j]:.3f}")

    d6 = float(P[:, ISOFORMS.index("CYP2D6")].std())
    print()
    if args.smoke:
        print(f"  CYP2D6 predicted sd {d6:.3f} — smoke run ({args.epochs} epochs, "
              f"{len(test_preds)} members).\n         An undertrained model predicts "
              f"near the mean, so a low spread here is expected\n         and the "
              f"abort gate is not meaningful. Judge it on a full run.")
    elif d6 < D6_SD_ABORT:
        print(f"  ABORT: CYP2D6 predicted sd {d6:.3f} < {D6_SD_ABORT}. This is the "
              f"censored-loss\n         collapse (results doc section 6.4). "
              f"DO NOT SUBMIT. Check --censor is 'drc'.")
    elif d6 < 0.8:
        print(f"  WARNING: CYP2D6 predicted sd {d6:.3f} is low (expected ~0.9-1.1). "
              f"Investigate\n           before submitting.")
    else:
        print(f"  OK: CYP2D6 predicted sd {d6:.3f} — no collapse.")

    print(f"\nwrote {out / 'submission_ensemble_pseudo.csv'}")
    print(f"total {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
