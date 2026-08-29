"""Potency-gated ChemProp + XGBoost ensemble for the CYP TDI track.

A fresh pipeline that adds the ``pIC50_TDI_condition`` potency head from
``chemprop_cyp2d6_tdi_pair_classifier.py`` to the two-arm ensemble of
``tdi_ensemble_chemprop_xgb.py``, and thresholds a **combined probability** instead of
the bare ``is_TDI`` probability. Nothing in either of those scripts is modified — the
shared primitives (features, scaffold folds, prevalence-corrected MCC, operating point,
XGBoost helpers, ChemProp fitting) are imported from ``tdi_ensemble_chemprop_xgb`` and
used read-only, so this script and both parents can be run side by side on the same VM
and their out-of-fold numbers compared on identical folds.

────────────────────────────────────────────────────────────────────────────────
WHY GATE AT ALL — THE LABEL IS A CONJUNCTION, AND ITS FACTORS ARE MEASURABLE
────────────────────────────────────────────────────────────────────────────────
The scored label is a threshold on a floored shift,

    is_TDI  <=>  max(pIC50_TDI_condition, 4.0) - max(pIC50_direct, 4.0) > log10(2)

which implies two conditions that hold *without a single exception* in the training
file (re-verified at run time by :func:`verify_gates`):

    potency gate   pIC50_TDI_condition > 4.301          (= 4.0 + log10 2)
    fitted gate    a direct-inhibition curve was fitted  (the CYP3A4 convention)

                     labelled   fail potency   fail fitted   pass both   prevalence
                                (of which +)   (of which +)              in the pass set
        CYP2D6         1497       86  (0)         4  (0)       1407         0.230
        CYP3A4         3584     1123  (0)      1249  (0)       1406         0.543

Read the CYP3A4 row twice. The two gates throw away 61% of the compounds and **keep
100% of the positives** (764/764), lifting prevalence from 0.213 to 0.543 — a 2.55x
enrichment obtained from two facts about the assay rather than from any model. A single
``is_TDI`` head has to rediscover that from the binary label alone; here each factor
gets its own supervised head and the three probabilities are multiplied back together.

The pair classifier established this idea on CYP2D6, where the potency gate is a 94/6
split and therefore nearly degenerate — which is why ``primary`` was a plausible winner
there. On CYP3A4 the same gate is a 69/31 split and the fitted gate a 65/35 split, so
this is the endpoint where the decomposition should actually pay.

────────────────────────────────────────────────────────────────────────────────
WHAT IS DIFFERENT FROM tdi_ensemble_chemprop_xgb.py
────────────────────────────────────────────────────────────────────────────────
1.  **Both arms predict three quantities**, not one:

        p_tdi      P(is_TDI)
        p_potent   P(pIC50_TDI_condition > 4.301)
        p_fitted   P(a direct curve was fitted)      [dropped if near-degenerate]

    ChemProp gets them as three supervised heads sharing one encoder (weight
    ``gate_weight``, tuned by Optuna); XGBoost gets them as three classifiers on the
    same feature matrix and the same folds. The parent script already had
    ``*_active_tdi`` / ``*_direct_fitted`` among its *auxiliary* heads, but only ever
    read head 0 — here they are decision heads whose probabilities reach the threshold.

2.  **The gate heads are defined only on rows labelled for that isoform.** This is not
    cosmetic. ``CYP2D6_pIC50_direct_inhibition`` is non-null on 1,493 of 6,145 rows;
    outside the CYP2D6 label set "no direct fit" mostly means "never assayed on this
    panel", so a dense fitted head would be a panel-membership model, and gating a test
    prediction on it would be gating on an artefact. Restricted to the labelled
    population it is what the label convention says it is.

3.  **A combiner sweep**, run per arm on the same out-of-fold probabilities and scored
    by the same prevalence-corrected MCC as everything else:

        primary                p_tdi                              (the parent's score)
        and_potent             p_tdi * p_potent
        and_potent_fitted      p_tdi * p_potent * p_fitted
        pow<a>_...             p_tdi * (prod gates) ** a          a in --power-alphas
        gate_...               p_tdi, demoted below every compound that passes

    ``pow`` interpolates between trusting the gate fully (a=1, i.e. ``and``) and barely
    at all (a=0.25); ``gate`` is the hard version. ``primary`` is in the list as the
    honest control: if it wins, the gate heads acted only as auxiliary tasks on the
    trunk and the combined probability bought nothing, which is a real result and is
    reported as such.

4.  **Optuna tunes ChemProp against the combined score**, not against head 0, so the
    search optimises the thing that is eventually thresholded.

Everything else — scaffold folds, the MBI-alert feature block, stacked auxiliary pIC50
surfaces, blend-by-rank, blind-prevalence reweighting, the theory/argmax operating
point — is the parent's, imported unchanged.

────────────────────────────────────────────────────────────────────────────────
HOW THE FINAL SCORE IS SELECTED
────────────────────────────────────────────────────────────────────────────────
    1.  XGBoost: sweep tree count on OOF MCC (head 0), then fit the two gate heads at
        the winning tree count.
    2.  ChemProp: Optuna over the parent's PXR space + ``gate_weight``, objective =
        best-combiner OOF MCC. Then 5-fold CV at the winner, then a seed ensemble on
        all data for the test set.
    3.  Per arm, sweep the combiners -> each arm keeps its own best combined score.
        (The arms are calibrated differently; forcing one combiner on both would
        confound "this gate helps" with "this arm is sharper".)
    4.  Blend the two combined scores by rank average, weight swept on OOF MCC.
    5.  Compare chemprop / xgboost / blend, pick the best, read the call rate off the
        OOF MCC-vs-rate curve, cut the test scores at that rate.

Steps 3-5 all select on the same out-of-fold sample that scores them, exactly as in the
parent; the combiner sweep adds ~11 more comparisons to that budget, so treat the
reported MCC as an upper bound and read ``combiner_comparison.csv`` for the spread
before believing a 0.005 win.

────────────────────────────────────────────────────────────────────────────────
USAGE
────────────────────────────────────────────────────────────────────────────────
    conda activate chemprop
    pip install xgboost optuna                  # not in the chemprop env by default

    python -u tdi_ensemble_potency_gated.py --quick --out-dir /tmp/gated_smoke
    python -u tdi_ensemble_potency_gated.py --isoform CYP3A4      # the one to run first
    nohup python -u tdi_ensemble_potency_gated.py > tdi_gated.log 2>&1 &

    # reuse the feature/stack caches from a parent run — saves ~1 min + ~30 min
    python -u tdi_ensemble_potency_gated.py \
        --reuse-cache-from TDI/output/ensemble_chemprop_xgb

Runtime is the parent's (3-6 h per isoform at the defaults, Optuna ~70% of it) plus two
extra XGBoost fits per fold for the gate heads, which is minutes.

Outputs (``--out-dir``, default ``TDI/output/ensemble_potency_gated/``):

    my_potency_gated_tdi_submission.csv   the file to upload
    summary.csv                           one row per isoform
    <ISO>/gate_diagnostics.csv            per-head OOF AUC, gate necessity, gate stats
    <ISO>/combiner_comparison.csv         arm x combiner, OOF MCC/AUC — read this one
    <ISO>/blend_sweep.csv                 MCC vs blend weight at the chosen combiners
    <ISO>/arm_comparison.csv              chemprop / xgboost / blend at their own points
    <ISO>/threshold_sweep.csv             MCC vs predicted-positive rate, per arm
    <ISO>/oof_predictions.csv             all three head probabilities, both arms
    <ISO>/test_scores.csv                 head probabilities, combined score, the call
    <ISO>/xgb_trees_sweep.csv             MCC/AUC vs number of trees
    <ISO>/chemprop_hpo_trials.csv         every Optuna trial and its params
    <ISO>/best_params.json                everything chosen, in one file
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:                     # importable from a notebook
    sys.path.insert(0, str(PROJECT_ROOT))

# The parent pipeline, imported read-only. Every primitive below comes from it, so the
# two scripts share folds, features and the MCC definition by construction rather than
# by copy-and-paste that drifts.
import tdi_ensemble_chemprop_xgb as base                                    # noqa: E402
from tdi_ensemble_chemprop_xgb import (                                     # noqa: E402
    ACTIVITY_CUT,
    ALL_ISOFORMS,
    BANNER,
    FILE_TEST,
    FILE_TRAIN_TDI,
    TDI_ISOFORMS,
    Context,
    best_mcc,
    build_mols,
    choose_operating_point,
    clean_columns,
    descriptor_block,
    fit_mpnn,
    make_dataset,
    mcc_at_rule,
    murcko_scaffold_folds,
    predict_proba,
    rank_pct,
    read_table,
    scale_descriptors,
    tune_xgb_trees,
    verify_label_rule,
    xgb_oof_and_test,
)

warnings.filterwarnings("ignore", category=UserWarning)


# ══════════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class GatedConfig(base.Config):
    """The parent's knobs plus the ones this script adds."""

    out_dir: Path = PROJECT_ROOT / "TDI" / "output" / "ensemble_potency_gated"

    #: Loss weight on the two gate heads. Below 1.0 so they shape the trunk without
    #: outvoting the scored head; Optuna searches over it unless --fixed-gate-weight.
    gate_weight: float = 0.5
    gate_weight_grid: list[float] = field(default_factory=lambda: [0.25, 0.5, 1.0])
    tune_gate_weight: bool = True

    #: Hard-gate cut for the ``gate_*`` combiners.
    gate_cut: float = 0.5
    #: Exponents for the soft ``pow<a>_*`` combiners (a = 1 is ``and``, already listed).
    power_alphas: list[float] = field(default_factory=lambda: [0.25, 0.5, 2.0])

    #: Which gates are allowed at all. Turning both off (``--no-gates``) reproduces the
    #: parent pipeline inside this script, on identical folds — the control run.
    use_potency_gate: bool = True
    use_fitted_gate: bool = True
    #: A gate with fewer negatives than this cannot be modelled and is dropped
    #: automatically (CYP2D6's fitted gate has 4).
    min_gate_negatives: int = 30

    #: Copy feature/stack caches from a previous run's out_dir before featurising.
    reuse_cache_from: Path | None = None
    #: Cache the stacked auxiliary matrix too — the parent recomputes it every run.
    cache_stack: bool = True

    def gate_names(self, use_fitted: bool) -> list[str]:
        return ["potent", "fitted"] if use_fitted else ["potent"]


def quick_config(**overrides) -> GatedConfig:
    """Small everything — a smoke test that exercises every path in ~10 min/isoform."""
    cfg = GatedConfig(
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
        power_alphas=[0.5],
        gate_weight_grid=[0.5],
        tune_gate_weight=False,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# ══════════════════════════════════════════════════════════════════════════════
# The gates: definition, verification, targets
# ══════════════════════════════════════════════════════════════════════════════

def gate_labels(df: pd.DataFrame, iso: str) -> dict[str, np.ndarray]:
    """The two gate targets for one isoform, as float arrays with NaN = "no target".

    Both are defined **only where ``<iso>_is_TDI`` is defined**. Restricting them to the
    labelled population is what makes them properties of the compound rather than of
    the assay panel: outside its label set an isoform's direct-inhibition column is
    mostly missing because the compound was never run, not because the curve failed.
    """
    labelled = df[f"{iso}_is_TDI"].notna().to_numpy()

    tdi_cond = df[f"{iso}_pIC50_TDI_condition"]
    potent = np.where(
        labelled & tdi_cond.notna().to_numpy(),
        (tdi_cond > ACTIVITY_CUT).astype(float),
        np.nan,
    )
    fitted = np.where(
        labelled, df[f"{iso}_pIC50_direct_inhibition"].notna().astype(float), np.nan
    )
    return {"potent": potent, "fitted": fitted}


def verify_gates(df: pd.DataFrame, iso: str) -> tuple[list[str], pd.DataFrame]:
    """Check both gates are *necessary* conditions for ``is_TDI``, and say which to use.

    A gate is only sound if no positive fails it — otherwise multiplying by the gate
    probability throws away true actives. Both hold exactly on the current release
    (0/324 CYP2D6 and 0/764 CYP3A4 positives fail either gate); a violation here is not
    fatal but it is the number that decides whether ``and``/``gate`` can be trusted, so
    it is printed rather than assumed, and a real violation raises.

    Returns the usable gate names and a one-row-per-gate diagnostic frame.
    """
    lab = df[f"{iso}_is_TDI"]
    labelled = lab.notna().to_numpy()
    y = lab.fillna(0).astype(int).to_numpy()
    gates = gate_labels(df, iso)

    rows, usable = [], []
    for name, g in gates.items():
        defined = np.isfinite(g)
        fails = labelled & defined & (g == 0)
        n_neg = int(fails.sum())
        pos_lost = int(y[fails].sum())
        frac_lost = pos_lost / max(int(y[labelled].sum()), 1)
        rows.append(
            dict(
                isoform=iso,
                gate=name,
                n_defined=int((labelled & defined).sum()),
                n_fail=n_neg,
                positives_lost=pos_lost,
                frac_positives_lost=frac_lost,
                prevalence_all=float(y[labelled].mean()),
                prevalence_passing=float(y[labelled & defined & (g == 1)].mean())
                if (labelled & defined & (g == 1)).any()
                else float("nan"),
            )
        )
        if frac_lost > 0.02:
            raise ValueError(
                f"{iso}: the '{name}' gate is not a necessary condition — it excludes "
                f"{pos_lost} of {int(y[labelled].sum())} positives. Multiplying by it "
                f"would discard true actives; re-derive the label rule before gating."
            )
        if n_neg < 1:
            print(f"    {name:7s}: no negatives at all — gate carries no information, dropped")
            continue
        usable.append(name)

    diag = pd.DataFrame(rows)
    for r in rows:
        print(
            f"    {r['gate']:7s}: {r['n_fail']:5d} of {r['n_defined']} fail the gate, "
            f"{r['positives_lost']} positives lost ({100 * r['frac_positives_lost']:.2f}%) | "
            f"prevalence {r['prevalence_all']:.3f} -> {r['prevalence_passing']:.3f} among those passing"
        )

    # Joint enrichment, which is the whole point of the decomposition.
    passing = labelled.copy()
    for name in usable:
        passing &= np.isfinite(gates[name]) & (gates[name] == 1)
    if usable:
        print(
            f"    joint  : {int(passing.sum())}/{int(labelled.sum())} pass every gate "
            f"({passing.sum() / max(labelled.sum(), 1):.3f}) holding "
            f"{int(y[passing].sum())}/{int(y[labelled].sum())} positives, "
            f"prevalence {y[passing].mean():.3f}"
        )
    return usable, diag


def usable_gates(df: pd.DataFrame, iso: str, cfg: GatedConfig) -> tuple[list[str], pd.DataFrame]:
    """Verified gates, minus any the data cannot support a model for.

    CYP2D6 has 4 compounds with no direct fit. A classifier trained on 4 negatives is
    noise, and a noisy gate multiplied into every prediction is worse than no gate, so
    it is dropped rather than modelled.
    """
    gates, diag = verify_gates(df, iso)
    enabled = {"potent": cfg.use_potency_gate, "fitted": cfg.use_fitted_gate}
    labels = gate_labels(df, iso)
    kept = []
    for name in gates:
        if not enabled[name]:
            print(f"    {name:7s}: disabled on the command line")
            continue
        g = labels[name]
        n_neg = int(np.nansum(g == 0))
        if n_neg < cfg.min_gate_negatives:
            print(
                f"    {name:7s}: only {n_neg} negatives (< {cfg.min_gate_negatives}) — "
                f"not enough to fit a gate model, dropped"
            )
            continue
        kept.append(name)
    print(f"    gates in use: {kept if kept else '(none — this reduces to the parent script)'}")
    return kept, diag


def gated_targets(
    df: pd.DataFrame, iso: str, gates: list[str], aux_weight: float, gate_weight: float
) -> tuple[np.ndarray, list[str], np.ndarray, int]:
    """ChemProp target matrix: the scored head, the gate heads, then the auxiliaries.

    Column order is fixed and relied on downstream — ``0 = is_TDI``, then one column per
    entry of ``gates``, then the auxiliary block. Only the first ``1 + len(gates)``
    columns are ever read back out; the rest exist to shape the encoder.

    The auxiliary block is the parent's, minus the columns that are now decision heads
    for *this* isoform (they would otherwise be supervised twice at two weights).
    """
    cols = [df[f"{iso}_is_TDI"].astype(float).to_numpy()]
    names = [f"{iso}_is_TDI"]
    weights = [1.0]

    labels = gate_labels(df, iso)
    for name in gates:
        cols.append(labels[name])
        names.append(f"{iso}_gate_{name}")
        weights.append(gate_weight)
    n_decision = len(cols)

    if aux_weight > 0:
        for other in ALL_ISOFORMS:
            if other == iso and "potent" in gates:
                continue                      # already a decision head above
            v = df[f"{other}_pIC50_TDI_condition"]
            cols.append(np.where(v.isna(), np.nan, (v > ACTIVITY_CUT).astype(float)))
            names.append(f"{other}_active_tdi")
            weights.append(aux_weight)
        for other in ALL_ISOFORMS:
            v = df[f"{other}_pIC50_direct_inhibition"]
            cols.append(np.where(v.isna(), np.nan, (v > ACTIVITY_CUT).astype(float)))
            names.append(f"{other}_active_direct")
            weights.append(aux_weight)
        for other in TDI_ISOFORMS:
            if other == iso and "fitted" in gates:
                continue
            cols.append(df[f"{other}_pIC50_direct_inhibition"].notna().astype(float).to_numpy())
            names.append(f"{other}_direct_fitted")
            weights.append(aux_weight)

    return np.column_stack(cols), names, np.asarray(weights, dtype=float), n_decision


# ══════════════════════════════════════════════════════════════════════════════
# Combiners — how the head probabilities become one score
# ══════════════════════════════════════════════════════════════════════════════

def combiner_specs(gates: list[str], cfg: GatedConfig) -> list[dict]:
    """Every combined score to be compared, in the order they are reported.

    With both gates available this is 1 + 2 x (1 + |alphas| + 1) = 11 candidates; with
    only the potency gate, 6. All are cheap arithmetic on probabilities already
    computed, so the sweep costs seconds — the cost is in the extra selection freedom,
    not in compute.
    """
    specs = [dict(name="primary", kind="primary", gates=[], alpha=1.0)]
    subsets = []
    if gates:
        subsets.append(gates[:1])                       # potency alone
    if len(gates) > 1:
        subsets.append(list(gates))                     # potency AND fitted
    for subset in subsets:
        tag = "_".join(subset)
        specs.append(dict(name=f"and_{tag}", kind="and", gates=subset, alpha=1.0))
        for a in cfg.power_alphas:
            specs.append(dict(name=f"pow{a:g}_{tag}", kind="power", gates=subset, alpha=float(a)))
        specs.append(dict(name=f"gate_{tag}", kind="gate", gates=subset, alpha=1.0))
    return specs


def combine(heads: dict[str, np.ndarray], spec: dict, cfg: GatedConfig) -> np.ndarray:
    """Fold ``p_tdi`` and the gate probabilities into one score.

    ``heads`` maps ``"tdi"`` / ``"potent"`` / ``"fitted"`` to probability vectors.

    The hard ``gate`` combiner *demotes* by subtracting 1.0 per failed gate rather than
    zeroing. Zeroing is what the pair classifier does and it is fine when a probability
    cut is applied, but the threshold here is a predicted-positive **rate** read off a
    quantile: a block of exact ties at 0 makes ``np.quantile`` return 0 for any rate
    past the tie, at which point ``score >= cut`` calls every compound positive.
    Subtracting keeps the ordering strict and total, and since only the ranking is used
    the two are otherwise identical.
    """
    p = np.asarray(heads["tdi"], dtype=float).copy()
    if spec["kind"] == "primary" or not spec["gates"]:
        return p

    g = np.ones_like(p)
    for name in spec["gates"]:
        g = g * np.clip(np.nan_to_num(heads[name], nan=1.0), 0.0, 1.0)

    if spec["kind"] == "and":
        return p * g
    if spec["kind"] == "power":
        return p * np.power(g, spec["alpha"])
    if spec["kind"] == "gate":
        failed = np.zeros_like(p)
        for name in spec["gates"]:
            failed += (np.nan_to_num(heads[name], nan=1.0) < cfg.gate_cut).astype(float)
        return p - failed
    raise ValueError(f"unknown combiner kind {spec['kind']!r}")


def sweep_combiners(
    heads: dict[str, np.ndarray],
    y_true: np.ndarray,
    mask: np.ndarray,
    specs: list[dict],
    p_test: float,
    cfg: GatedConfig,
) -> pd.DataFrame:
    """OOF MCC/AUC for every combiner on one arm's probabilities."""
    rows = []
    for spec in specs:
        score = combine(heads, spec, cfg)[mask]
        if not np.isfinite(score).all() or np.ptp(score) == 0:
            rows.append(dict(combiner=spec["name"], mcc=np.nan, auc=np.nan, pos_rate=np.nan,
                             mcc_argmax=np.nan))
            continue
        mcc, rate = mcc_at_rule(y_true, score, p_test, cfg)
        auc = float(roc_auc_score(y_true, score)) if len(np.unique(y_true)) > 1 else np.nan
        mcc_argmax, _ = best_mcc(y_true, score, p_test, cfg)
        rows.append(dict(combiner=spec["name"], mcc=mcc, auc=auc, pos_rate=rate,
                         mcc_argmax=mcc_argmax))
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# XGBoost arm — one classifier per head, same features, same folds
# ══════════════════════════════════════════════════════════════════════════════

def xgb_arm(
    ctx: Context, iso: str, gates: list[str], y_masked: np.ndarray, labelled: np.ndarray,
    p_test: float, cfg: GatedConfig,
) -> tuple[dict, dict, int, pd.DataFrame]:
    """``p_tdi`` (tree count swept) plus one classifier per gate at the winning count.

    The gate models reuse the scored head's tree count rather than being tuned
    separately: they are auxiliary quantities feeding a product, and tuning three
    capacities independently on the same out-of-fold sample buys a little fit and a lot
    of selection noise.
    """
    print(f"\n  XGBoost — sweeping {len(cfg.xgb_n_estimators_grid)} tree counts "
          f"({cfg.xgb_n_estimators_grid}) on {ctx.X.shape[1]} features")
    best_trees, oof_tdi, test_tdi, trees_sweep = tune_xgb_trees(
        ctx.X, y_masked, labelled, ctx.folds, ctx.X_test, p_test, cfg
    )
    heads_oof = {"tdi": oof_tdi}
    heads_test = {"tdi": test_tdi}

    labels = gate_labels(ctx.train_df, iso)
    for name in gates:
        t0 = time.time()
        y_gate = labels[name]
        oof, test = xgb_oof_and_test(
            ctx.X, y_gate, ctx.folds, ctx.X_test, best_trees, cfg, "binary", cfg.fit_test
        )
        heads_oof[name] = oof
        heads_test[name] = test
        m = np.isfinite(y_gate) & np.isfinite(oof)
        auc = (
            float(roc_auc_score(y_gate[m].astype(int), oof[m]))
            if len(np.unique(y_gate[m])) > 1 else float("nan")
        )
        print(f"  gate '{name}': n={int(np.isfinite(y_gate).sum())}  "
              f"positive={100 * np.nanmean(y_gate):.1f}%  OOF AUC={auc:.4f}  "
              f"({time.time() - t0:.0f}s)")
    return heads_oof, heads_test, int(best_trees), trees_sweep


# ══════════════════════════════════════════════════════════════════════════════
# ChemProp arm — one encoder, three decision heads
# ══════════════════════════════════════════════════════════════════════════════

def suggest_params(trial, cfg: GatedConfig) -> dict:
    """The parent's PXR search space, plus the weight on the gate heads."""
    params = base.suggest_chemprop_params(trial)
    if cfg.tune_gate_weight:
        params["gate_weight"] = trial.suggest_categorical("gate_weight", cfg.gate_weight_grid)
    else:
        params["gate_weight"] = cfg.gate_weight
    return params


def default_params(cfg: GatedConfig) -> dict:
    return dict(base.DEFAULT_CHEMPROP_PARAMS, gate_weight=cfg.gate_weight)


def chemprop_cv_heads(
    ctx: Context, iso: str, gates: list[str], params: dict, cfg: GatedConfig,
    folds_to_run: list[int], max_epochs: int, patience: int, on_fold=None,
) -> np.ndarray:
    """Out-of-fold probabilities for **every decision head**, shape (n_train, 1+|gates|).

    The parent's ``chemprop_cv`` returns head 0 only; the gate probabilities are the
    whole point here, so the full decision block comes back. Rows in folds that were not
    run stay NaN, which is what lets the Optuna objective score a subset of folds.
    """
    Y, _names, weights, n_dec = gated_targets(
        ctx.train_df, iso, gates, params.get("aux_weight", 0.25), params.get("gate_weight", 0.5)
    )
    keep = np.isfinite(Y).any(axis=1)
    oof = np.full((len(ctx.train_mols), n_dec), np.nan)

    for f in folds_to_run:
        tr = (ctx.folds != f) & keep
        # ``keep`` on the validation side too: an all-masked row makes val_loss NaN and
        # silently breaks early stopping.
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
            train_dset, val_dset, ctx.n_desc, weights, params,
            seed=cfg.scaffold_seed, cfg=cfg, max_epochs=max_epochs, patience=patience,
        )
        oof[va] = predict_proba(mpnn, trainer, val_dset, cfg)[:, :n_dec]
        if on_fold is not None:
            on_fold(f, oof)
    return oof


def heads_from_matrix(matrix: np.ndarray, gates: list[str]) -> dict[str, np.ndarray]:
    """Name the decision columns: column 0 is ``is_TDI``, then the gates in order."""
    heads = {"tdi": matrix[:, 0]}
    for j, name in enumerate(gates, start=1):
        heads[name] = matrix[:, j]
    return heads


def chemprop_hpo_gated(
    ctx: Context, iso: str, gates: list[str], specs: list[dict], p_test: float, cfg: GatedConfig
) -> tuple[dict, pd.DataFrame]:
    """Optuna TPE, scored on the **best combined** out-of-fold MCC.

    Tuning against head 0 alone would optimise a score the submission never uses. The
    objective here is the same quantity step 3 selects on — the best combiner's MCC —
    so the search, the combiner choice, the blend weight and the threshold are all one
    criterion. The trade is that the search can favour a model whose head 0 is mediocre
    but whose gates are sharp; ``chemprop_hpo_trials.csv`` records the winning combiner
    and the primary-only MCC per trial so that is visible rather than hidden.
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
    folds_to_run = list(range(min(cfg.hpo_folds, cfg.n_folds)))
    hpo_mask = labelled & np.isin(ctx.folds, folds_to_run)

    def score_matrix(matrix: np.ndarray, mask: np.ndarray) -> tuple[float, str, float]:
        """(best combined MCC, its combiner, the primary-only MCC) on ``mask``."""
        heads = heads_from_matrix(matrix, gates)
        y = y_full[mask].astype(int).to_numpy()
        best = (-np.inf, "primary")
        primary = np.nan
        for spec in specs:
            s = combine(heads, spec, cfg)[mask]
            if not np.isfinite(s).all() or np.ptp(s) == 0:
                continue
            mcc, _ = mcc_at_rule(y, s, p_test, cfg)
            if spec["name"] == "primary":
                primary = mcc
            if mcc > best[0]:
                best = (mcc, spec["name"])
        return float(best[0]), best[1], float(primary)

    def objective(trial) -> float:
        params = suggest_params(trial, cfg)

        def on_fold(f, matrix):
            done = hpo_mask & np.isfinite(matrix[:, 0])
            if done.sum() < 50 or len(np.unique(y_full[done].astype(int))) < 2:
                return
            partial, _, _ = score_matrix(matrix, done)
            trial.report(partial, step=f)
            if trial.should_prune():
                raise optuna.TrialPruned()

        matrix = chemprop_cv_heads(
            ctx, iso, gates, params, cfg, folds_to_run,
            max_epochs=cfg.hpo_max_epochs, patience=cfg.hpo_patience, on_fold=on_fold,
        )
        m = hpo_mask & np.isfinite(matrix[:, 0])
        mcc, which, primary = score_matrix(matrix, m)
        y = y_full[m].astype(int).to_numpy()
        auc = float(roc_auc_score(y, matrix[m, 0])) if len(np.unique(y)) > 1 else float("nan")
        trial.set_user_attr("auc_primary_head", auc)
        trial.set_user_attr("best_combiner", which)
        trial.set_user_attr("mcc_primary_only", primary)
        trial.set_user_attr("n_scored", int(m.sum()))
        print(f"    trial {trial.number:3d}  MCC={mcc:.4f} via '{which}' "
              f"(primary {primary:.4f})  head0 AUC={auc:.4f}  {params}")
        return mcc

    sampler = optuna.samplers.TPESampler(seed=cfg.optuna_seed, n_startup_trials=5)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    print(f"\n  Optuna: {cfg.n_trials} trials x {len(folds_to_run)} folds "
          f"({int(hpo_mask.sum())} of {int(labelled.sum())} labels scored per trial), "
          f"objective = best-combiner MCC")
    study.optimize(objective, n_trials=cfg.n_trials, gc_after_trial=True)

    rows = []
    for t in study.trials:
        rows.append(
            dict(
                number=t.number,
                state=str(t.state).split(".")[-1],
                mcc=t.value if t.value is not None else np.nan,
                mcc_primary_only=t.user_attrs.get("mcc_primary_only", np.nan),
                best_combiner=t.user_attrs.get("best_combiner", ""),
                auc_primary_head=t.user_attrs.get("auc_primary_head", np.nan),
                **t.params,
            )
        )
    trials = pd.DataFrame(rows)

    best = default_params(cfg)
    best.update(study.best_params)
    print(f"  --> best trial {study.best_trial.number}: MCC={study.best_value:.4f}")
    print(f"      {best}")
    return best, trials


def chemprop_test_heads(
    ctx: Context, iso: str, gates: list[str], params: dict, cfg: GatedConfig
) -> tuple[dict, np.ndarray]:
    """Seed ensemble on all data -> mean head probabilities on the blind set (+ spread)."""
    Y, _names, weights, n_dec = gated_targets(
        ctx.train_df, iso, gates, params.get("aux_weight", 0.25), params.get("gate_weight", 0.5)
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
            full_dset, None, ctx.n_desc, weights, params,
            seed=seed, cfg=cfg, max_epochs=cfg.max_epochs, patience=cfg.patience,
        )
        per_seed.append(predict_proba(mpnn, trainer, test_dset, cfg)[:, :n_dec])
        print(f"    seed {seed} done")
    stack = np.stack(per_seed)
    return heads_from_matrix(stack.mean(axis=0), gates), stack.std(axis=0)[:, 0]


# ══════════════════════════════════════════════════════════════════════════════
# Shared context — the parent's, plus cache reuse and a cached stack matrix
# ══════════════════════════════════════════════════════════════════════════════

def _stack_cache_path(cfg: GatedConfig, smiles: list[str], folds: np.ndarray) -> Path:
    import hashlib

    key = "|".join(smiles) + f"::{folds.tobytes().hex()[:64]}::{cfg.stack_n_estimators}"
    key += f"::{cfg.n_folds}::{sorted(cfg.xgb_seeds)}::{cfg.fit_test}"
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    cache = cfg.out_dir / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    return cache / f"stack_{h}.npz"


def import_caches(cfg: GatedConfig) -> None:
    """Copy a previous run's cached feature matrices into this run's cache directory.

    The feature blocks are keyed by a hash of the SMILES, not by the run, so a cache
    written by ``tdi_ensemble_chemprop_xgb.py`` is valid here — same molecules, same
    featuriser, same code. Saves the ~1 min featurisation and, if that run also used
    this script, the 25-40 min stacking build.
    """
    src = Path(cfg.reuse_cache_from)
    src = src / "cache" if (src / "cache").is_dir() else src
    if not src.is_dir():
        print(f"  cache reuse: {src} does not exist — ignoring")
        return
    dst = cfg.out_dir / "cache"
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for path in sorted(src.glob("*.npz")):
        target = dst / path.name
        if not target.exists():
            shutil.copy2(path, target)
            n += 1
    print(f"  cache reuse: copied {n} file(s) from {src}")


def prepare_context(cfg: GatedConfig) -> Context:
    """Load, standardise, featurise and split once for both isoforms and both arms.

    Identical to the parent's, so the folds and features are the same objects the parent
    would have built (same seed, same routine) and the two runs are comparable
    compound-by-compound. The only additions are cache import and stack caching.
    """
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    if cfg.reuse_cache_from:
        import_caches(cfg)

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
    X_all, names = base._featurize_cached(cfg, mols, df["SMILES"].tolist(), "train")
    X_test_all, _ = base._featurize_cached(cfg, test_mols, test_df["SMILES"].tolist(), "test")
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
        train_df=df, test_df=test_df, train_mols=mols, test_mols=test_mols, folds=folds,
        X=X, X_test=X_test, feature_names=names,
        x_d_train=x_d_train, x_d_test=x_d_test, n_desc=n_desc,
    )

    if cfg.use_stack_features:
        path = _stack_cache_path(cfg, df["SMILES"].tolist(), folds)
        if cfg.cache_stack and path.exists():
            blob = np.load(path, allow_pickle=True)
            S, S_test, stack_names = blob["S"], blob["S_test"], list(blob["names"])
            print(f"\nStacking features from cache {path.name}")
        else:
            S, S_test, stack_names = base.build_stack_features(df, X, X_test, folds, cfg)
            if cfg.cache_stack:
                np.savez_compressed(
                    path, S=S, S_test=S_test, names=np.array(stack_names, dtype=object)
                )
        ctx.X = np.hstack([X, np.nan_to_num(S, nan=0.0)])
        ctx.X_test = np.hstack([X_test, np.nan_to_num(S_test, nan=0.0)])
        ctx.feature_names = names + stack_names
        print(f"  XGBoost feature matrix: {ctx.X.shape}")

    return ctx


# ══════════════════════════════════════════════════════════════════════════════
# One isoform, end to end
# ══════════════════════════════════════════════════════════════════════════════

def head_auc(scores: np.ndarray, target: np.ndarray) -> float:
    m = np.isfinite(scores) & np.isfinite(target)
    if m.sum() < 20 or len(np.unique(target[m])) < 2:
        return float("nan")
    return float(roc_auc_score(target[m].astype(int), scores[m]))


def run_isoform(ctx: Context, iso: str, cfg: GatedConfig) -> dict:
    """Verify the gates, train both arms, combine, blend, threshold, write everything."""
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

    print("\n  Gate check (is every gate a necessary condition for is_TDI?)")
    gates, gate_diag = usable_gates(ctx.train_df, iso, cfg)
    specs = combiner_specs(gates, cfg)
    print(f"  combiners to compare: {[s['name'] for s in specs]}")
    gate_targets_ = gate_labels(ctx.train_df, iso)

    # ── Arm B: XGBoost ────────────────────────────────────────────────────────
    xgb_oof, xgb_test, best_trees, trees_sweep = xgb_arm(
        ctx, iso, gates, y_masked, labelled, p_test, cfg
    )
    trees_sweep.to_csv(out / "xgb_trees_sweep.csv", index=False)

    # ── Arm A: ChemProp ───────────────────────────────────────────────────────
    print("\n  ChemProp — hyperparameter search")
    t0 = time.time()
    best_params, trials = chemprop_hpo_gated(ctx, iso, gates, specs, p_test, cfg)
    trials.to_csv(out / "chemprop_hpo_trials.csv", index=False)
    print(f"  search took {(time.time() - t0) / 60:.1f} min")

    print(f"\n  ChemProp — {cfg.n_folds}-fold CV at the chosen hyperparameters")
    cp_matrix = chemprop_cv_heads(
        ctx, iso, gates, best_params, cfg, list(range(cfg.n_folds)),
        max_epochs=cfg.max_epochs, patience=cfg.patience,
    )
    cp_oof = heads_from_matrix(cp_matrix, gates)

    cp_test = {k: np.full(len(ctx.test_mols), np.nan) for k in ["tdi", *gates]}
    cp_seed_sd = np.full(len(ctx.test_mols), np.nan)
    if cfg.fit_test:
        print(f"  ChemProp — final ensemble on all data, seeds {cfg.chemprop_seeds}")
        cp_test, cp_seed_sd = chemprop_test_heads(ctx, iso, gates, best_params, cfg)

    # ── Head-level diagnostics ────────────────────────────────────────────────
    print("\n  Head AUCs (out of fold, each head against its own target)")
    print(f"    {'head':9s} {'chemprop':>9s} {'xgboost':>9s}")
    auc_cols = {"chemprop": {}, "xgboost": {}}
    for head in ["tdi", *gates]:
        target = y_masked if head == "tdi" else gate_targets_[head]
        a_cp = head_auc(cp_oof[head], target)
        a_xgb = head_auc(xgb_oof[head], target)
        auc_cols["chemprop"][head] = a_cp
        auc_cols["xgboost"][head] = a_xgb
        print(f"    {head:9s} {a_cp:9.4f} {a_xgb:9.4f}")
    gate_diag["oof_auc_chemprop"] = gate_diag["gate"].map(auc_cols["chemprop"])
    gate_diag["oof_auc_xgboost"] = gate_diag["gate"].map(auc_cols["xgboost"])
    gate_diag["in_use"] = gate_diag["gate"].isin(gates)
    gate_diag.to_csv(out / "gate_diagnostics.csv", index=False)

    # ── Combiner sweep, per arm ───────────────────────────────────────────────
    print(f"\n  Combiner sweep — OOF MCC at prevalence {p_test:.4f}")
    comb_frames, chosen_spec = [], {}
    for tag, heads in (("chemprop", cp_oof), ("xgboost", xgb_oof)):
        table = sweep_combiners(heads, y_true, labelled, specs, p_test, cfg)
        table.insert(0, "arm", tag)
        table.insert(0, "isoform", iso)
        comb_frames.append(table)
        ranked = table.dropna(subset=["mcc"]).sort_values("mcc", ascending=False)
        name = str(ranked.iloc[0]["combiner"])
        chosen_spec[tag] = next(s for s in specs if s["name"] == name)
        primary = float(table.loc[table["combiner"] == "primary", "mcc"].iloc[0])
        print(f"    {tag:9s} best '{name}' MCC={ranked.iloc[0]['mcc']:.4f} "
              f"(primary {primary:.4f}, delta {ranked.iloc[0]['mcc'] - primary:+.4f})")
        for _, r in ranked.iterrows():
            print(f"        {r['combiner']:24s} MCC={r['mcc']:.4f}  AUC={r['auc']:.4f}  "
                  f"rate={r['pos_rate']:.3f}")
    combiners_table = pd.concat(comb_frames, ignore_index=True)
    combiners_table.to_csv(out / "combiner_comparison.csv", index=False)

    if all(chosen_spec[t]["name"] == "primary" for t in ("chemprop", "xgboost")):
        print(
            "\n  NOTE: neither arm improved on the ungated probability. The gate heads\n"
            "        acted only as auxiliary tasks on the trunk, which is still a real\n"
            "        result — compare this run's MCC against tdi_ensemble_chemprop_xgb.py\n"
            "        to see whether that auxiliary supervision helped on its own."
        )

    # ── Blend the two combined scores ─────────────────────────────────────────
    cp_score_oof = combine(cp_oof, chosen_spec["chemprop"], cfg)
    xgb_score_oof = combine(xgb_oof, chosen_spec["xgboost"], cfg)
    cp_rank = rank_pct(cp_score_oof[labelled])
    xgb_rank = rank_pct(xgb_score_oof[labelled])

    blend_rows = []
    for w in cfg.blend_weight_grid:
        score = w * cp_rank + (1 - w) * xgb_rank
        mcc, rate = mcc_at_rule(y_true, score, p_test, cfg)
        mcc_argmax, _ = best_mcc(y_true, score, p_test, cfg)
        blend_rows.append(dict(weight_chemprop=float(w), mcc=mcc,
                               auc=float(roc_auc_score(y_true, score)),
                               pos_rate=rate, mcc_argmax=mcc_argmax))
    blend = pd.DataFrame(blend_rows)
    blend.to_csv(out / "blend_sweep.csv", index=False)
    best_w = float(blend.loc[blend["mcc"].idxmax(), "weight_chemprop"])
    print(f"\n  Blend weight sweep -> w_chemprop={best_w:.2f} "
          f"(MCC {blend['mcc'].max():.4f}; pure ChemProp {blend.iloc[-1]['mcc']:.4f}, "
          f"pure XGBoost {blend.iloc[0]['mcc']:.4f})")
    if best_w in (0.0, 1.0):
        print("  WARNING: the blend weight is pinned at an endpoint — the ensemble is "
              "contributing nothing; check the weaker arm's AUC before shipping it.")

    cp_score_test = combine(cp_test, chosen_spec["chemprop"], cfg)
    xgb_score_test = combine(xgb_test, chosen_spec["xgboost"], cfg)
    blend_oof = np.full(len(y), np.nan)
    blend_oof[labelled] = best_w * cp_rank + (1 - best_w) * xgb_rank
    arms = {
        "chemprop": (cp_score_oof, cp_score_test),
        "xgboost": (xgb_score_oof, xgb_score_test),
        "blend": (
            blend_oof,
            best_w * rank_pct(cp_score_test) + (1 - best_w) * rank_pct(xgb_score_test),
        ),
    }

    # ── Arm comparison and operating point ────────────────────────────────────
    print(f"\n  Arm comparison ({iso}) — MCC at prevalence {p_test:.4f}")
    arm_rows, sweeps, ops = [], [], {}
    for tag, (oof_scores, _t) in arms.items():
        op = choose_operating_point(y_true, oof_scores[labelled], p_test, cfg)
        sweeps.append(op.pop("sweep").assign(isoform=iso, arm=tag))
        ops[tag] = op
        arm_rows.append(dict(
            isoform=iso, arm=tag,
            combiner=chosen_spec.get(tag, {"name": "blend"})["name"], **op,
        ))
        print(
            f"    {tag:9s} AUC={op['auc']:.4f}  MCC={op['expected_mcc']:.4f} "
            f"@ rate={op['pos_rate']:.3f}  P={op['precision']:.3f} R={op['recall']:.3f}  "
            f"(theory rate {op['theory_rate']:.3f}, argmax IQR "
            f"[{op['argmax_iqr_low']:.2f},{op['argmax_iqr_high']:.2f}])"
        )
    arm_table = pd.DataFrame(arm_rows)
    arm_table.to_csv(out / "arm_comparison.csv", index=False)
    pd.concat(sweeps).to_csv(out / "threshold_sweep.csv", index=False)

    chosen_arm = str(arm_table.loc[arm_table["expected_mcc"].idxmax(), "arm"])
    op = ops[chosen_arm]
    print(f"  --> chose arm '{chosen_arm}'")

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
    oof_frame = pd.DataFrame({
        "Molecule_Name": ctx.train_df["Molecule_Name"].to_numpy(),
        "fold": ctx.folds,
        f"{iso}_true": np.where(labelled, y, np.nan),
    })
    for head in ["tdi", *gates]:
        oof_frame[f"{iso}_chemprop_p_{head}"] = cp_oof[head]
        oof_frame[f"{iso}_xgboost_p_{head}"] = xgb_oof[head]
    for name in gates:
        oof_frame[f"{iso}_gate_{name}_true"] = gate_targets_[name]
    for tag, (oof_scores, _t) in arms.items():
        oof_frame[f"{iso}_{tag}"] = oof_scores
    oof_frame.to_csv(out / "oof_predictions.csv", index=False)

    calls = None
    if cfg.fit_test:
        test_score = arms[chosen_arm][1]
        cut = float(np.quantile(test_score, 1 - op["pos_rate"]))
        calls = (test_score >= cut).astype(bool)
        test_frame = pd.DataFrame({
            "SMILES": ctx.test_df["SMILES"].to_numpy(),
            "Molecule_Name": ctx.test_df["Molecule_Name"].to_numpy(),
        })
        for head in ["tdi", *gates]:
            test_frame[f"{iso}_chemprop_p_{head}"] = cp_test[head]
            test_frame[f"{iso}_xgboost_p_{head}"] = xgb_test[head]
        test_frame[f"{iso}_chemprop_seed_sd"] = cp_seed_sd
        test_frame[f"{iso}_chemprop"] = cp_score_test
        test_frame[f"{iso}_xgboost"] = xgb_score_test
        test_frame[f"{iso}_blend"] = arms["blend"][1]
        test_frame[f"{iso}_score"] = test_score
        test_frame[f"{iso}_is_TDI"] = calls
        test_frame.to_csv(out / "test_scores.csv", index=False)
        gate_pass = np.ones(len(calls), dtype=float)
        for name in gates:
            gate_pass *= (np.nan_to_num(cp_test[name], nan=1.0) >= cfg.gate_cut).astype(float)
        print(
            f"  test: calling {int(calls.sum())}/{len(calls)} positive ({calls.mean():.4f}) "
            f"against an expected {p_test:.4f} ({int(round(p_test * len(calls)))} compounds); "
            f"{int(gate_pass.sum())} test compounds pass every gate (ChemProp heads)"
        )

    summary = dict(
        isoform=iso,
        n_labels=int(labelled.sum()),
        train_prevalence=float(y_true.mean()),
        eval_prevalence=float(p_test),
        gates=",".join(gates) if gates else "none",
        combiner_chemprop=chosen_spec["chemprop"]["name"],
        combiner_xgboost=chosen_spec["xgboost"]["name"],
        arm=chosen_arm,
        blend_weight_chemprop=best_w,
        xgb_n_estimators=int(best_trees),
        auc=float(op["auc"]),
        mcc=float(op["expected_mcc"]),
        mcc_primary_only=float(
            combiners_table.query("arm == @chosen_arm and combiner == 'primary'")["mcc"].iloc[0]
        ) if chosen_arm != "blend" else float("nan"),
        pos_rate=float(op["pos_rate"]),
        precision=float(op["precision"]),
        recall=float(op["recall"]),
        chemprop_params=best_params,
    )
    (out / "best_params.json").write_text(json.dumps(summary, indent=2, default=str))

    return dict(summary=summary, oof=oof_frame, arms=arm_table, blend_sweep=blend,
                combiners=combiners_table, trees_sweep=trees_sweep, hpo_trials=trials,
                gate_diagnostics=gate_diag, calls=calls)


# ══════════════════════════════════════════════════════════════════════════════
# Submission
# ══════════════════════════════════════════════════════════════════════════════

SUBMISSION_NAME = "my_potency_gated_tdi_submission.csv"


def write_submission(cfg: GatedConfig, ctx: Context | None = None) -> Path | None:
    """Assemble the two-column submission from whatever per-isoform runs exist.

    Reads ``<iso>/test_scores.csv``, so ``--isoform CYP3A4`` today and ``--isoform
    CYP2D6`` tomorrow still produce one valid file.
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

    base_frame = next(iter(frames.values()))[["SMILES", "Molecule_Name"]].copy()
    for iso, frame in frames.items():
        base_frame = base_frame.merge(
            frame[["Molecule_Name", f"{iso}_is_TDI"]], on="Molecule_Name", how="left"
        )
        base_frame[f"{iso}_is_TDI"] = base_frame[f"{iso}_is_TDI"].astype(bool)

    path = cfg.out_dir / SUBMISSION_NAME
    base_frame.to_csv(path, index=False)
    print(f"\n  submission -> {path}  ({len(base_frame)} rows)")

    if len(frames) < len(TDI_ISOFORMS):
        print("  NOTE: only one isoform present — this file is not yet submittable.")
        return path

    try:
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
    p.add_argument("--reuse-cache-from", type=Path, default=None,
                   help="a previous run's out_dir (or its cache/) to copy feature caches from")
    p.add_argument("--folds", type=int, default=None, help="scaffold CV folds")
    p.add_argument("--n-trials", type=int, default=None, help="Optuna trials for ChemProp")
    p.add_argument("--hpo-folds", type=int, default=None, help="folds scored per Optuna trial")
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--xgb-trees", type=int, nargs="+", default=None,
                   help="up to 4 values for n_estimators (default 200 400 800 1600)")
    p.add_argument("--chemprop-seeds", type=int, nargs="+", default=None)
    p.add_argument("--prevalence", choices=["blind", "train"], default=None)
    p.add_argument("--operating-point", choices=["theory", "argmax"], default=None,
                   help="how the MCC-optimal call rate is read off the OOF curve")
    # ── gate-specific ─────────────────────────────────────────────────────────
    p.add_argument("--no-fitted-gate", action="store_true",
                   help="use only the pIC50_TDI_condition potency gate")
    p.add_argument("--no-gates", action="store_true",
                   help="control run: no gate heads at all, i.e. the parent pipeline "
                        "re-run inside this script — the honest baseline to compare against")
    p.add_argument("--gate-weight", type=float, default=None,
                   help="fix the loss weight on the gate heads instead of tuning it")
    p.add_argument("--gate-cut", type=float, default=None,
                   help="probability cut for the hard gate_* combiners (default 0.5)")
    p.add_argument("--power-alphas", type=float, nargs="*", default=None,
                   help="exponents for the soft pow<a>_* combiners (default 0.25 0.5 2)")
    p.add_argument("--combiner", default=None,
                   help="force one combiner by name instead of sweeping (e.g. and_potent)")
    # ── parent flags ──────────────────────────────────────────────────────────
    p.add_argument("--no-stack", action="store_true", help="drop the stacked auxiliary features")
    p.add_argument("--no-descriptors", action="store_true", help="ChemProp on the graph only")
    p.add_argument("--no-test", action="store_true", help="out-of-fold only, no submission")
    p.add_argument("--quick", action="store_true", help="tiny run that exercises every path")
    p.add_argument("--progress", action="store_true")
    return p.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> GatedConfig:
    cfg = quick_config() if args.quick else GatedConfig()
    if args.data_dir:
        cfg.data_dir = args.data_dir
    if args.out_dir:
        cfg.out_dir = args.out_dir
    if args.reuse_cache_from:
        cfg.reuse_cache_from = args.reuse_cache_from
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
    if args.no_fitted_gate:
        cfg.use_fitted_gate = False
    if args.no_gates:
        cfg.use_potency_gate = False
        cfg.use_fitted_gate = False
    if args.gate_weight is not None:
        cfg.gate_weight = args.gate_weight
        cfg.tune_gate_weight = False
    if args.gate_cut is not None:
        cfg.gate_cut = args.gate_cut
    if args.power_alphas is not None:
        cfg.power_alphas = list(args.power_alphas)
    if args.no_stack:
        cfg.use_stack_features = False
    if args.no_descriptors:
        cfg.use_descriptors = False
    if args.no_test:
        cfg.fit_test = False
    cfg.progress = args.progress
    cfg.isoforms = list(TDI_ISOFORMS) if args.isoform == "both" else [args.isoform]
    return cfg


def apply_forced_combiner(cfg: GatedConfig, name: str | None) -> None:
    """``--combiner X`` collapses the sweep to a single candidate.

    Implemented by filtering :func:`combiner_specs`, so the rest of the pipeline —
    including the Optuna objective — sees exactly one combiner and cannot select
    another. Useful for a clean A/B against a previous run.
    """
    if not name:
        return
    original = combiner_specs

    def filtered(gates: list[str], cfg_: GatedConfig) -> list[dict]:
        specs = original(gates, cfg_)
        keep = [s for s in specs if s["name"] == name]
        if not keep:
            raise SystemExit(
                f"--combiner {name!r} is not among the available combiners: "
                f"{[s['name'] for s in specs]}"
            )
        return keep

    globals()["combiner_specs"] = filtered


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = config_from_args(args)
    apply_forced_combiner(cfg, args.combiner)

    t0 = time.time()
    ctx = prepare_context(cfg)

    summaries = []
    for iso in cfg.isoforms:
        summaries.append(run_isoform(ctx, iso, cfg)["summary"])

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
