"""Train a multitask LightGBM model for the CYP direct-inhibition track.

Script version of ``notebooks/multitask_activity_prediction.ipynb`` — same pipeline, no
Jupyter. Run it end to end and it will scaffold-split the training set, compare the
multitask formulations with the challenge's own scorer, refit the winner on all the data,
and write a validated submission file.

Built on `openadmet-models <https://github.com/OpenADMET/openadmet-models>`_:

* ``FeatureConcatenator`` (ECFP4 fingerprints + RDKit 2D descriptors),
* ``ScaffoldSplitter`` (Bemis-Murcko scaffold split, backed by ``splito``),
* LightGBM architectures, compared head to head:

  ``A`` — ``MultiOutputLGBMRegressorModel``: one LGBM per isoform, trained in a single
  call with shared hyperparameters, NaN rows masked per task. The trees are *not*
  shared, so this is the honest single-task baseline.

  ``B`` — ``LGBMRegressorModel`` over task-stacked data: one shared LGBM over all
  ``(compound, isoform)`` pairs with a one-hot isoform block appended to the features.
  Every tree sees every isoform's measurements — a genuine shared-representation
  multitask GBDT, and the formulation that suits this very sparse label matrix.

  ``C`` — ``B`` plus auxiliary CYP substrate-classification heads, stacked in as extra
  tasks from the Ni et al. 2025 curated dataset (see ``cyp_substrate_aux.py``). Enabled
  with ``--aux-substrate``. Substrate status is a *different* endpoint from inhibition,
  so this is never extra pIC50 data; the bet is that making the shared trees also
  separate substrates from non-substrates buys the primary tasks a better representation.
  Model ``A`` gains nothing from auxiliary tasks by construction — its per-task LGBMs
  share no structure, so extra output columns would just train estimators nobody queries.

The hold-out is scored with ``evaluation/`` (ported from the leaderboard backend), so
ST-RAE / MAE / R2 / Spearman / Kendall come out on the same scale as the leaderboard.

Environment
-----------
Needs the ``openadmet-models`` environment, not ``oadmet_cyp_tutorial``::

    conda activate openadmet-models

Examples
--------
Full run — compare the models over three scaffold splits, then submit-ready output::

    python train_multitask_cyp.py

Quick run — one split, no plots::

    python train_multitask_cyp.py --seeds 42

Skip the comparison and go straight to fitting the multitask model on everything::

    python train_multitask_cyp.py --no-holdout-eval --final-model B

Test whether auxiliary substrate heads help — adds model ``C`` to the comparison::

    python cyp_substrate_aux.py --download          # one-off, ~1.6 MB from Figshare
    python train_multitask_cyp.py --aux-substrate

Score the hold-out predictions this script writes (or, later, a real submission against
the released answers) with the standalone scorer::

    python evaluate_submission.py \
        --predictions outputs/holdout_predictions_multitask.csv \
        --ground-truth outputs/holdout_ground_truth.csv
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger  # noqa: E402

# openadmet-models: importing the registries populates the model/featurizer/splitter
# catalogues, the same way the Anvil workflow runner does.
from openadmet.models.registries import *  # noqa: E402,F401,F403
from openadmet.models.architecture.lgbm import (  # noqa: E402
    LGBMRegressorModel,
    MultiOutputLGBMRegressorModel,
)
from openadmet.models.features.combine import FeatureConcatenator  # noqa: E402
from openadmet.models.split.scaffold import ScaffoldSplitter  # noqa: E402

# The challenge's own scorer and submission validator, straight from this repo.
from evaluation.config import ACTIVITY_METRICS, REGRESSION_ENDPOINTS  # noqa: E402
from evaluation.evaluate_predictions import (  # noqa: E402
    add_macro_endpoint,
    average_bootstrap_results_by_endpoint,
    score_activity_predictions,
)
from validation.activity_validation import validate_activity_submission  # noqa: E402

# Auxiliary CYP substrate labels (Ni et al. 2025), used only when --aux-substrate is set.
from cyp_substrate_aux import (  # noqa: E402
    ALL_ISOFORMS,
    CHALLENGE_ISOFORMS,
    DEFAULT_AUX_DIR,
    drop_holdout_leakage,
    load_substrate_labels,
    map_labels_to_target_scale,
)

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------

CYP_ISOFORMS = ["CYP1A2", "CYP2C9", "CYP2D6", "CYP3A4"]
TARGET_COLS = [f"{cyp}_pIC50_direct_inhibition" for cyp in CYP_ISOFORMS]
CONF_COLS = [f"{col}{suffix}" for col in TARGET_COLS for suffix in ("_conf_low", "_conf_high")]
N_TASKS = len(TARGET_COLS)

# The scorer's endpoint list must line up with our column order, or the per-endpoint
# results would be silently mislabelled.
assert TARGET_COLS == list(REGRESSION_ENDPOINTS)

HF_TRAIN = "hf://datasets/openadmet/cyp-challenge-train-test/cyp-challenge-TRAIN_inhibition.csv"
HF_TEST = "hf://datasets/openadmet/cyp-challenge-train-test/cyp-challenge-TEST-BLINDED.csv"

# Shared hyperparameters for both models, so the A/B comparison is about the multitask
# formulation and nothing else. Sensible defaults for a ~2k-feature, ~5k-row problem —
# tune them (Optuna, or openadmet-models' SKLearnGridSearchTrainer) before chasing the
# leaderboard.
LGBM_PARAMS = dict(
    n_estimators=800,
    learning_rate=0.03,
    num_leaves=63,
    min_child_samples=10,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.4,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

MODEL_LABELS = {
    "A": "A: per-task",
    "B": "B: multitask",
    "C": "C: multitask + substrate aux",
}


# --------------------------------------------------------------------------------------
# Data and features
# --------------------------------------------------------------------------------------


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the dose-response training set and the blinded test set from Hugging Face."""
    train_df = pd.read_csv(HF_TRAIN)
    test_df = pd.read_csv(HF_TEST)
    print(f"train: {len(train_df)} compounds, test: {len(test_df)} compounds")
    labels = train_df[TARGET_COLS]
    coverage = pd.DataFrame(
        {
            "n_measured": labels.notna().sum().values,
            "pct_of_compounds": (100 * labels.notna().mean()).round(1).values,
        },
        index=CYP_ISOFORMS,
    )
    print("\nLabel coverage (the matrix is sparse — this is why multitask helps):")
    print(coverage.to_string())
    print("\nIsoforms measured per compound:")
    print(labels.notna().sum(axis=1).value_counts().sort_index().to_string())
    return train_df, test_df


def build_featurizer() -> FeatureConcatenator:
    """ECFP4 fingerprints concatenated with RDKit 2D descriptors."""
    return FeatureConcatenator(
        featurizers={
            "FingerprintFeaturizer": {"fp_type": "ecfp:4"},
            "DescriptorFeaturizer": {"descr_type": "desc2d"},
        }
    )


def featurize(smiles, tag: str, featurizer: FeatureConcatenator, cache_dir: Path) -> np.ndarray:
    """Featurize a list of SMILES, caching the result as a .npy file.

    Raises if any molecule fails to featurize: ``FeatureConcatenator`` returns the
    *intersection* of the indices each featurizer succeeded on but concatenates the full
    arrays, so silently dropping a molecule here would misalign features and labels.
    """
    cache_file = Path(cache_dir) / f"X_{tag}_ecfp4_desc2d.npy"
    if cache_file.exists():
        X = np.load(cache_file)
        print(f"{tag}: loaded cached features {X.shape}")
        return X

    smiles = list(smiles)
    start = time.time()
    X, ok_idx = featurizer.featurize(smiles)
    if len(ok_idx) != len(smiles):
        failed = sorted(set(range(len(smiles))) - set(np.asarray(ok_idx).tolist()))
        raise RuntimeError(f"featurization failed for {len(failed)} molecule(s): {failed[:10]}")

    # A handful of RDKit descriptors (e.g. Ipc) can overflow to +/-inf; LightGBM tolerates
    # NaN but not inf, so squash both to 0.
    X = np.nan_to_num(np.asarray(X, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_file, X)
    print(f"{tag}: featurized {X.shape} in {time.time() - start:.0f}s -> {cache_file.name}")
    return X


def featurize_lenient(smiles, tag: str, featurizer: FeatureConcatenator, cache_dir: Path):
    """Featurize SMILES, dropping rather than raising on failures.

    ``featurize`` is deliberately strict, because a silently dropped molecule there would
    misalign the challenge features from the challenge labels. Auxiliary data is different:
    it is external, best-effort, and losing a few rows costs nothing — so this variant
    returns the surviving feature rows alongside the positional indices they came from, and
    the caller subsets its label matrix to match.
    """
    cache_file = Path(cache_dir) / f"X_{tag}_ecfp4_desc2d.npz"
    if cache_file.exists():
        cached = np.load(cache_file)
        print(f"{tag}: loaded cached features {cached['X'].shape}")
        return cached["X"], cached["ok_idx"]

    smiles = list(smiles)
    start = time.time()
    X, ok_idx = featurizer.featurize(smiles)
    ok_idx = np.sort(np.asarray(ok_idx, dtype=int))
    X = np.nan_to_num(np.asarray(X, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if len(ok_idx) != len(smiles):
        print(f"{tag}: {len(smiles) - len(ok_idx)} molecule(s) failed featurization, dropped")

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_file, X=X, ok_idx=ok_idx)
    print(f"{tag}: featurized {X.shape} in {time.time() - start:.0f}s -> {cache_file.name}")
    return X, ok_idx


def build_aux_bundle(
    featurizer: FeatureConcatenator,
    cache_dir: Path,
    aux_dir: Path,
    isoforms: list[str],
    weight: float,
    spread: float,
) -> dict:
    """Load, featurize and package the auxiliary substrate tasks.

    Returns
    -------
    dict
        ``X`` (n, d) features, ``Y`` (n, n_aux) labels with NaN for unassayed isoform,
        ``smiles`` / ``scaffold`` for hold-out filtering, and the ``weight`` / ``spread``
        hyperparameters that control how hard the auxiliary rows pull on the shared trees.

    """
    print(f"\nAuxiliary substrate tasks from {aux_dir}")
    aux_df = load_substrate_labels(aux_dir, isoforms)
    label_cols = [f"{iso}_is_substrate" for iso in isoforms]

    X_aux, ok_idx = featurize_lenient(aux_df["SMILES"], "aux_substrate", featurizer, cache_dir)
    aux_df = aux_df.iloc[ok_idx].reset_index(drop=True)

    return {
        "X": X_aux,
        "Y": aux_df[label_cols].to_numpy(dtype=np.float64),
        "smiles": aux_df["SMILES"].to_numpy(),
        "scaffold": aux_df["scaffold"].to_numpy(),
        "isoforms": list(isoforms),
        "weight": weight,
        "spread": spread,
    }


def aux_for_holdout(aux: dict | None, holdout_smiles) -> dict | None:
    """Subset an auxiliary bundle to the compounds safe to train on for a given hold-out.

    See ``cyp_substrate_aux.drop_holdout_leakage`` — the substrate labels leak no pIC50
    information, but an auxiliary compound sharing a scaffold with the hold-out would
    still put that core in front of the shared trees and undercut the split's premise.
    """
    if aux is None:
        return None
    frame = pd.DataFrame({"SMILES": aux["smiles"], "scaffold": aux["scaffold"]})
    frame["position"] = np.arange(len(frame))
    kept = drop_holdout_leakage(frame, holdout_smiles)["position"].to_numpy()
    return {**aux, "X": aux["X"][kept], "Y": aux["Y"][kept], "smiles": aux["smiles"][kept]}


# --------------------------------------------------------------------------------------
# Scaffold splitting
# --------------------------------------------------------------------------------------


def scaffold_split_indices(smiles, test_size: float = 0.2, random_state: int = 42):
    """Return (train_idx, holdout_idx) positional indices from a Bemis-Murcko scaffold split.

    Whole scaffold groups stay on one side of the split, so no hold-out compound shares a
    core with a training compound. The splitter returns the split *data* rather than
    indices, so ``np.arange(n)`` is passed as ``y`` and the indices read back out of it.
    """
    smiles = np.asarray(list(smiles))
    splitter = ScaffoldSplitter(
        train_size=1.0 - test_size,
        val_size=0.0,
        test_size=test_size,
        random_state=random_state,
    )
    _, _, _, train_idx, _, holdout_idx, _ = splitter.split(smiles, np.arange(len(smiles)))
    return np.sort(train_idx), np.sort(holdout_idx)


# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------


def fit_predict_multioutput(X_fit, Y_fit, X_pred, params: dict = LGBM_PARAMS, aux: dict | None = None):
    """Model A: train ``MultiOutputLGBMRegressorModel`` and predict all tasks.

    One LGBM per target column; NaN rows are masked per task inside the estimator, so a
    compound measured only against CYP3A4 still contributes to CYP3A4.

    ``aux`` is accepted for a uniform dispatch signature and ignored: the per-task
    estimators share no structure, so auxiliary output columns could not influence the
    primary predictions even in principle.

    Returns
    -------
    tuple
        ``(model, predictions)`` with predictions of shape ``(n_samples, n_tasks)``.

    """
    model = MultiOutputLGBMRegressorModel(n_tasks=N_TASKS, **params)
    model.train(X_fit, Y_fit)
    return model, model.predict(X_pred)


def stack_tasks(X, Y=None, n_tasks: int = N_TASKS, n_task_cols: int | None = None, task_offset: int = 0):
    """Expand ``(n, d)`` features to long format: ``(n * n_tasks, d + n_task_cols)`` with a one-hot task block.

    ``n_task_cols`` defaults to ``n_tasks``; pass a larger value (with ``task_offset``) to
    place these tasks inside a wider one-hot block shared with other task groups, which is
    how the auxiliary substrate tasks and the primary pIC50 tasks end up in one design
    matrix. Prediction stacks only the primary tasks, leaving the auxiliary columns zero.

    Returns ``(X_long, row_index, task_index, y_long)``. When ``Y`` is given, rows whose
    label is NaN (compound not measured against that isoform) are dropped and ``y_long``
    is returned; when it is None, every ``(compound, task)`` pair is kept and ``y_long``
    is None. ``task_index`` is local to this group — it does *not* include ``task_offset``
    — so it can index straight into a predictions array.

    Note this materialises ``n * n_tasks`` rows — ~175 MB for the full training set at
    ~2.2k float32 features. Chunk the prediction side if you add many more tasks or
    features.
    """
    X = np.asarray(X, dtype=np.float32)
    n = X.shape[0]
    n_task_cols = n_tasks if n_task_cols is None else n_task_cols

    one_hot = np.zeros((n_tasks, n_task_cols), dtype=np.float32)
    one_hot[np.arange(n_tasks), task_offset + np.arange(n_tasks)] = 1.0
    task_block = np.tile(one_hot, (n, 1))

    X_long = np.concatenate([np.repeat(X, n_tasks, axis=0), task_block], axis=1)
    row_index = np.repeat(np.arange(n), n_tasks)
    task_index = np.tile(np.arange(n_tasks), n)

    if Y is None:
        return X_long, row_index, task_index, None

    y_long = np.asarray(Y, dtype=np.float64).reshape(-1)
    measured = np.isfinite(y_long)
    return X_long[measured], row_index[measured], task_index[measured], y_long[measured]


def fit_predict_stacked(X_fit, Y_fit, X_pred, params: dict = LGBM_PARAMS, aux: dict | None = None):
    """Models B and C: train one shared LGBM over stacked (compound, task) rows; predict all tasks.

    Every tree split is estimated from all four isoforms' measurements at once, and the
    one-hot task block lets the trees specialise where the isoforms genuinely differ.

    Passing ``aux`` turns this into model C: the auxiliary substrate rows are appended with
    their own one-hot task columns, their binary labels mapped onto the pIC50 scale (see
    ``map_labels_to_target_scale``), and a per-row ``sample_weight`` of ``aux["weight"]``
    so their influence on the shared trees can be dialled down. The mapping is computed
    from ``y_long`` — this training fold's primary labels only — so no hold-out statistic
    reaches the model.

    Returns
    -------
    tuple
        ``(model, predictions)`` with predictions of shape ``(n_samples, n_tasks)``.

    """
    n_aux_tasks = 0 if aux is None else aux["Y"].shape[1]
    n_task_cols = N_TASKS + n_aux_tasks

    X_long, _, _, y_long = stack_tasks(X_fit, Y_fit, n_tasks=N_TASKS, n_task_cols=n_task_cols)
    weights = np.ones(len(y_long), dtype=np.float64)

    if aux is not None:
        X_aux_long, _, _, labels = stack_tasks(
            aux["X"], aux["Y"], n_tasks=n_aux_tasks, n_task_cols=n_task_cols, task_offset=N_TASKS
        )
        y_aux = map_labels_to_target_scale(labels, y_long, aux["spread"])
        X_long = np.concatenate([X_long, X_aux_long])
        y_long = np.concatenate([y_long, y_aux])
        weights = np.concatenate([weights, np.full(len(y_aux), aux["weight"], dtype=np.float64)])
        print(
            f"  stacked rows: {len(y_long) - len(y_aux)} primary + {len(y_aux)} auxiliary "
            f"(weight {aux['weight']}, spread {aux['spread']})"
        )

    model = LGBMRegressorModel(**params)
    # LGBMModelBase.train() takes no sample_weight, so build and fit the estimator directly.
    model.build()
    model.estimator = model.estimator.fit(X_long, y_long, sample_weight=weights)

    X_pred_long, row_index, task_index, _ = stack_tasks(
        X_pred, n_tasks=N_TASKS, n_task_cols=n_task_cols
    )
    flat = np.asarray(model.predict(X_pred_long)).ravel()
    preds = np.empty((np.asarray(X_pred).shape[0], N_TASKS), dtype=float)
    preds[row_index, task_index] = flat
    return model, preds


def fit_predict(key: str, X_fit, Y_fit, X_pred, params: dict, aux: dict | None = None):
    """Dispatch to the model named by ``key``; ``aux`` reaches only model C."""
    if key == "A":
        return fit_predict_multioutput(X_fit, Y_fit, X_pred, params)
    if key == "B":
        return fit_predict_stacked(X_fit, Y_fit, X_pred, params, aux=None)
    if key == "C":
        if aux is None:
            raise ValueError("model C needs auxiliary data — pass --aux-substrate")
        return fit_predict_stacked(X_fit, Y_fit, X_pred, params, aux=aux)
    raise ValueError(f"unknown model key {key!r}")


# --------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------


def make_prediction_frame(pred, molecule_names) -> pd.DataFrame:
    """Wrap an ``(n, n_tasks)`` prediction array in the submission-style frame the scorer wants."""
    frame = pd.DataFrame(np.asarray(pred), columns=TARGET_COLS)
    frame.insert(0, "Molecule_Name", np.asarray(molecule_names))
    return frame


def make_ground_truth_frame(train_df: pd.DataFrame, indices) -> pd.DataFrame:
    """Slice the training set into an answers frame: labels + credible-interval bounds."""
    return train_df.iloc[indices][["Molecule_Name", *TARGET_COLS, *CONF_COLS]].reset_index(drop=True)


def score_holdout(train_df: pd.DataFrame, pred, indices) -> pd.DataFrame:
    """Score hold-out predictions with the official challenge scorer.

    Returns a DataFrame indexed by endpoint (plus the macro-averaged ``MA`` row) with
    ``<metric>_mean`` / ``<metric>_std`` columns over the 1,000 bootstrap samples.
    """
    predictions = make_prediction_frame(pred, train_df["Molecule_Name"].to_numpy()[indices])
    ground_truth = make_ground_truth_frame(train_df, indices)
    bootstrap = score_activity_predictions(predictions, ground_truth, REGRESSION_ENDPOINTS)
    bootstrap = add_macro_endpoint(bootstrap, REGRESSION_ENDPOINTS, ACTIVITY_METRICS)
    return average_bootstrap_results_by_endpoint(bootstrap)


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    """Format a scorer result table as ``mean ± std``, primary metric first."""
    out = pd.DataFrame(index=results.index)
    for metric in ["ST-RAE", "MAE", "R2", "Spearman_R", "Kendall_Tau"]:
        out[metric] = [
            f"{m:.3f} ± {s:.3f}"
            for m, s in zip(results[f"{metric}_mean"], results[f"{metric}_std"])
        ]
    return out


def evaluate_over_scaffold_splits(
    train_df: pd.DataFrame,
    X: np.ndarray,
    Y: np.ndarray,
    seeds: list[int],
    holdout_fraction: float,
    params: dict,
    model_keys: tuple[str, ...] = ("A", "B"),
    aux: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Train and score each model on a scaffold split per seed.

    A single scaffold split is one draw — which scaffolds land in the hold-out moves the
    numbers by more than the between-model gap usually is — so repeat over a few seeds
    before believing a winner.

    The auxiliary bundle is re-filtered per split: compounds sharing a scaffold with *this*
    seed's hold-out are removed before training, so the auxiliary tasks never smuggle a
    hold-out core into the shared trees.

    Returns
    -------
    tuple
        ``(per_seed_results, first_split)`` where ``first_split`` carries the detailed
        tables and predictions of the first seed, for reporting and for writing the
        hold-out files.

    """
    rows = []
    first_split: dict = {}

    for seed in seeds:
        train_idx, holdout_idx = scaffold_split_indices(
            train_df["SMILES"], test_size=holdout_fraction, random_state=seed
        )
        coverage = pd.DataFrame(
            {
                "train": np.isfinite(Y[train_idx]).sum(axis=0),
                "holdout": np.isfinite(Y[holdout_idx]).sum(axis=0),
            },
            index=CYP_ISOFORMS,
        )
        print(f"\n--- scaffold split, seed {seed}: {len(train_idx)} train / {len(holdout_idx)} hold-out ---")
        print(coverage.to_string())
        split_aux = aux_for_holdout(aux, train_df["SMILES"].to_numpy()[holdout_idx])

        for key in model_keys:
            start = time.time()
            _, pred = fit_predict(key, X[train_idx], Y[train_idx], X[holdout_idx], params, split_aux)
            results = score_holdout(train_df, pred, holdout_idx)
            print(f"\nModel {MODEL_LABELS[key]}  ({time.time() - start:.0f}s)")
            print(summarise(results).to_string())

            rows.append(
                {
                    "seed": seed,
                    "model": MODEL_LABELS[key],
                    "MA_ST-RAE": results.loc["MA", "ST-RAE_mean"],
                    "MA_MAE": results.loc["MA", "MAE_mean"],
                    "MA_R2": results.loc["MA", "R2_mean"],
                    "MA_Spearman": results.loc["MA", "Spearman_R_mean"],
                }
            )
            if seed == seeds[0]:
                first_split.setdefault("holdout_idx", holdout_idx)
                first_split.setdefault("results", {})[key] = results
                first_split.setdefault("preds", {})[key] = pred

    return pd.DataFrame(rows), first_split


# --------------------------------------------------------------------------------------
# Plots (optional)
# --------------------------------------------------------------------------------------


def write_plots(train_df: pd.DataFrame, first_split: dict, best_key: str, output_dir: Path) -> None:
    """Write the ST-RAE comparison bar chart and a predicted-vs-observed panel."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_style("whitegrid")
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for key, results in first_split["results"].items():
        frame = results["ST-RAE_mean"].rename("ST-RAE").to_frame()
        frame["Model"] = MODEL_LABELS[key]
        frames.append(frame)
    plot_df = pd.concat(frames).reset_index()
    plot_df["Endpoint"] = plot_df["Endpoint"].str.replace(
        "_pIC50_direct_inhibition", "", regex=False
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=plot_df, x="Endpoint", y="ST-RAE", hue="Model", ax=ax)
    ax.axhline(1.0, color="crimson", linestyle="--", label="mean predictor")
    ax.set_title("Scaffold hold-out: soft-threshold RAE (lower is better)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(plot_dir / "holdout_st_rae.png", dpi=150)
    plt.close(fig)

    holdout = train_df.iloc[first_split["holdout_idx"]].reset_index(drop=True)
    best_pred = first_split["preds"][best_key]
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.6))
    for ax, cyp, col, pred_col in zip(axes, CYP_ISOFORMS, TARGET_COLS, best_pred.T):
        measured = holdout[col].notna().to_numpy()
        y_true = holdout.loc[measured, col].to_numpy()
        y_pred = pred_col[measured]
        lo = holdout.loc[measured, f"{col}_conf_low"].to_numpy()
        hi = holdout.loc[measured, f"{col}_conf_high"].to_numpy()
        ax.errorbar(
            y_true, y_pred, xerr=[y_true - lo, hi - y_true],
            fmt="o", ms=3.5, alpha=0.45, elinewidth=0.6, color="steelblue",
        )
        limits = [min(y_true.min(), y_pred.min()) - 0.3, max(y_true.max(), y_pred.max()) + 0.3]
        ax.plot(limits, limits, "k--", lw=1)
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_xlabel("Observed pIC$_{50}$")
        ax.set_ylabel("Predicted pIC$_{50}$")
        ax.set_title(f"{cyp} (N = {measured.sum()})")
    fig.suptitle(
        f"Scaffold hold-out predictions — {MODEL_LABELS[best_key]} "
        "(x error bars = credible interval)",
        y=1.03,
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(plot_dir / "holdout_predicted_vs_observed.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"plots written to {plot_dir}")


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42, 7, 2024],
        help="Scaffold-split seeds for the hold-out comparison (default: 42 7 2024)",
    )
    parser.add_argument(
        "--holdout-fraction", type=float, default=0.2,
        help="Fraction of compounds held out by each scaffold split (default: 0.2)",
    )
    parser.add_argument(
        "--final-model", choices=["A", "B", "C", "auto"], default="auto",
        help="Model refit on all data for the submission: A = per-task, B = multitask, "
             "C = multitask + substrate aux (needs --aux-substrate), "
             "auto = whichever wins on mean MA ST-RAE (default: auto)",
    )
    aux_group = parser.add_argument_group(
        "auxiliary substrate tasks",
        "Stack CYP substrate-classification heads (Ni et al. 2025) into the multitask "
        "model as model C. Substrate status is a different endpoint from inhibition, so "
        "this only ever helps via a shared representation — which is what it measures.",
    )
    aux_group.add_argument(
        "--aux-substrate", action="store_true",
        help="Add model C to the comparison, using the substrate labels in --aux-dir",
    )
    aux_group.add_argument(
        "--aux-dir", type=Path, default=DEFAULT_AUX_DIR,
        help="Directory holding the Ni et al. per-isoform CSVs "
             "(fetch with `python cyp_substrate_aux.py --download`)",
    )
    aux_group.add_argument(
        "--aux-isoforms", nargs="+", default=CHALLENGE_ISOFORMS, choices=ALL_ISOFORMS,
        help="Isoforms to use as auxiliary tasks. CYP2C19 and CYP2E1 are not scored by the "
             "challenge but still add shared-encoder signal (default: the four scored ones)",
    )
    aux_group.add_argument(
        "--aux-weight", type=float, default=0.3,
        help="LightGBM sample weight on auxiliary rows relative to primary rows (default: 0.3)",
    )
    aux_group.add_argument(
        "--aux-spread", type=float, default=0.5,
        help="Substrate/non-substrate separation in units of the primary label's standard "
             "deviation, after mapping binary labels onto the pIC50 scale (default: 0.5)",
    )
    parser.add_argument(
        "--no-holdout-eval", action="store_true",
        help="Skip the scaffold-split comparison and go straight to the final fit "
             "(requires an explicit --final-model)",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument(
        "--cache-dir", type=Path, default=None,
        help="Where to cache feature arrays (default: <output-dir>/features)",
    )
    parser.add_argument(
        "--submission-name", default="my_multitask_activity_submission.csv",
        help="Filename for the submission, written inside --output-dir",
    )
    parser.add_argument("--plots", action="store_true", help="Write diagnostic PNGs to <output-dir>/plots")
    parser.add_argument(
        "--save-model", action="store_true",
        help="Joblib-dump the final fitted estimator to <output-dir>/models",
    )
    parser.add_argument("--n-estimators", type=int, default=None, help="Override LGBM n_estimators")
    parser.add_argument("--learning-rate", type=float, default=None, help="Override LGBM learning_rate")
    parser.add_argument("--verbose", action="store_true", help="Show the scorer's progress logging")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    """Run the full pipeline: featurize, compare, refit, predict, validate."""
    args = parse_args(argv)
    if args.no_holdout_eval and args.final_model == "auto":
        raise SystemExit("--no-holdout-eval needs an explicit --final-model A, B or C")
    if args.final_model == "C" and not args.aux_substrate:
        raise SystemExit("--final-model C needs --aux-substrate")

    if not args.verbose:
        logger.remove()  # the scorer logs one line per endpoint per call
    warnings.filterwarnings("ignore", message="X does not have valid feature names")

    params = dict(LGBM_PARAMS)
    if args.n_estimators is not None:
        params["n_estimators"] = args.n_estimators
    if args.learning_rate is not None:
        params["learning_rate"] = args.learning_rate

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir or (output_dir / "features")
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("1. Loading data")
    print("=" * 88)
    train_df, test_df = load_data()

    print("\n" + "=" * 88)
    print("2. Featurizing (ECFP4 + RDKit 2D descriptors)")
    print("=" * 88)
    featurizer = build_featurizer()
    X_train_all = featurize(train_df["SMILES"], "train", featurizer, cache_dir)
    X_test = featurize(test_df["SMILES"], "test", featurizer, cache_dir)
    Y_train_all = train_df[TARGET_COLS].to_numpy(dtype=np.float64)  # NaN = not measured
    print(
        f"feature matrix: {X_train_all.shape[1]} columns "
        f"(2048 ECFP4 bits + {X_train_all.shape[1] - 2048} 2D descriptors)"
    )

    aux = None
    model_keys = ("A", "B")
    if args.aux_substrate:
        aux = build_aux_bundle(
            featurizer, cache_dir, args.aux_dir, args.aux_isoforms,
            args.aux_weight, args.aux_spread,
        )
        model_keys = ("A", "B", "C")

    final_key = args.final_model
    first_split: dict = {}

    if not args.no_holdout_eval:
        print("\n" + "=" * 88)
        print("3. Scaffold-split comparison, scored with the challenge metrics")
        print("=" * 88)
        seed_df, first_split = evaluate_over_scaffold_splits(
            train_df, X_train_all, Y_train_all, args.seeds, args.holdout_fraction, params,
            model_keys=model_keys, aux=aux,
        )

        summary = seed_df.groupby("model").agg(["mean", "std"]).round(3)
        summary.columns = ["_".join(c) for c in summary.columns]
        summary = summary[[c for c in summary.columns if not c.startswith("seed")]]
        print("\n" + "-" * 88)
        print(f"Across {len(args.seeds)} scaffold split(s), macro-averaged over the four isoforms:")
        print(summary.to_string())
        print(
            "\nST-RAE is the leaderboard's ranking metric: lower is better, "
            "1.0 == predicting the mean."
        )

        seed_file = output_dir / "scaffold_split_comparison.csv"
        seed_df.to_csv(seed_file, index=False)
        print(f"per-seed results written to {seed_file}")

        best_label = seed_df.groupby("model")["MA_ST-RAE"].mean().idxmin()
        best_key = next(k for k, v in MODEL_LABELS.items() if v == best_label)
        print(f"\nBest mean MA ST-RAE: {best_label}")
        if final_key == "auto":
            final_key = best_key

        # Write the first split's hold-out in submission format, so evaluate_submission.py
        # can re-score it later without retraining anything.
        holdout_idx = first_split["holdout_idx"]
        holdout_predictions = make_prediction_frame(
            first_split["preds"][best_key], train_df["Molecule_Name"].to_numpy()[holdout_idx]
        )
        holdout_predictions.insert(0, "SMILES", train_df["SMILES"].to_numpy()[holdout_idx])
        holdout_predictions.to_csv(output_dir / "holdout_predictions_multitask.csv", index=False)
        make_ground_truth_frame(train_df, holdout_idx).to_csv(
            output_dir / "holdout_ground_truth.csv", index=False
        )
        print(
            f"hold-out predictions + answers ({len(holdout_idx)} compounds) written to "
            f"{output_dir}/holdout_{{predictions_multitask,ground_truth}}.csv"
        )

        if args.plots:
            write_plots(train_df, first_split, best_key, output_dir)

    print("\n" + "=" * 88)
    print(f"4. Refitting model {MODEL_LABELS[final_key]} on all {len(train_df)} compounds")
    print("=" * 88)
    start = time.time()
    # No hold-out to protect here, so the auxiliary set is used in full. The blinded test
    # SMILES are public, and the substrate labels carry no pIC50 information, so an
    # auxiliary compound that coincides with a test compound leaks nothing.
    final_model, test_pred = fit_predict(final_key, X_train_all, Y_train_all, X_test, params, aux)
    print(f"final model trained in {time.time() - start:.0f}s")

    if args.save_model:
        model_dir = output_dir / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_file = model_dir / f"cyp_multitask_lgbm_{final_key}.pkl"
        final_model.save(model_file)  # joblib dump of the fitted estimator
        print(f"model written to {model_file}")

    print("\n" + "=" * 88)
    print("5. Writing and validating the submission")
    print("=" * 88)
    submission = test_df[["SMILES", "Molecule_Name"]].copy()
    for i, col in enumerate(TARGET_COLS):
        submission[col] = test_pred[:, i]

    # Sanity check: tree models shrink toward the training mean, so a narrower predicted
    # spread is expected — a *shifted* centre is the thing to worry about.
    spread = pd.DataFrame(
        {
            "train_mean": train_df[TARGET_COLS].mean().values,
            "pred_mean": submission[TARGET_COLS].mean().values,
            "train_std": train_df[TARGET_COLS].std().values,
            "pred_std": submission[TARGET_COLS].std().values,
        },
        index=CYP_ISOFORMS,
    ).round(2)
    print("Measured training vs. predicted test pIC50 distributions:")
    print(spread.to_string())

    submission_file = output_dir / args.submission_name
    submission.to_csv(submission_file, index=False)
    print(f"\nSubmission rows: {len(submission)} -> {submission_file}")

    is_valid, validation_errors = validate_activity_submission(
        submission_file, expected_ids=set(test_df["Molecule_Name"])
    )
    if is_valid:
        print("✅ Activity submission file is valid.")
        print("\nUpload it at https://huggingface.co/spaces/openadmet/cyp-challenge")
    else:
        print("❌ Activity submission file is invalid:")
        for msg in validation_errors:
            print(f" - {msg}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
