"""First-pass CACHE PGK2 predictive model and submission generator.

This script implements the first executable version of the immediate action
plan:
  - ECFP4/ECFP6 featurization
  - LightGBM broad/strict/orthosteric classifiers plus z-score regressor
  - scaffold-aware holdout diagnostics
  - validation/test scoring
  - diversity-selected top-50 submission files

Run from the repository root:
  python pipeline/first_model_pipeline.py --config pipeline/first_model_config.yaml
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMClassifier, LGBMRegressor
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy import sparse
from sklearn.metrics import (
    average_precision_score,
    mean_squared_error,
    roc_auc_score,
)


@dataclass(frozen=True)
class FingerprintSpec:
    radii: tuple[int, ...]
    fp_size: int


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def mol_from_smiles(smiles: str) -> Chem.Mol | None:
    if not isinstance(smiles, str) or not smiles:
        return None
    return Chem.MolFromSmiles(smiles)


def murcko_scaffold(smiles: str) -> str:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return ""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception:
        return ""


def add_scaffolds(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "murcko_scaffold" not in out.columns:
        out["murcko_scaffold"] = out["SMILES"].map(murcko_scaffold)
    return out


def maybe_debug_sample(
    df: pd.DataFrame, max_rows: int | None, seed: int, label_column: str | None = None
) -> pd.DataFrame:
    if max_rows is None or max_rows <= 0 or len(df) <= max_rows:
        return df
    if label_column and label_column in df.columns:
        parts = []
        counts = df[label_column].value_counts()
        for value, count in counts.items():
            n_value = max(1, round(max_rows * count / len(df)))
            part = df[df[label_column] == value].sample(
                n=min(n_value, count), random_state=seed
            )
            parts.append(part)
        sampled = pd.concat(parts, ignore_index=True)
        if len(sampled) > max_rows:
            sampled = sampled.sample(n=max_rows, random_state=seed)
        return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df.sample(n=max_rows, random_state=seed).reset_index(drop=True)


def make_scaffold_split(
    df: pd.DataFrame, holdout_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    scaffolds = df["murcko_scaffold"].fillna("").to_numpy()
    unique = np.array(sorted(set(scaffolds)))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)

    target_holdout = int(round(len(df) * holdout_fraction))
    holdout_scaffolds: set[str] = set()
    holdout_count = 0
    scaffold_counts = Counter(scaffolds)
    for scaffold in unique:
        if holdout_count >= target_holdout:
            break
        holdout_scaffolds.add(scaffold)
        holdout_count += scaffold_counts[scaffold]

    holdout_mask = np.array([s in holdout_scaffolds for s in scaffolds])
    train_mask = ~holdout_mask
    return train_mask, holdout_mask


def fingerprint_matrix(smiles: pd.Series, spec: FingerprintSpec) -> sparse.csr_matrix:
    """Build concatenated Morgan bit fingerprints as CSR matrix."""
    blocks: list[sparse.csr_matrix] = []
    for radius in spec.radii:
        generator = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=spec.fp_size
        )
        rows: list[int] = []
        cols: list[int] = []
        data: list[np.float32] = []
        for row_idx, smi in enumerate(smiles):
            mol = mol_from_smiles(smi)
            if mol is None:
                continue
            fp = generator.GetFingerprint(mol)
            on_bits = list(fp.GetOnBits())
            rows.extend([row_idx] * len(on_bits))
            cols.extend(on_bits)
            data.extend([1.0] * len(on_bits))
        block = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(len(smiles), spec.fp_size),
            dtype=np.float32,
        )
        blocks.append(block)
    return sparse.hstack(blocks, format="csr", dtype=np.float32)


def build_positive_mask(df: pd.DataFrame, positive_cfg: dict[str, Any]) -> pd.Series:
    if "label_column" in positive_cfg:
        return df[positive_cfg["label_column"]] == positive_cfg.get("label_value", 1)

    mask = pd.Series(True, index=df.index)
    if "min_count_pgk2" in positive_cfg:
        mask &= df["count_PGK2"] >= positive_cfg["min_count_pgk2"]
    if "min_zscore_pgk2" in positive_cfg:
        mask &= df["zscore_PGK2"] >= positive_cfg["min_zscore_pgk2"]
    if "max_count_ntc" in positive_cfg:
        mask &= df["count_NTC"] <= positive_cfg["max_count_ntc"]
    if "max_historic_hits" in positive_cfg:
        mask &= df["historic_hits"] < positive_cfg["max_historic_hits"]
    if "max_count_pgk2_with_inhibitor" in positive_cfg:
        mask &= (
            df["count_PGK2_with_inhibitor"]
            <= positive_cfg["max_count_pgk2_with_inhibitor"]
        )
    return mask


def make_target_frame(
    train_df: pd.DataFrame, target_name: str, target_cfg: dict[str, Any]
) -> tuple[pd.DataFrame, np.ndarray]:
    include = target_cfg.get("include", "broad_all")

    if target_cfg["kind"] == "regressor":
        y = train_df[target_cfg["target_column"]].to_numpy(dtype=np.float32)
        return train_df.copy(), y

    positive_mask = build_positive_mask(train_df, target_cfg["positive"])
    broad_negative_mask = train_df["LABEL"] == 0

    if include == "broad_all":
        use_mask = pd.Series(True, index=train_df.index)
    elif include in {"strict_pos_and_broad_neg", "orthosteric_pos_and_broad_neg"}:
        use_mask = positive_mask | broad_negative_mask
    else:
        raise ValueError(f"Unsupported include mode for {target_name}: {include}")

    out = train_df.loc[use_mask].copy()
    y = positive_mask.loc[use_mask].astype(np.int8).to_numpy()
    return out, y


def classifier_from_config(cfg: dict[str, Any], seed: int, n_jobs: int) -> LGBMClassifier:
    params = dict(cfg)
    return LGBMClassifier(
        objective="binary",
        class_weight="balanced",
        random_state=seed,
        n_jobs=n_jobs,
        verbosity=-1,
        **params,
    )


def regressor_from_config(cfg: dict[str, Any], seed: int, n_jobs: int) -> LGBMRegressor:
    params = dict(cfg)
    return LGBMRegressor(
        objective="regression",
        random_state=seed,
        n_jobs=n_jobs,
        verbosity=-1,
        **params,
    )


def top50_summary(df: pd.DataFrame, score: np.ndarray, label: np.ndarray) -> dict[str, Any]:
    order = np.argsort(-score)
    top = df.iloc[order[:50]].copy()
    top_labels = label[order[:50]]
    return {
        "top50_hits": int(top_labels.sum()),
        "top50_distinct_scaffolds": int(top["murcko_scaffold"].nunique()),
        "top50_scaffolds": top["murcko_scaffold"].head(10).tolist(),
    }


def evaluate_model(
    target_name: str,
    kind: str,
    model: Any,
    x_holdout: sparse.csr_matrix,
    y_holdout: np.ndarray,
    holdout_df: pd.DataFrame,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {"target": target_name, "kind": kind}
    if kind == "classifier":
        score = model.predict_proba(x_holdout)[:, 1]
        if len(np.unique(y_holdout)) > 1:
            metrics["roc_auc"] = float(roc_auc_score(y_holdout, score))
            metrics["pr_auc"] = float(average_precision_score(y_holdout, score))
        metrics.update(top50_summary(holdout_df, score, y_holdout))
    else:
        pred = model.predict(x_holdout)
        metrics["rmse"] = float(math.sqrt(mean_squared_error(y_holdout, pred)))
        broad_label = holdout_df["LABEL"].to_numpy(dtype=np.int8)
        if len(np.unique(broad_label)) > 1:
            metrics["broad_roc_auc"] = float(roc_auc_score(broad_label, pred))
            metrics["broad_pr_auc"] = float(average_precision_score(broad_label, pred))
        metrics.update(top50_summary(holdout_df, pred, broad_label))
    return metrics


def percentile_rank(score: np.ndarray) -> np.ndarray:
    order = np.argsort(score)
    ranks = np.empty(len(score), dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, len(score), dtype=np.float32)
    return ranks


def predict_score(kind: str, model: Any, x: sparse.csr_matrix) -> np.ndarray:
    if kind == "classifier":
        return model.predict_proba(x)[:, 1]
    return model.predict(x)


def bitvect_for_smiles(smiles: str, radius: int = 2, fp_size: int = 2048):
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)
    return generator.GetFingerprint(mol)


def select_diverse_top50(
    scored_df: pd.DataFrame,
    pick_count: int,
    top_pool_size: int,
    max_per_scaffold: int,
    min_tanimoto_distance: float,
) -> pd.DataFrame:
    pool = scored_df.sort_values("ensemble_score", ascending=False).head(top_pool_size)
    max_similarity = 1.0 - min_tanimoto_distance

    selected_indices: list[int] = []
    selected_fps: list[Any] = []
    scaffold_counts: defaultdict[str, int] = defaultdict(int)

    def try_add(row_idx: int, row: pd.Series, use_similarity: bool, use_scaffold: bool):
        scaffold = row.get("murcko_scaffold", "")
        if use_scaffold and scaffold_counts[scaffold] >= max_per_scaffold:
            return False
        fp = bitvect_for_smiles(row["SMILES"])
        if fp is None:
            return False
        if use_similarity and selected_fps:
            sims = DataStructs.BulkTanimotoSimilarity(fp, selected_fps)
            if max(sims) > max_similarity:
                return False
        selected_indices.append(row_idx)
        selected_fps.append(fp)
        scaffold_counts[scaffold] += 1
        return True

    passes = [
        (True, True),
        (False, True),
        (False, False),
    ]
    for use_similarity, use_scaffold in passes:
        if len(selected_indices) >= pick_count:
            break
        for row_idx, row in pool.iterrows():
            if row_idx in selected_indices:
                continue
            try_add(row_idx, row, use_similarity, use_scaffold)
            if len(selected_indices) >= pick_count:
                break

    selected = scored_df.loc[selected_indices].copy()
    selected["selection_rank"] = np.arange(1, len(selected) + 1)
    return selected


def write_validation_submission(
    selected: pd.DataFrame, output_dir: Path, team_name: str, submission_id: str
) -> Path:
    path = output_dir / f"{team_name}_{submission_id}_validation.txt"
    selected["CatalogID"].head(50).to_csv(path, index=False, header=False)
    return path


def write_test_submission(
    scored: pd.DataFrame,
    selected: pd.DataFrame,
    output_dir: Path,
    team_name: str,
    submission_id: str,
) -> Path:
    path = output_dir / f"{team_name}_{submission_id}_test.csv"
    selected_ids = set(selected["CatalogID"].head(50))
    out = scored[["CatalogID", "ensemble_score"]].copy()
    out["Sel_50"] = out["CatalogID"].isin(selected_ids).astype(int)
    out = out.rename(columns={"ensemble_score": "Score"})
    out = out[["CatalogID", "Sel_50", "Score"]]
    out.to_csv(path, index=False)
    return path


def save_top_table(selected: pd.DataFrame, output_dir: Path, name: str) -> Path:
    path = output_dir / f"{name}_selected_top50.csv"
    cols = [
        "selection_rank",
        "CatalogID",
        "SMILES",
        "ensemble_score",
        "murcko_scaffold",
    ]
    score_cols = [c for c in selected.columns if c.startswith("score_")]
    selected[cols + score_cols].to_csv(path, index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("pipeline/first_model_config.yaml"),
        help="YAML config path.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(cfg["paths"]["output_dir"])
    ensure_output_dir(output_dir)

    seed = int(cfg["run"]["seed"])
    n_jobs = int(cfg["run"]["n_jobs"])
    spec = FingerprintSpec(
        radii=tuple(int(r) for r in cfg["features"]["radii"]),
        fp_size=int(cfg["features"]["fp_size"]),
    )

    print("Loading data")
    train = pd.read_parquet(cfg["paths"]["train_labeled"])
    validation = pd.read_csv(cfg["paths"]["validation_csv"])
    test = pd.read_csv(cfg["paths"]["test_csv"])

    train = maybe_debug_sample(
        train, cfg["run"].get("debug_max_train_rows"), seed, label_column="LABEL"
    )
    validation = maybe_debug_sample(
        validation, cfg["run"].get("debug_max_validation_rows"), seed
    )
    test = maybe_debug_sample(test, cfg["run"].get("debug_max_test_rows"), seed)

    train = add_scaffolds(train)
    validation = add_scaffolds(validation)
    test = add_scaffolds(test)

    print(f"Training rows: {len(train):,}")
    print(f"Validation rows: {len(validation):,}")
    print(f"Test rows: {len(test):,}")

    print("Creating scaffold holdout split")
    base_train_mask, base_holdout_mask = make_scaffold_split(
        train,
        float(cfg["split"]["scaffold_holdout_fraction"]),
        seed,
    )

    models: dict[str, dict[str, Any]] = {}
    metrics: list[dict[str, Any]] = []

    for target_name, target_cfg in cfg["targets"].items():
        kind = target_cfg["kind"]
        print(f"\nTraining target: {target_name} ({kind})")
        target_df, y = make_target_frame(train, target_name, target_cfg)
        target_indices = target_df.index.to_numpy()
        train_mask = np.isin(target_indices, train.index[base_train_mask])
        holdout_mask = np.isin(target_indices, train.index[base_holdout_mask])

        x = fingerprint_matrix(target_df["SMILES"], spec)
        x_train = x[train_mask]
        x_holdout = x[holdout_mask]
        y_train = y[train_mask]
        y_holdout = y[holdout_mask]
        holdout_df = target_df.iloc[np.flatnonzero(holdout_mask)].copy()

        if kind == "classifier":
            print(
                f"Rows: {len(target_df):,}; positives: {int(y.sum()):,}; "
                f"train positives: {int(y_train.sum()):,}; "
                f"holdout positives: {int(y_holdout.sum()):,}"
            )
            model = classifier_from_config(
                cfg["models"]["classifier"],
                seed=seed,
                n_jobs=n_jobs,
            )
        else:
            print(f"Rows: {len(target_df):,}")
            model = regressor_from_config(
                cfg["models"]["regressor"],
                seed=seed,
                n_jobs=n_jobs,
            )

        model.fit(x_train, y_train)
        model_metrics = evaluate_model(
            target_name, kind, model, x_holdout, y_holdout, holdout_df
        )
        metrics.append(model_metrics)
        print(json.dumps(model_metrics, indent=2))

        model_path = output_dir / f"model_{target_name}.joblib"
        joblib.dump(model, model_path)
        models[target_name] = {
            "kind": kind,
            "model": model,
            "path": str(model_path),
        }

    metrics_path = output_dir / "holdout_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nSaved holdout metrics: {metrics_path}")

    def score_dataset(df: pd.DataFrame, name: str) -> pd.DataFrame:
        print(f"\nScoring {name}")
        x = fingerprint_matrix(df["SMILES"], spec)
        scored = df.copy()
        weighted = np.zeros(len(scored), dtype=np.float32)
        weight_total = 0.0
        for target_name, model_info in models.items():
            raw_score = predict_score(model_info["kind"], model_info["model"], x)
            scored[f"score_{target_name}"] = raw_score
            rank_score = percentile_rank(raw_score)
            weight = float(cfg["ensemble"].get(target_name, 0.0))
            weighted += weight * rank_score
            weight_total += weight
        if weight_total <= 0:
            raise ValueError("Ensemble weights sum to zero.")
        scored["ensemble_score"] = weighted / weight_total

        scored_path = output_dir / f"{name}_scores.parquet"
        scored.to_parquet(scored_path, index=False)
        print(f"Saved scores: {scored_path}")
        return scored

    validation_scored = score_dataset(validation, "validation")
    test_scored = score_dataset(test, "test")

    print("\nSelecting diverse top 50")
    sel_cfg = cfg["selection"]
    validation_selected = select_diverse_top50(
        validation_scored,
        pick_count=int(cfg["run"]["final_pick_count"]),
        top_pool_size=int(cfg["run"]["top_pool_size"]),
        max_per_scaffold=int(sel_cfg["max_per_murcko_scaffold"]),
        min_tanimoto_distance=float(sel_cfg["min_tanimoto_distance"]),
    )
    test_selected = select_diverse_top50(
        test_scored,
        pick_count=int(cfg["run"]["final_pick_count"]),
        top_pool_size=int(cfg["run"]["top_pool_size"]),
        max_per_scaffold=int(sel_cfg["max_per_murcko_scaffold"]),
        min_tanimoto_distance=float(sel_cfg["min_tanimoto_distance"]),
    )

    save_top_table(validation_selected, output_dir, "validation")
    save_top_table(test_selected, output_dir, "test")

    team_name = cfg["run"]["team_name"]
    submission_id = cfg["run"]["submission_id"]
    validation_submission = write_validation_submission(
        validation_selected, output_dir, team_name, submission_id
    )
    test_submission = write_test_submission(
        test_scored, test_selected, output_dir, team_name, submission_id
    )

    run_summary = {
        "config": str(args.config),
        "models": {k: {"kind": v["kind"], "path": v["path"]} for k, v in models.items()},
        "metrics_path": str(metrics_path),
        "validation_submission": str(validation_submission),
        "test_submission": str(test_submission),
        "validation_selected_count": int(len(validation_selected)),
        "test_selected_count": int(len(test_selected)),
    }
    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    print("\nRun complete")
    print(json.dumps(run_summary, indent=2))


if __name__ == "__main__":
    main()
