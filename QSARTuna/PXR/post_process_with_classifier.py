"""
Post-processing: apply the LGBM activity gate on top of ChemProp multitask predictions.

Usage:
    conda activate openadmet-models
    cd /home/spal/dockerimages/QSARTuna/PXR
    python post_process_with_classifier.py \
        --chemprop_pred   <path/to/chemprop_multitask_test_predictions.csv> \
        --classifier_dir  <path/to/pxr_lgbm_classifier_output_dir> \
        --test_smiles     test.csv \
        --output          final_predictions_phase2.csv

The ChemProp output CSV must contain columns: Molecule Name, SMILES, pEC50 (task 0 prediction).
The classifier output directory is the anvil output folder produced by openadmet train.

Gating logic
------------
P_active = classifier P(active=1) for each compound

P_active < GATE_LOW  (default 0.30):
    blend = BLEND_WEIGHT * INACTIVE_MEDIAN + (1 - BLEND_WEIGHT) * chemprop_pred
    Rationale: model says inactive, we pull the prediction strongly toward the
    known inactive median (3.0 log units) to correct systematic overprediction.

P_active > GATE_HIGH (default 0.70):
    final = chemprop_pred  (trust the regressor for clear actives)

GATE_LOW <= P_active <= GATE_HIGH:
    final = chemprop_pred  (uncertain zone — don't override)

Scorecard
---------
If ground truth is available (via --truth_csv), the script also prints the
per-class MAE scorecard so you can compare against the phase-1 baseline:
    inactive (<3.5) MAE baseline: 1.114
    overall MAE baseline:         0.476
    Spearman rho baseline:        0.789
"""

import argparse
import joblib
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from scipy.stats import spearmanr

# ── Gating thresholds ─────────────────────────────────────────────────────────
GATE_LOW = 0.30       # below this P(active): blend toward inactive median
GATE_HIGH = 0.70      # above this P(active): trust regressor directly
INACTIVE_MEDIAN = 3.0 # pEC50 centre for the inactive blend
BLEND_WEIGHT = 0.60   # fraction of INACTIVE_MEDIAN in the blended prediction
# final = 0.60 * 3.0 + 0.40 * chemprop_pred
# ─────────────────────────────────────────────────────────────────────────────


def load_classifier_probas(classifier_dir: str, smiles: list) -> pd.DataFrame:
    """
    Run the trained LGBM classifier on a list of SMILES and return P(active=1).

    The anvil run saves the trained model as model.pkl and the featurizer config
    in recipe_components/procedure.yaml.  No test_prediction.csv is written for
    the blinded test compounds, so we rebuild the featurizer and call predict_proba
    directly.
    """
    cdir = Path(classifier_dir)
    model_path = cdir / "model.pkl"
    proc_path  = cdir / "recipe_components" / "procedure.yaml"

    if not model_path.exists():
        raise FileNotFoundError(f"model.pkl not found in {cdir}")
    if not proc_path.exists():
        raise FileNotFoundError(f"procedure.yaml not found in {cdir / 'recipe_components'}")

    # ── Load trained model ────────────────────────────────────────────────────
    with open(model_path, "rb") as f:
        model = joblib.load(f)

    # ── Rebuild featurizer from saved procedure config ────────────────────────
    import openadmet.models.registries  # noqa: F401 — populates the featurizer registry
    from openadmet.models.features.combine import FeatureConcatenator
    with open(proc_path) as f:
        proc = yaml.safe_load(f)
    feat_params = proc["feat"]["params"]          # featurizers: {DescriptorFeaturizer: ..., ...}
    featurizer = FeatureConcatenator(**feat_params)

    # ── Featurize test SMILES ─────────────────────────────────────────────────
    X, _ = featurizer.featurize(smiles)  # returns (array, common_indices)

    # ── predict_proba → P(active=1) is column index 1 ────────────────────────
    probas = model.predict_proba(X)
    p_active = probas[:, 1]

    return pd.Series(p_active, name="p_active")


def scorecard(df: pd.DataFrame, pred_col: str, truth_col: str = "pEC50", label: str = "Model") -> None:
    """Print the per-class MAE scorecard used to compare against phase-1 baseline."""
    sub = df.dropna(subset=[truth_col, pred_col])
    if len(sub) == 0:
        print(f"[{label}] No truth values available — skipping scorecard.")
        return
    res = sub[truth_col] - sub[pred_col]
    abs_res = res.abs()
    rho, _ = spearmanr(sub[truth_col], sub[pred_col])
    r2 = np.corrcoef(sub[truth_col], sub[pred_col])[0, 1] ** 2

    bins = [0, 3.5, 4.5, 5.5, 10]
    labels = ["inactive(<3.5)", "weak(3.5-4.5)", "moderate(4.5-5.5)", "active(>5.5)"]
    activity_class = pd.cut(sub[truth_col], bins=bins, labels=labels)

    print(f"\n{'='*55}")
    print(f"  {label}  (n={len(sub)})")
    print(f"{'='*55}")
    print(f"  Overall MAE:    {abs_res.mean():.3f}   (baseline: 0.476)")
    print(f"  Overall RMSE:   {np.sqrt((res**2).mean()):.3f}   (baseline: 0.675)")
    print(f"  Pearson R²:     {r2:.3f}   (baseline: 0.579)")
    print(f"  Spearman rho:   {rho:.3f}   (baseline: 0.789)")
    print(f"  |resid|>=0.5:   {(abs_res>=0.5).sum()} ({(abs_res>=0.5).mean()*100:.1f}%)   (baseline: 88, 34.8%)")
    print(f"\n  --- By activity class ---")
    for lbl in labels:
        mask = activity_class == lbl
        if mask.sum() > 0:
            cls_mae = abs_res[mask].mean()
            n_high = (abs_res[mask] >= 0.5).sum()
            print(f"  {lbl:<22} n={mask.sum():>3}  MAE={cls_mae:.3f}  |>=0.5:{n_high:>3} ({n_high/mask.sum()*100:.0f}%)")
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chemprop_pred", required=True,
                        help="CSV with ChemProp multitask predictions (must have Molecule Name, SMILES, pEC50 columns)")
    parser.add_argument("--classifier_dir", required=True,
                        help="OpenAdmet output directory from pxr_lgbm_classifier training run")
    parser.add_argument("--test_smiles", default="test.csv",
                        help="CSV with all 513 test Molecule Names and SMILES")
    parser.add_argument("--output", default="final_predictions_phase2.csv",
                        help="Output CSV path for final gated predictions")
    parser.add_argument("--truth_csv", default=None,
                        help="Optional: CSV with true pEC50 values for the scorecard (e.g. test_phase1.csv)")
    parser.add_argument("--gate_low", type=float, default=GATE_LOW)
    parser.add_argument("--gate_high", type=float, default=GATE_HIGH)
    parser.add_argument("--blend_weight", type=float, default=BLEND_WEIGHT)
    parser.add_argument("--inactive_median", type=float, default=INACTIVE_MEDIAN)
    args = parser.parse_args()

    # ── Load ChemProp predictions ─────────────────────────────────────────────
    df_cp = pd.read_csv(args.chemprop_pred)
    # openadmet predict copies all input columns (including the true pEC50) into
    # its output CSV alongside the OADMET_PRED_* columns. Always prefer the
    # OADMET_PRED column so we don't accidentally rename the truth column.
    pred_cols = [c for c in df_cp.columns
                 if "PRED" in c and "pEC50" in c and "counter" not in c]
    if not pred_cols:
        # Fallback for hand-crafted CSVs that don't follow the OADMET_PRED naming
        pred_cols = [c for c in df_cp.columns
                     if "pEC50" in c and "counter" not in c
                     and "actual" not in c.lower() and "std" not in c.lower()
                     and c != "pEC50"]
    if pred_cols:
        print(f"Prediction column: {pred_cols[0]}")
        df_cp = df_cp.rename({pred_cols[0]: "pEC50_chemprop"}, axis=1)
        # Drop any residual pEC50 column openadmet copied from the input file;
        # the truth will be added cleanly from --truth_csv.
        if "pEC50" in df_cp.columns:
            df_cp = df_cp.drop(columns=["pEC50"])
    elif "pEC50" in df_cp.columns:
        df_cp = df_cp.rename({"pEC50": "pEC50_chemprop"}, axis=1)
    print(f"Loaded ChemProp predictions: {len(df_cp)} compounds")

    # ── Load test SMILES (needed to featurize for the classifier) ─────────────
    df_test = pd.read_csv(args.test_smiles)
    smiles_col = "SMILES" if "SMILES" in df_test.columns else df_test.columns[1]
    # Align to the same order as df_cp using Molecule Name when available
    if "Molecule Name" in df_cp.columns and "Molecule Name" in df_test.columns:
        df_test = df_cp[["Molecule Name"]].merge(df_test[["Molecule Name", smiles_col]], on="Molecule Name", how="left")
    smiles_list = df_test[smiles_col].tolist()

    # ── Run classifier on test SMILES ─────────────────────────────────────────
    p_active_series = load_classifier_probas(args.classifier_dir, smiles_list)
    print(f"Classifier predicted P(active) for {len(p_active_series)} compounds")
    print(f"  P(active) distribution:  mean={p_active_series.mean():.3f}, "
          f"<{args.gate_low}: {(p_active_series<args.gate_low).sum()}, "
          f">{args.gate_high}: {(p_active_series>args.gate_high).sum()}")

    # ── Attach probabilities to the ChemProp predictions ─────────────────────
    df = df_cp.copy()
    df["p_active"] = p_active_series.values

    # ── Apply gating ──────────────────────────────────────────────────────────
    df["pEC50_final"] = df["pEC50_chemprop"].copy()

    low_mask = df["p_active"] < args.gate_low
    df.loc[low_mask, "pEC50_final"] = (
        args.blend_weight * args.inactive_median
        + (1 - args.blend_weight) * df.loc[low_mask, "pEC50_chemprop"]
    )

    print(f"\nGating applied:")
    print(f"  Predicted inactive (P<{args.gate_low}):  {low_mask.sum()} compounds  "
          f"→ blended toward {args.inactive_median}")
    print(f"  Uncertain zone:                          {(~low_mask & (df['p_active']<=args.gate_high)).sum()} compounds  "
          f"→ ChemProp prediction unchanged")
    print(f"  Clear actives (P>{args.gate_high}):     {(df['p_active']>args.gate_high).sum()} compounds  "
          f"→ ChemProp prediction unchanged")

    # ── Scorecard (if truth is available) ─────────────────────────────────────
    if args.truth_csv:
        df_truth = pd.read_csv(args.truth_csv)[["Molecule Name", "pEC50"]]
        df = df.merge(df_truth, on="Molecule Name", how="left")
        scorecard(df, "pEC50_chemprop", label="ChemProp only (before gate)")
        scorecard(df, "pEC50_final",    label="ChemProp + classifier gate")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_cols = ["Molecule Name", "SMILES", "pEC50_chemprop", "p_active", "pEC50_final"]
    out_cols = [c for c in out_cols if c in df.columns]
    df[out_cols].to_csv(args.output, index=False)
    print(f"\nSaved: {args.output}  ({len(df)} rows)")
    print(f"  Final pEC50 range: {df['pEC50_final'].min():.2f} – {df['pEC50_final'].max():.2f}")
    print(f"  Final pEC50 mean:  {df['pEC50_final'].mean():.2f}")


if __name__ == "__main__":
    main()
