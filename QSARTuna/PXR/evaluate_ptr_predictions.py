"""
Reverse-transform PTR predictions back to pEC50 and evaluate against
test_phase1.csv ground truth (253 compounds).

The reverse transform uses a fixed std = max(median_train_std_error, STD_FLOOR)
for all 253 test compounds, matching the effective std used for the majority
of training compounds during the forward transform.

Usage:
    conda activate openadmet-models
    cd /home/spal/dockerimages/QSARTuna/PXR

    python evaluate_ptr_predictions.py \
        --pred_csv  <model_output_dir>/test_prediction.csv \
        --output    ptr_pec50_predictions.csv

Optional LGBM gate:
    python evaluate_ptr_predictions.py \
        --pred_csv       <model_output_dir>/test_prediction.csv \
        --classifier_dir <lgbm_output_dir> \
        --output         ptr_pec50_predictions_gated.csv
"""

import argparse
import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

# ── PTR parameters (must match make_ptr_training_data.py) ─────────────────────
THRESHOLD          = 5.0
STD_FLOOR          = 0.40
PTR_CLIP           = 1e-9
# Median raw std_error from training = 0.150; floor dominates → fixed reverse std = 0.40
REVERSE_STD        = max(0.150, STD_FLOOR)   # = 0.40

# ── Paths ─────────────────────────────────────────────────────────────────────
TEST_TRUTH = "/home/spal/dockerimages/QSARTuna/PXR/test_phase1.csv"

# Baseline scores from phase 1 (for comparison)
BASELINE = {"mae": 0.476, "rmse": 0.661, "spearman": 0.789}


def ptr_reverse(ptr_pred, std=REVERSE_STD):
    clipped = np.clip(ptr_pred, PTR_CLIP, 1 - PTR_CLIP)
    return norm.ppf(clipped, loc=THRESHOLD, scale=std)


def scorecard(truth, pred, label=""):
    mae      = np.mean(np.abs(truth - pred))
    rmse     = np.sqrt(np.mean((truth - pred) ** 2))
    rho, _   = spearmanr(truth, pred)
    print(f"\n{'─'*50}")
    print(f"  {label}")
    print(f"{'─'*50}")
    print(f"  MAE      : {mae:.4f}  (baseline {BASELINE['mae']:.3f})")
    print(f"  RMSE     : {rmse:.4f}  (baseline {BASELINE['rmse']:.3f})")
    print(f"  Spearman : {rho:.4f}  (baseline {BASELINE['spearman']:.3f})")
    print(f"{'─'*50}")

    # Per-zone breakdown
    inactive = truth < 3.5
    boundary = (truth >= 3.5) & (truth < 5.5)
    active   = truth >= 5.5
    for mask, zone in [(inactive, "inactive <3.5"),
                       (boundary, "boundary 3.5-5.5"),
                       (active,   "active ≥5.5")]:
        if mask.sum() > 0:
            z_mae = np.mean(np.abs(truth[mask] - pred[mask]))
            print(f"  {zone:20s}  n={mask.sum():3d}  MAE={z_mae:.4f}")
    return {"mae": mae, "rmse": rmse, "spearman": rho}


def apply_lgbm_gate(df_pred, classifier_dir, smiles_col="SMILES"):
    """Blend predictions for compounds the LGBM gate calls inactive."""
    import joblib, yaml
    from pathlib import Path

    gate_low, gate_high  = 0.30, 0.70
    inactive_median      = 3.0
    blend_weight         = 0.60

    cdir = Path(classifier_dir)
    with open(cdir / "recipe_components" / "procedure.yaml") as f:
        proc = yaml.safe_load(f)
    model = joblib.load(cdir / "model.pkl")

    feat_cols = proc.get("feat", {}).get("params", {}).get("feature_cols", None)
    if feat_cols is None:
        from openadmet.features import get_featurizer
        feat = get_featurizer(proc["feat"]["type"], **proc["feat"].get("params", {}))
        X = feat.transform(df_pred[smiles_col].tolist())
    else:
        X = df_pred[feat_cols].values

    proba = model.predict_proba(X)[:, 1]
    df_pred = df_pred.copy()
    df_pred["p_active"] = proba

    gated = df_pred["pEC50_pred"].copy()
    low_mask = proba < gate_low
    gated[low_mask] = (
        blend_weight * inactive_median
        + (1 - blend_weight) * df_pred["pEC50_pred"][low_mask]
    )
    df_pred["pEC50_pred_gated"] = gated
    print(f"\nLGBM gate: {low_mask.sum()} compounds blended toward inactive median "
          f"(P_active < {gate_low})")
    return df_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_csv",       required=True,
                        help="test_prediction.csv from openadmet predict")
    parser.add_argument("--output",         default="ptr_pec50_predictions.csv",
                        help="Output CSV with reverse-transformed pEC50 predictions")
    parser.add_argument("--classifier_dir", default=None,
                        help="Optional: path to trained LGBM classifier dir for gating")
    args = parser.parse_args()

    # ── Load predictions (PTR space) ──────────────────────────────────────────
    pred = pd.read_csv(args.pred_csv)
    # Find the pEC50 prediction column (task 0, not counter screen)
    pred_col = next(c for c in pred.columns if "PRED" in c and "pEC50" in c
                    and "counter" not in c)
    print(f"Using prediction column: {pred_col}")
    print(f"Reverse transform: norm.ppf(ptr_pred, threshold={THRESHOLD}, std={REVERSE_STD})")

    # ── Load ground truth ─────────────────────────────────────────────────────
    truth_df = pd.read_csv(TEST_TRUTH)

    # ── Merge predictions with ground truth ───────────────────────────────────
    df = pred[["Molecule Name", "SMILES", pred_col]].copy()
    df = df.rename(columns={pred_col: "ptr_pred"})
    df = df.merge(truth_df[["SMILES", "pEC50"]], on="SMILES", how="left")

    # ── Reverse PTR transform → pEC50 using fixed training median std ─────────
    df["pEC50_pred"] = ptr_reverse(df["ptr_pred"].values)

    # ── Evaluate: PTR + ChemProp only ────────────────────────────────────────
    valid = df["pEC50"].notna() & df["pEC50_pred"].notna()
    scorecard(df.loc[valid, "pEC50"].values,
              df.loc[valid, "pEC50_pred"].values,
              label="PTR + ChemProp (no gate)")

    # ── Optional LGBM gate ────────────────────────────────────────────────────
    if args.classifier_dir:
        df = apply_lgbm_gate(df, args.classifier_dir)
        scorecard(df.loc[valid, "pEC50"].values,
                  df.loc[valid, "pEC50_pred_gated"].values,
                  label="PTR + ChemProp + LGBM gate")
        df["pEC50_final"] = df["pEC50_pred_gated"]
    else:
        df["pEC50_final"] = df["pEC50_pred"]

    # ── Save output ───────────────────────────────────────────────────────────
    out_cols = ["Molecule Name", "SMILES", "ptr_pred",
                "pEC50_pred", "pEC50_final", "pEC50"]
    if "pEC50_pred_gated" in df.columns:
        out_cols.insert(5, "pEC50_pred_gated")
    if "p_active" in df.columns:
        out_cols.insert(5, "p_active")
    df[out_cols].to_csv(args.output, index=False)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
