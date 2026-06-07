"""
Evaluate PTR v3 predictions: reverse-transform PTR -> pEC50 and score
against test_phase1.csv ground truth (253 compounds).

PTR v3 parameters: threshold=5.0, std=0.55
Reverse transform: pEC50_pred = norm.ppf(ptr_pred, 5.0, 0.55)

Usage:
    conda activate openadmet-models
    cd /home/spal/OpenAdmet_After_Phase1

    # Without LGBM gate:
    python ~/dockerimages/QSARTuna/PXR/evaluate_ptr_predictions_v3.py \
        --pred_csv  <model_output_dir>/test_prediction.csv \
        --output    ptr_v3_pec50_predictions.csv

    # With LGBM gate:
    python ~/dockerimages/QSARTuna/PXR/evaluate_ptr_predictions_v3.py \
        --pred_csv       <model_output_dir>/test_prediction.csv \
        --classifier_dir classifier_dir \
        --output         ptr_v3_pec50_predictions_gated.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

# ── PTR v3 parameters ─────────────────────────────────────────────────────────
THRESHOLD = 5.0
STD       = 0.55
PTR_CLIP  = 1e-9

TEST_TRUTH = "/home/spal/dockerimages/QSARTuna/PXR/test_phase1.csv"
BASELINE   = {"mae": 0.476, "rmse": 0.675, "r2": 0.579, "spearman": 0.789}

# Gate parameters
GATE_LOW        = 0.30
GATE_HIGH       = 0.70
INACTIVE_MEDIAN = 3.0
BLEND_WEIGHT    = 0.60


def ptr_reverse(ptr_pred):
    return norm.ppf(np.clip(ptr_pred, PTR_CLIP, 1 - PTR_CLIP), THRESHOLD, STD)


def scorecard(truth, pred, label=""):
    truth, pred = np.array(truth, dtype=float), np.array(pred, dtype=float)
    valid = np.isfinite(truth) & np.isfinite(pred)
    truth, pred = truth[valid], pred[valid]

    mae    = np.mean(np.abs(truth - pred))
    rmse   = np.sqrt(np.mean((truth - pred) ** 2))
    r2     = np.corrcoef(truth, pred)[0, 1] ** 2
    rho, _ = spearmanr(truth, pred)
    large  = np.mean(np.abs(truth - pred) >= 0.5)

    print(f"\n{'='*55}")
    print(f"  {label}  (n={len(truth)})")
    print(f"{'='*55}")
    print(f"  MAE        : {mae:.4f}   (baseline {BASELINE['mae']:.3f})")
    print(f"  RMSE       : {rmse:.4f}   (baseline {BASELINE['rmse']:.3f})")
    print(f"  Pearson R² : {r2:.4f}   (baseline {BASELINE['r2']:.3f})")
    print(f"  Spearman ρ : {rho:.4f}   (baseline {BASELINE['spearman']:.3f})")
    print(f"  |err|>=0.5 : {large*100:.1f}%   (baseline 34.8%)")
    print(f"\n  --- By activity class ---")
    bins   = [0, 3.5, 4.5, 5.5, 10]
    labels = ["inactive  (<3.5)", "weak   (3.5-4.5)", "moderate(4.5-5.5)", "active    (>5.5)"]
    cuts   = pd.cut(truth, bins=bins, labels=labels)
    for lbl in labels:
        m = cuts == lbl
        if m.sum() > 0:
            z_mae = np.mean(np.abs(truth[m] - pred[m]))
            n_lg  = (np.abs(truth[m] - pred[m]) >= 0.5).sum()
            print(f"  {lbl}  n={m.sum():3d}  MAE={z_mae:.4f}  |>=0.5: {n_lg} ({n_lg/m.sum()*100:.0f}%)")
    return {"mae": mae, "rmse": rmse, "r2": r2, "spearman": rho}


def apply_lgbm_gate(df, classifier_dir):
    import joblib
    import yaml
    from openadmet.models.features.combine import FeatureConcatenator
    import openadmet.models.registries  # noqa: F401

    cdir  = Path(classifier_dir)
    model = joblib.load(cdir / "model.pkl")
    with open(cdir / "recipe_components" / "procedure.yaml") as f:
        proc = yaml.safe_load(f)

    featurizer = FeatureConcatenator(**proc["feat"]["params"])
    X, _       = featurizer.featurize(df["SMILES"].tolist())
    p_active   = model.predict_proba(X)[:, 1]

    df = df.copy()
    df["p_active"] = p_active

    gated    = df["pEC50_pred"].copy()
    low_mask = p_active < GATE_LOW
    gated[low_mask] = (
        BLEND_WEIGHT * INACTIVE_MEDIAN
        + (1 - BLEND_WEIGHT) * df.loc[low_mask, "pEC50_pred"]
    )
    df["pEC50_pred_gated"] = gated

    print(f"\nLGBM gate: {low_mask.sum()} compounds blended toward {INACTIVE_MEDIAN} "
          f"(P_active < {GATE_LOW}), "
          f"{(p_active > GATE_HIGH).sum()} clear actives unchanged")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_csv",       required=True,
                        help="test_prediction.csv from openadmet predict (PTR v3 space)")
    parser.add_argument("--output",         default="ptr_v3_pec50_predictions.csv")
    parser.add_argument("--classifier_dir", default=None,
                        help="Optional: trained LGBM classifier dir for gating")
    args = parser.parse_args()

    # ── Load PTR predictions ──────────────────────────────────────────────────
    pred = pd.read_csv(args.pred_csv)
    pred_col = next(
        (c for c in pred.columns if "PRED" in c and "pEC50" in c and "counter" not in c),
        None,
    )
    if pred_col is None:
        sys.exit("ERROR: could not find pEC50 prediction column in pred_csv")
    print(f"Prediction column : {pred_col}")
    print(f"Reverse transform : norm.ppf(ptr_pred, threshold={THRESHOLD}, std={STD})")

    # ── Load ground truth ─────────────────────────────────────────────────────
    truth = pd.read_csv(TEST_TRUTH)[["Molecule Name", "SMILES", "pEC50"]]

    # ── Merge & reverse transform ─────────────────────────────────────────────
    df = pred[["Molecule Name", "SMILES", pred_col]].rename(columns={pred_col: "ptr_pred"})
    df = df.merge(truth, on=["Molecule Name", "SMILES"], how="left")
    df["pEC50_pred"] = ptr_reverse(df["ptr_pred"].values)

    # ── Evaluate: no gate ─────────────────────────────────────────────────────
    scorecard(df["pEC50"].values, df["pEC50_pred"].values,
              label="PTR v3 + ChemProp (no gate)")

    # ── Evaluate: with LGBM gate ──────────────────────────────────────────────
    if args.classifier_dir:
        df = apply_lgbm_gate(df, args.classifier_dir)
        scorecard(df["pEC50"].values, df["pEC50_pred_gated"].values,
                  label="PTR v3 + ChemProp + LGBM gate")
        df["pEC50_final"] = df["pEC50_pred_gated"]
    else:
        df["pEC50_final"] = df["pEC50_pred"]

    # ── Save ──────────────────────────────────────────────────────────────────
    out_cols = ["Molecule Name", "SMILES", "ptr_pred", "pEC50_pred", "pEC50_final", "pEC50"]
    if "p_active" in df.columns:
        out_cols.insert(4, "p_active")
    if "pEC50_pred_gated" in df.columns:
        out_cols.insert(5, "pEC50_pred_gated")
    df[out_cols].to_csv(args.output, index=False)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
