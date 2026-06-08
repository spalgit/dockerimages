"""
Evaluate three-threshold PTR predictions against test_phase1.csv.

PTR parameters:
  T1=3.0, σ=0.70   T2=5.0, σ=0.55   T3=6.5, σ=0.50

Reconstruction:
  weight_k = ptr_k * (1 - ptr_k)
  pEC50 = Σ(weight_k * norm.ppf(ptr_k, T_k, σ_k)) / Σ(weight_k)

Usage:
  conda activate kepler_ai
  python evaluate_ptr3_predictions.py --pred_csv <model_dir>/test_prediction_ptr3.csv
"""

import argparse
import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

THRESHOLDS = [(3.0, 0.70), (5.0, 0.55), (6.5, 0.50)]
PTR_CLIP   = 1e-9

TRUTH_PATH = "/home/spal/dockerimages/QSARTuna/PXR/test_phase1.csv"
BASELINE   = {"name": "Chemprop cliff-smoothed", "mae": 0.4905, "n_gt05": 79}

BASELINES = {
    "Chemprop cliff-smoothed": "/home/spal/OpenAdmet_After_Phase1/OpenAdmet_Chemprop_rdkit_cliff_smoothed/test_prediction.csv",
    "rdkit2d ensemble":        "/home/spal/OpenAdmet/Prediction_OpenAdmet_pxr_rdkit2d_cw_ensemble_19_May_2026_cleaned.csv",
}


def ptr_reverse_weighted(ptr_preds):
    """Reconstruct pEC50 from 3 PTR predictions using weighted average."""
    ptrs = np.clip(ptr_preds, PTR_CLIP, 1 - PTR_CLIP)
    wts  = ptrs * (1 - ptrs)
    ests = np.array([norm.ppf(p, t, s) for p, (t, s) in zip(ptrs, THRESHOLDS)])
    total_w = wts.sum()
    if total_w < 1e-9:
        return float(np.mean(ests))
    return float((wts * ests).sum() / total_w)


def scorecard(truth, pred, label=""):
    truth, pred = np.asarray(truth, float), np.asarray(pred, float)
    ok  = np.isfinite(truth) & np.isfinite(pred)
    t, p = truth[ok], pred[ok]
    mae  = np.mean(np.abs(t - p))
    rmse = np.sqrt(np.mean((t - p)**2))
    r2   = np.corrcoef(t, p)[0, 1]**2
    rho  = spearmanr(t, p).statistic
    n_gt = (np.abs(t - p) >= 0.5).sum()
    print(f"\n{'='*60}")
    print(f"  {label}  (N={ok.sum()})")
    print(f"{'='*60}")
    print(f"  MAE          : {mae:.4f}")
    print(f"  RMSE         : {rmse:.4f}")
    print(f"  Pearson R²   : {r2:.4f}")
    print(f"  Spearman ρ   : {rho:.4f}")
    print(f"  |err| >= 0.5 : {n_gt}  ({n_gt/ok.sum()*100:.1f}%)")
    cuts   = pd.cut(t, bins=[0,3.5,4.5,5.5,10],
                    labels=["inactive(<3.5)","weak(3.5-4.5)","moderate(4.5-5.5)","potent(>5.5)"])
    print(f"\n  By activity class:")
    for lbl in ["inactive(<3.5)","weak(3.5-4.5)","moderate(4.5-5.5)","potent(>5.5)"]:
        m = cuts == lbl
        if m.sum():
            z = np.abs(t[m] - p[m])
            print(f"    {lbl:22s}  N={m.sum():3d}  MAE={z.mean():.4f}  |>=0.5: {(z>=0.5).sum()}")
    return mae


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", required=True)
    ap.add_argument("--output",   default="ptr3_pec50_predictions.csv")
    args = ap.parse_args()

    truth = pd.read_csv(TRUTH_PATH)[["Molecule Name","SMILES","pEC50",
                                      "Emax.vs.pos.ctrl_estimate (dimensionless)"]].rename(
        columns={"pEC50":"pEC50_exp",
                 "Emax.vs.pos.ctrl_estimate (dimensionless)":"Emax_exp"})

    pred = pd.read_csv(args.pred_csv)
    ptr1_col = next(c for c in pred.columns if "PRED" in c and "ptr1" in c)
    ptr2_col = next(c for c in pred.columns if "PRED" in c and "ptr2" in c)
    ptr3_col = next(c for c in pred.columns if "PRED" in c and "ptr3" in c)
    print(f"PTR columns: {ptr1_col}, {ptr2_col}, {ptr3_col}")

    df = pred[["Molecule Name","SMILES",ptr1_col,ptr2_col,ptr3_col]].merge(
         truth, on=["Molecule Name","SMILES"], how="inner")

    df["pEC50_pred"] = [
        ptr_reverse_weighted([r[ptr1_col], r[ptr2_col], r[ptr3_col]])
        for _, r in df.iterrows()
    ]

    # Show active weights per compound at prediction
    df["w1"] = df[ptr1_col].clip(PTR_CLIP,1-PTR_CLIP) * (1 - df[ptr1_col].clip(PTR_CLIP,1-PTR_CLIP))
    df["w2"] = df[ptr2_col].clip(PTR_CLIP,1-PTR_CLIP) * (1 - df[ptr2_col].clip(PTR_CLIP,1-PTR_CLIP))
    df["w3"] = df[ptr3_col].clip(PTR_CLIP,1-PTR_CLIP) * (1 - df[ptr3_col].clip(PTR_CLIP,1-PTR_CLIP))
    df["dominant_threshold"] = df[["w1","w2","w3"]].idxmax(axis=1).map(
        {"w1":"T=3.0","w2":"T=5.0","w3":"T=6.5"})

    # ── Baselines ─────────────────────────────────────────────────────────────
    b1 = pd.read_csv(BASELINES["Chemprop cliff-smoothed"])[["Molecule Name","pEC50_ensemble"]]
    b2 = pd.read_csv(BASELINES["rdkit2d ensemble"])[["Molecule Name","pEC50"]].rename(
         columns={"pEC50":"pEC50_b2"})
    df = df.merge(b1, on="Molecule Name", how="left")
    df = df.merge(b2, on="Molecule Name", how="left")

    scorecard(df["pEC50_exp"], df["pEC50_ensemble"], "Baseline — Chemprop cliff-smoothed")
    scorecard(df["pEC50_exp"], df["pEC50_b2"],       "Baseline — rdkit2d ensemble")
    mae_ptr3 = scorecard(df["pEC50_exp"], df["pEC50_pred"], "PTR3 — three-threshold reconstruction")

    # ── Weak full agonist breakdown ───────────────────────────────────────────
    wfa = df[(df["pEC50_exp"] < 3.5) & (df["Emax_exp"] > 0.85)]
    print(f"\n{'='*60}")
    print(f"  WEAK FULL AGONISTS (pEC50<3.5, Emax>0.85)  N={len(wfa)}")
    print(f"  Baseline1 MAE  : {(wfa['pEC50_ensemble']-wfa['pEC50_exp']).abs().mean():.4f}")
    print(f"  Baseline2 MAE  : {(wfa['pEC50_b2']-wfa['pEC50_exp']).abs().mean():.4f}")
    print(f"  PTR3 MAE       : {(wfa['pEC50_pred']-wfa['pEC50_exp']).abs().mean():.4f}")

    print(f"\n  Per-compound (sorted by baseline error):")
    wfa_out = wfa[["Molecule Name","pEC50_exp","pEC50_ensemble","pEC50_pred",
                   ptr1_col,ptr2_col,ptr3_col,"dominant_threshold"]].copy()
    wfa_out["err_b1"]  = (wfa_out["pEC50_ensemble"] - wfa_out["pEC50_exp"]).abs()
    wfa_out["err_ptr3"]= (wfa_out["pEC50_pred"]     - wfa_out["pEC50_exp"]).abs()
    print(wfa_out.sort_values("err_b1", ascending=False).round(3).to_string(index=False))

    df.to_csv(args.output, index=False)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
