"""
Three-threshold PTR label generation.

Thresholds chosen to give good gradient signal across the full range:
  T1 = 3.0, σ=0.70  — covers inactive/weak region  (pEC50 1.6–4.4)
  T2 = 5.0, σ=0.55  — covers moderate/active region (pEC50 3.4–6.7)  [same as v3]
  T3 = 6.5, σ=0.50  — covers potent/highly potent   (pEC50 5.0–8.0)

Forward transform:
    PTR_k(y) = norm.cdf(y, T_k, σ_k)

Reverse transform (weighted average at inference):
    weight_k = PTR_k * (1 - PTR_k)       # peaks at 0.5, zero at saturation
    pEC50_pred = Σ(weight_k * norm.ppf(PTR_k, T_k, σ_k)) / Σ(weight_k)

Why three thresholds fixes the inactive overprediction:
    Single PTR at T=5.0: pEC50=1.7 → PTR≈0.0000 (no gradient)
                         pEC50=3.5 → PTR=0.0032 (still almost no gradient)
    PTR at T=3.0:        pEC50=1.7 → PTR=0.032  (strong gradient)
                         pEC50=2.5 → PTR=0.238  (strong gradient)
                         pEC50=3.5 → PTR=0.763  (strong gradient)
    Model now discriminates within the weak full agonist population.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

# ── Parameters ────────────────────────────────────────────────────────────────
THRESHOLDS = [(3.0, 0.70), (5.0, 0.55), (6.5, 0.50)]
PTR_CLIP   = 1e-9

BASE_FILE  = "/home/spal/dockerimages/QSARTuna/PXR/train_multitask_weighted.csv"
OUT_FILE   = "/home/spal/dockerimages/QSARTuna/PXR/train_ptr3.csv"


def ptr_forward(y, threshold, std):
    result = np.full(len(y), np.nan)
    mask = np.isfinite(y)
    result[mask] = norm.cdf(y[mask], threshold, std).clip(PTR_CLIP, 1 - PTR_CLIP)
    return result


def ptr_reverse_weighted(ptrs, thresholds):
    """Weighted average reconstruction from multiple PTR predictions."""
    estimates, weights = [], []
    for ptr, (t, s) in zip(ptrs, thresholds):
        ptr_c = np.clip(ptr, PTR_CLIP, 1 - PTR_CLIP)
        estimates.append(norm.ppf(ptr_c, t, s))
        weights.append(ptr_c * (1 - ptr_c))
    total_w = sum(weights)
    if total_w < 1e-6:
        return np.mean(estimates)
    return sum(w * e for w, e in zip(weights, estimates)) / total_w


# ── Load base training data ───────────────────────────────────────────────────
base = pd.read_csv(BASE_FILE)
pec50 = base["pEC50"].values

# ── Apply PTR transforms ──────────────────────────────────────────────────────
out = pd.DataFrame({
    "Molecule Name": base["Molecule Name"],
    "SMILES":        base["SMILES"],
    "pEC50_ptr1":    ptr_forward(pec50, *THRESHOLDS[0]),   # T=3.0
    "pEC50_ptr2":    ptr_forward(pec50, *THRESHOLDS[1]),   # T=5.0
    "pEC50_ptr3":    ptr_forward(pec50, *THRESHOLDS[2]),   # T=6.5
    "weight":        base["weight"],
})

out.to_csv(OUT_FILE, index=False)

# ── Validation ────────────────────────────────────────────────────────────────
print(f"Saved: {OUT_FILE}  ({len(out)} rows, 3 PTR tasks)")
print()
print(f"{'pEC50':>8}  {'PTR1(T=3.0)':>12}  {'PTR2(T=5.0)':>12}  {'PTR3(T=6.5)':>12}  {'Reconstructed':>14}  {'RoundTripErr':>13}")
for pec in [1.7, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]:
    ptrs = [np.clip(norm.cdf(pec, t, s), PTR_CLIP, 1-PTR_CLIP) for t, s in THRESHOLDS]
    rec  = ptr_reverse_weighted(ptrs, THRESHOLDS)
    print(f"{pec:>8.1f}  {ptrs[0]:>12.5f}  {ptrs[1]:>12.5f}  {ptrs[2]:>12.5f}  {rec:>14.4f}  {abs(pec-rec):>13.5f}")

print()
print("PTR label distributions:")
for col in ["pEC50_ptr1", "pEC50_ptr2", "pEC50_ptr3"]:
    print(f"  {col}: min={out[col].min():.4f}  mean={out[col].mean():.4f}  max={out[col].max():.4f}")

print()
print("Weight distribution:")
print(out["weight"].value_counts().sort_index().to_string())
