"""
Inference script: F2 ChemProp ensemble → Phase 1 test set

Steps:
  1. Extract phase=1 compounds from the F1 SDF → test_phase1.sdf
  2. Re-derive descriptor column mask + StandardScaler from the F2 training set
     (mirrors what the training script does; these artefacts were not saved separately)
  3. Load each saved ensemble .pt model and predict
  4. Report performance metrics against known pEC50 labels

Usage (inside the VM):
    conda activate chemprop
    python predict_phase1_f2_model.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightning import pytorch as pl
from rdkit import Chem
from rdkit.Chem import Descriptors
from scipy import stats
from sklearn.preprocessing import StandardScaler

from chemprop import data, featurizers, models

# ── Paths ──────────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent

F1_SDF         = HERE / "train_set_AND_phase_one_results_4392_ligpreped_f_1_n_1_2_3.sdf"
F2_TRAIN_SDF   = HERE / "train_set_4139_ligpreped_f_2_n_1_2_3.sdf"
ENSEMBLE_DIR   = HERE / "f2_ensemble_models"
TEST_PHASE1_SDF = HERE / "test_phase1.sdf"
OUTPUT_CSV     = HERE / "test_phase1_f2_predictions.csv"

ENSEMBLE_SEEDS = [42, 123, 456, 789, 1337, 2024, 31415, 99999]
NUM_WORKERS    = 0

# ── RDKit descriptor helpers ───────────────────────────────────────────────────
_DESC_LIST     = [(name, fn) for name, fn in Descriptors.descList]
ALL_DESC_NAMES = [name for name, _ in _DESC_LIST]


def compute_rdkit_descriptors(mols: list) -> np.ndarray:
    rows = []
    for mol in mols:
        row = []
        for _, fn in _DESC_LIST:
            try:
                v = fn(mol)
                row.append(float(v) if v is not None else np.nan)
            except Exception:
                row.append(np.nan)
        rows.append(row)
    return np.array(rows, dtype=float)


def select_valid_columns(arr: np.ndarray) -> np.ndarray:
    finite = np.all(np.isfinite(arr), axis=0)
    varied = np.var(arr, axis=0) > 0
    return finite & varied


# ── Metrics ────────────────────────────────────────────────────────────────────
def report_metrics(actual: np.ndarray, predicted: np.ndarray, label: str = "") -> dict:
    mae  = float(np.mean(np.abs(actual - predicted)))
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - np.mean(actual)) ** 2)
    r2   = float(1.0 - ss_res / ss_tot)
    range_actual = actual.max() - actual.min()
    rae  = float(mae / (range_actual / 2.0)) if range_actual > 0 else float("nan")
    rho, _ = stats.spearmanr(actual, predicted)
    tau, _ = stats.kendalltau(actual, predicted)
    tag = f"[{label}]  " if label else ""
    print(
        f"  {tag}MAE={mae:.4f}  RMSE={rmse:.4f}  RAE={rae:.4f}  "
        f"R²={r2:.4f}  Spearman={rho:.4f}  Kendall={tau:.4f}"
    )
    return dict(mae=mae, rmse=rmse, rae=rae, r2=r2, spearman=rho, kendall=tau)


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — Extract phase=1 compounds → test_phase1.sdf
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Step 1: Extracting phase=1 compounds from F1 SDF")
print(f"{'='*70}")

suppl = Chem.SDMolSupplier(str(F1_SDF), removeHs=True)
writer = Chem.SDWriter(str(TEST_PHASE1_SDF))
phase1_count = 0
for mol in suppl:
    if mol is None:
        continue
    props = mol.GetPropsAsDict()
    phase_val = props.get("phase", None)
    try:
        if phase_val is not None and abs(float(phase_val) - 1.0) < 1e-6:
            writer.write(mol)
            phase1_count += 1
    except (TypeError, ValueError):
        pass
writer.close()
print(f"  Written {phase1_count} phase=1 compounds → {TEST_PHASE1_SDF}")

# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — Load test_phase1.sdf (with known pEC50 labels)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Step 2: Loading test_phase1.sdf")
print(f"{'='*70}")

test_suppl = Chem.SDMolSupplier(str(TEST_PHASE1_SDF), removeHs=True)
test_mols, test_names, test_smiles, test_pec50 = [], [], [], []
for mol in test_suppl:
    if mol is None:
        continue
    test_mols.append(mol)
    test_names.append(mol.GetProp("_Name") if mol.HasProp("_Name") else mol.GetPropsAsDict().get("Molecule Name", ""))
    test_smiles.append(Chem.MolToSmiles(mol))
    try:
        test_pec50.append(float(mol.GetPropsAsDict().get("pEC50", np.nan)))
    except (TypeError, ValueError):
        test_pec50.append(np.nan)

test_pec50 = np.array(test_pec50, dtype=float)
has_label  = ~np.isnan(test_pec50)
print(f"  Loaded {len(test_mols)} molecules")
print(f"  Molecules with known pEC50: {has_label.sum()}")
print(f"  pEC50 range: {np.nanmin(test_pec50):.2f} – {np.nanmax(test_pec50):.2f}  "
      f"(mean {np.nanmean(test_pec50):.2f})")

# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — Load F2 training set and re-derive descriptor mask + scaler
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Step 3: Re-deriving descriptor selector and scaler from F2 training set")
print(f"{'='*70}")

train_suppl = Chem.SDMolSupplier(str(F2_TRAIN_SDF), removeHs=True)
train_mols  = [m for m in train_suppl if m is not None]
print(f"  Loaded {len(train_mols)} F2 training molecules")

print("  Computing RDKit 2D descriptors for training set...")
x_d_train_all = compute_rdkit_descriptors(train_mols)
col_mask      = select_valid_columns(x_d_train_all)
x_d_train_raw = x_d_train_all[:, col_mask]
n_descriptors = x_d_train_raw.shape[1]
print(f"  Kept {n_descriptors} / {len(col_mask)} descriptors")

final_xd_scaler  = StandardScaler()
x_d_train_scaled = final_xd_scaler.fit_transform(x_d_train_raw)

print("  Computing RDKit 2D descriptors for test set...")
x_d_test_all    = compute_rdkit_descriptors(test_mols)
x_d_test_raw    = x_d_test_all[:, col_mask]
x_d_test_scaled = final_xd_scaler.transform(x_d_test_raw)

# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — Build chemprop DataLoader for test set
# ══════════════════════════════════════════════════════════════════════════════
feat = featurizers.SimpleMoleculeMolGraphFeaturizer()

test_dps = [
    data.MoleculeDatapoint(
        mol=mol,
        y=np.array([0.0]),           # dummy label; not used for inference
        weight=1.0,
        x_d=xd.astype(float),
    )
    for mol, xd in zip(test_mols, x_d_test_scaled)
]
test_dset   = data.MoleculeDataset(test_dps, feat)
test_loader = data.build_dataloader(test_dset, num_workers=NUM_WORKERS, shuffle=False)

# ══════════════════════════════════════════════════════════════════════════════
# Step 5 — Load each ensemble .pt model and predict
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Step 5: Running ensemble inference")
print(f"{'='*70}")

trainer = pl.Trainer(
    logger=False,
    enable_checkpointing=False,
    enable_progress_bar=False,
    accelerator="auto",
    devices=1,
)

all_preds = []
for seed in ENSEMBLE_SEEDS:
    model_path = ENSEMBLE_DIR / f"model_seed{seed}.pt"
    if not model_path.exists():
        print(f"  WARNING: {model_path} not found — skipping seed {seed}")
        continue
    mpnn = torch.load(model_path, map_location="cpu", weights_only=False)
    mpnn.eval()
    raw = trainer.predict(mpnn, test_loader)
    preds_i = torch.cat(raw).numpy().flatten()
    all_preds.append((seed, preds_i))
    print(f"  seed={seed:6d}  min={preds_i.min():.3f}  max={preds_i.max():.3f}  "
          f"mean={preds_i.mean():.3f}")

if not all_preds:
    raise RuntimeError(f"No model files found in {ENSEMBLE_DIR}. "
                       "Run the training script first or check the path.")

# ══════════════════════════════════════════════════════════════════════════════
# Step 6 — Ensemble mean + metrics
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("Step 6: Ensemble results")
print(f"{'='*70}")

seeds_used, preds_matrix = zip(*all_preds)
preds_array    = np.stack(preds_matrix, axis=0)
ensemble_preds = preds_array.mean(axis=0)
ensemble_std   = preds_array.std(axis=0)

print(f"\nEnsemble ({len(seeds_used)} seeds) prediction summary:")
print(f"  pEC50 range  : {ensemble_preds.min():.3f} – {ensemble_preds.max():.3f}")
print(f"  pEC50 mean   : {ensemble_preds.mean():.3f}")
print(f"  Mean std dev : {ensemble_std.mean():.3f}")

if has_label.sum() > 1:
    print(f"\nPerformance vs. known pEC50 (n={has_label.sum()}):")
    metrics = report_metrics(test_pec50[has_label], ensemble_preds[has_label], label="F2 ensemble → Phase 1")

# ══════════════════════════════════════════════════════════════════════════════
# Step 7 — Save predictions
# ══════════════════════════════════════════════════════════════════════════════
pred_cols = {f"pEC50_seed{s}": p for s, p in zip(seeds_used, preds_matrix)}
df_out = pd.DataFrame({
    "Molecule Name":  test_names,
    "SMILES":         test_smiles,
    "pEC50_actual":   test_pec50,
    "pEC50_ensemble": ensemble_preds,
    "pEC50_std":      ensemble_std,
    **pred_cols,
})
df_out.to_csv(OUTPUT_CSV, index=False)
print(f"\nPredictions saved → {OUTPUT_CSV}")
print(f"  Columns: Molecule Name, SMILES, pEC50_actual, pEC50_ensemble, "
      f"pEC50_std, {', '.join(pred_cols.keys())}")
