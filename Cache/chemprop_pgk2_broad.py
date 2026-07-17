"""ChemProp binary classifier for PGK2 hit prediction (broad, no-competitor label).

Renamed from chemprop_pgk2_orthosteric.py on 2026-07-17: the training data
this script consumes (PGK2_train_broad_2to1/50to50.parquet, built by
build_train_variants.py) now uses the broad hit label — count_PGK2 >= 3,
count_NTC == 0, historic_hits < 5 — instead of the stricter orthosteric
criteria (count_PGK2_with_inhibitor < 0.1 * count_PGK2). The CACHE
organizers specifically want training compounds picked from data tested in
the absence of a competitor, i.e. the count_PGK2 condition alone; the old
orthosteric criteria used the count_PGK2_with_inhibitor column (only
meaningful for compounds tested *with* a competitor present) to help decide
which compounds counted as hits at all, which both violated that guidance
and diverged from this project's own 2026-07-06 decision to include both
orthosteric and allosteric binders in the primary label. See
PGK2_VALIDATION_SUBMISSION_LOG.md for the full history: the four ChemProp
validation submissions built from the old orthosteric-labeled training data
all scored 0 hits, and were dominated by the same BB3-805 chemotype found
in the LightGBM ensemble's picks — this retrain is the direct follow-up.

Architecture: ChemProp's default/recommended parameters. Iqbal et al. 2025
(npj Drug Discovery 2:5, "Evaluation of DNA encoded library and machine
learning model combinations for hit discovery") explicitly state in Methods
("Machine learning algorithms"): "The ChemProp models were generated using
the default, recommended parameters" — unlike the four classical baselines
(RF/SVM/MLP/XGB), which they hyperparameter-tuned via cross-validation. So
"the best architecture" for ChemProp in that paper *is* the software default,
not a custom search result. Defaults below are read directly from chemprop
2.2.3's CLI (`chemprop.cli.train.TrainSubcommand`):
    message passing : BondMessagePassing, depth=3, hidden_dim=300, dropout=0.0
    aggregation      : norm  (NormAggregation)
    FFN              : BinaryClassificationFFN, hidden_dim=300, n_layers=1, dropout=0.0
    training         : epochs=50, batch_size=64, warmup_epochs=2,
                        init_lr=1e-4, max_lr=1e-3, final_lr=1e-4

Feature representation: pure ChemProp graph features, no concatenated RDKit
descriptors. Iqbal et al.'s Methods ("Feature representation") state the
2048-bit Morgan FP + six physchem descriptors were used for the four
classical models only; ChemProp used its own graph-derived features alone.

Internal split: scaffold-balanced 80/20 (chemprop's SCAFFOLD_BALANCED split),
not random. This project's own hit-series analysis
(~/Cache/pipeline/hit_series_analysis.py) found PGK2 DEL hits are heavily
organized into BB-driven series (53% of hits in multi-member Bemis-Murcko
scaffold series, one BB3 alone connecting 5,329 hits) — a random split would
leak near-duplicate scaffolds between train/val and overstate performance.

Trains one 5-seed ensemble per training-set variant (2:1 and 50:50
negative:positive, both built by build_train_variants.py with
cluster-stratified negative sampling).

Usage:
    conda activate chemprop
    python chemprop_pgk2_broad.py
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightning import pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from rdkit import Chem, RDLogger
from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef, recall_score

from chemprop import data, featurizers, models, nn
from chemprop.data.splitting import SplitType

RDLogger.DisableLog("rdApp.*")

# ── Paths ────────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent

VARIANTS = {
    "2to1": HERE / "PGK2_train_broad_2to1.parquet",
    "50to50": HERE / "PGK2_train_broad_50to50.parquet",
}

MODEL_DIR = HERE / "chemprop_models"
RESULTS_PATH = HERE / "chemprop_variant_results.csv"

# ── ChemProp default/recommended architecture (Iqbal et al.) ──────────────────
MP_DEPTH = 3
MP_HIDDEN_DIM = 300
DROPOUT = 0.0
FFN_HIDDEN_DIM = 300
FFN_N_LAYERS = 1

# ── Training (chemprop CLI defaults) ───────────────────────────────────────────
MAX_EPOCHS = 50
BATCH_SIZE = 64
WARMUP_EPOCHS = 2
INIT_LR = 1e-4
MAX_LR = 1e-3
FINAL_LR = 1e-4
PATIENCE = 10
NUM_WORKERS = 0

SPLIT_SIZES = (0.8, 0.2, 0.0)  # scaffold-balanced train/val, no held-out test here
SPLIT_SEED = 42
ENSEMBLE_SEEDS = [42, 123, 456, 789, 1337]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_variant(path: Path):
    df = pd.read_parquet(path, columns=["SMILES", "LABEL"])
    mols, labels = [], []
    for smi, y in zip(df["SMILES"], df["LABEL"]):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        mols.append(mol)
        labels.append(float(y))
    return mols, np.array(labels, dtype=float)


def scaffold_split(mols, seed=SPLIT_SEED):
    train_reps, val_reps, _ = data.make_split_indices(
        mols, split=SplitType.SCAFFOLD_BALANCED, sizes=SPLIT_SIZES, seed=seed, num_replicates=1
    )
    return np.array(train_reps[0]), np.array(val_reps[0])


def build_mpnn() -> models.MPNN:
    feat = featurizers.SimpleMoleculeMolGraphFeaturizer()
    mp = nn.BondMessagePassing(
        d_v=feat.atom_fdim, d_e=feat.bond_fdim,
        depth=MP_DEPTH, d_h=MP_HIDDEN_DIM, dropout=DROPOUT,
    )
    agg = nn.NormAggregation()
    ffn = nn.BinaryClassificationFFN(
        input_dim=mp.output_dim, hidden_dim=FFN_HIDDEN_DIM, n_layers=FFN_N_LAYERS,
        dropout=DROPOUT, criterion=nn.metrics.BCELoss(),
    )
    return models.MPNN(
        mp, agg, ffn, batch_norm=False,
        metrics=[nn.metrics.BinaryAUROC(), nn.metrics.BinaryF1Score(), nn.metrics.BinaryMCCMetric()],
        warmup_epochs=WARMUP_EPOCHS, init_lr=INIT_LR, max_lr=MAX_LR, final_lr=FINAL_LR,
    )


def make_datapoints(mols, labels):
    return [data.MoleculeDatapoint(mol=m, y=np.array([y], dtype=float)) for m, y in zip(mols, labels)]


def predict_proba(mpnn, loader, trainer):
    preds = torch.cat(trainer.predict(mpnn, loader)).numpy().flatten()
    return preds


def report_metrics(y_true, y_pred_proba, label=""):
    y_pred = (y_pred_proba >= 0.5).astype(int)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    tag = f"  [{label}]" if label else ""
    print(f"{tag}  BalancedAcc={bal_acc:.4f}  MCC={mcc:.4f}  F1={f1:.4f}  Recall={recall:.4f}")
    return dict(balanced_accuracy=bal_acc, mcc=mcc, f1=f1, recall=recall)


def train_one_seed(train_mols, train_labels, val_mols, val_labels, seed):
    set_seed(seed)
    feat = featurizers.SimpleMoleculeMolGraphFeaturizer()
    train_dset = data.MoleculeDataset(make_datapoints(train_mols, train_labels), feat)
    val_dset = data.MoleculeDataset(make_datapoints(val_mols, val_labels), feat)
    train_loader = data.build_dataloader(train_dset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    val_loader = data.build_dataloader(
        val_dset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False
    )

    mpnn = build_mpnn()
    trainer = pl.Trainer(
        logger=False, enable_checkpointing=False, enable_progress_bar=True,
        accelerator="auto", devices=1, max_epochs=MAX_EPOCHS,
        callbacks=[EarlyStopping(monitor="val_loss", patience=PATIENCE, mode="min")],
    )
    trainer.fit(mpnn, train_loader, val_loader)

    mpnn.eval()
    val_preds = predict_proba(mpnn, val_loader, trainer)
    return mpnn, val_preds


def run_variant(name, path):
    print(f"\n{'='*70}\nVariant: {name}  ({path.name})\n{'='*70}")
    mols, labels = load_variant(path)
    print(f"  Loaded {len(mols)} molecules  (positives={int(labels.sum())}, "
          f"negatives={int((labels == 0).sum())})")

    train_idx, val_idx = scaffold_split(mols)
    train_mols = [mols[i] for i in train_idx]
    train_labels = labels[train_idx]
    val_mols = [mols[i] for i in val_idx]
    val_labels = labels[val_idx]
    print(f"  Scaffold-balanced split: train={len(train_mols)} "
          f"(pos={int(train_labels.sum())}), val={len(val_mols)} (pos={int(val_labels.sum())})")

    variant_dir = MODEL_DIR / name
    variant_dir.mkdir(parents=True, exist_ok=True)

    ensemble_val_preds = []
    for i, seed in enumerate(ENSEMBLE_SEEDS):
        print(f"\n-- {name}: ensemble member {i+1}/{len(ENSEMBLE_SEEDS)}  seed={seed} --")
        mpnn, val_preds = train_one_seed(train_mols, train_labels, val_mols, val_labels, seed)
        torch.save(mpnn, variant_dir / f"model_seed{seed}.pt")
        ensemble_val_preds.append(val_preds)
        report_metrics(val_labels, val_preds, label=f"seed {seed}")

    ensemble_preds = np.mean(np.stack(ensemble_val_preds), axis=0)
    print(f"\n-- {name}: ensemble mean --")
    metrics = report_metrics(val_labels, ensemble_preds, label="ensemble")
    metrics["variant"] = name
    metrics["n_train"] = len(train_mols)
    metrics["n_val"] = len(val_mols)
    return metrics


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []
    for name, path in VARIANTS.items():
        if not path.exists():
            print(f"Skipping {name}: {path} not found (run build_train_variants.py first)")
            continue
        all_results.append(run_variant(name, path))

    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(RESULTS_PATH, index=False)
        print(f"\n{'='*70}\nSaved variant comparison to {RESULTS_PATH}\n{df.to_string(index=False)}")


if __name__ == "__main__":
    main()
