"""Score the ASMS PGK2 validation split with trained ChemProp ensembles.

For each training-set variant (2:1 and 50:50 cluster-stratified negative
sampling), loads the saved 5-seed ensemble from
`chemprop_models/<variant>/model_seed*.pt` (produced by
chemprop_pgk2_broad.py) and scores every compound in
`~/Cache/Val-Test-set/PGK2_Validation_split.csv` (244,328 compounds,
unlabeled) — the CACHE challenge's true out-of-distribution validation set,
as distinct from the internal scaffold-holdout used during training.

Per Iqbal et al. and the CACHE strategy doc
(~/Cache/Literature/Iqbal_et_al_DEL_ML_Analysis_and_CACHE_Strategy.md,
recommendation #4), model/negative-sampling choices should ultimately be
ranked by performance on this validation split rather than internal
cross-validation accuracy — this script produces the raw ensemble prediction
scores needed for that comparison, and keeps both variants' scores as
separate output files rather than converging on one (recommendation #5).

The "top N" preview printed here is a plain score-sorted cut for now.
Diversity-aware final compound selection (clustering predicted binders,
picking confident representatives per cluster — Iqbal et al.'s method for
their SPR follow-up) is a later refinement, not implemented here.

Usage:
    conda activate chemprop
    python predict_validation.py

    # BB3-805 ablation variant (see build_no_bb3805_variants.py): scores
    # with the chemprop_models_no_bb3805/ ensemble instead, writing
    # PGK2_validation_predictions_{2to1,50to50}_no_bb3805.parquet.
    python predict_validation.py --suffix no_bb3805
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightning import pytorch as pl
from rdkit import Chem, RDLogger

from chemprop import data, featurizers

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent
VALIDATION_CSV = Path.home() / "Cache" / "Val-Test-set" / "PGK2_Validation_split.csv"
OUT_DIR = HERE

BATCH_SIZE = 256
NUM_WORKERS = 0
TOP_N_PREVIEW = 50


def load_validation_mols():
    df = pd.read_csv(VALIDATION_CSV)
    mols = []
    valid_mask = np.zeros(len(df), dtype=bool)
    for i, smi in enumerate(df["SMILES"]):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        mols.append(mol)
        valid_mask[i] = True
    df = df.loc[valid_mask].reset_index(drop=True)
    print(
        f"Loaded {len(df)}/{len(valid_mask)} valid validation compounds "
        f"({(~valid_mask).sum()} unparseable SMILES skipped)"
    )
    return df, mols


def predict_ensemble(mols, variant_dir, trainer):
    feat = featurizers.SimpleMoleculeMolGraphFeaturizer()
    dps = [data.MoleculeDatapoint(mol=m, y=np.array([0.0])) for m in mols]
    dset = data.MoleculeDataset(dps, feat)
    loader = data.build_dataloader(
        dset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False
    )

    seed_paths = sorted(variant_dir.glob("model_seed*.pt"))
    all_preds = []
    for p in seed_paths:
        print(f"  predicting with {p.name}...")
        mpnn = torch.load(p, weights_only=False)
        mpnn.eval()
        preds = torch.cat(trainer.predict(mpnn, loader)).numpy().flatten()
        all_preds.append(preds)

    arr = np.stack(all_preds)
    seed_names = [p.stem.replace("model_", "") for p in seed_paths]
    return arr.mean(axis=0), arr.std(axis=0), seed_names, arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suffix", default=None,
        help="Optional variant suffix matching chemprop_pgk2_broad.py's "
             "--suffix, e.g. 'no_bb3805' to score with "
             "chemprop_models_no_bb3805/ instead of the default chemprop_models/.",
    )
    args = parser.parse_args()
    tag = f"_{args.suffix}" if args.suffix else ""
    model_dir = HERE / f"chemprop_models{tag}"

    if not model_dir.exists():
        raise FileNotFoundError(
            f"{model_dir} not found — run chemprop_pgk2_broad.py"
            f"{' --suffix ' + args.suffix if args.suffix else ''} "
            "(locally or on the VM, then copy it back) first."
        )

    df, mols = load_validation_mols()
    trainer = pl.Trainer(
        logger=False, enable_checkpointing=False, enable_progress_bar=True,
        accelerator="auto", devices=1,
    )

    variant_dirs = sorted(p for p in model_dir.iterdir() if p.is_dir())
    for variant_dir in variant_dirs:
        seed_paths = sorted(variant_dir.glob("model_seed*.pt"))
        if not seed_paths:
            print(f"Skipping {variant_dir.name}: no trained models found")
            continue

        print(
            f"\n{'='*70}\nScoring validation set with {variant_dir.name} ensemble "
            f"({len(seed_paths)} seeds)\n{'='*70}"
        )
        mean_pred, std_pred, seed_names, per_seed = predict_ensemble(mols, variant_dir, trainer)

        out = df.copy()
        out["pred_proba_mean"] = mean_pred
        out["pred_proba_std"] = std_pred
        for name, preds in zip(seed_names, per_seed):
            out[f"pred_{name}"] = preds
        out = out.sort_values("pred_proba_mean", ascending=False).reset_index(drop=True)

        out_path = OUT_DIR / f"PGK2_validation_predictions_{variant_dir.name}{tag}.parquet"
        out.to_parquet(out_path, index=False)
        print(f"Saved {len(out)} scored compounds -> {out_path}")

        print(f"\nTop {TOP_N_PREVIEW} by ensemble mean probability ({variant_dir.name}):")
        print(
            out.head(TOP_N_PREVIEW)[["CatalogID", "SMILES", "pred_proba_mean", "pred_proba_std"]]
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
