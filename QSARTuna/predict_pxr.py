# predict_pxr.py
# conda activate chemprop
# python predict_pxr.py --input molecules.csv --smiles_col SMILES --output preds.csv

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightning import pytorch as pl
from chemprop import data, featurizers

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default=str(Path.home() / "pxr_chemeleon_final.pkl"))
    parser.add_argument("--input",      required=True, help="CSV or .smi file")
    parser.add_argument("--smiles_col", default="SMILES")
    parser.add_argument("--name_col",   default=None)
    parser.add_argument("--output",     default="pxr_predictions.csv")
    args = parser.parse_args()

    # Load model
    model = torch.load(args.model, weights_only=False)
    model.eval()

    # Read input
    df = pd.read_csv(args.input)
    smiles_list = df[args.smiles_col].tolist()
    names = df[args.name_col].tolist() if args.name_col else smiles_list

    # Build datapoints (dummy target=0.0, ignored at predict time)
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    points, valid_idx = [], []
    for i, smi in enumerate(smiles_list):
        dp = data.MoleculeDatapoint.from_smi(smi, [0.0])
        if dp.mol is not None:
            points.append(dp)
            valid_idx.append(i)
        else:
            print(f"  Warning: skipped unparseable SMILES at row {i}: {smi}")

    dataset    = data.MoleculeDataset(points, featurizer)
    loader     = data.build_dataloader(dataset, num_workers=0, shuffle=False)

    # Predict
    trainer    = pl.Trainer(logger=False, enable_progress_bar=True, accelerator="auto", devices=1)
    raw_preds  = trainer.predict(model, loader)
    preds      = torch.cat(raw_preds).numpy().flatten()

    # Output
    df_out = pd.DataFrame({
        "Name":            [names[i] for i in valid_idx],
        "SMILES":          [smiles_list[i] for i in valid_idx],
        "pEC50_predicted": preds,
    })
    df_out.to_csv(args.output, index=False)
    print(f"\nPredictions saved to {args.output}")
    print(df_out.head())

if __name__ == "__main__":
    main()
