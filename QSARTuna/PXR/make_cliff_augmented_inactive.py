"""
SMILES augmentation for inactive compounds (pEC50 <= 3.5).

Instead of upweighting inactive compounds, generates 3 additional randomized
SMILES representations per inactive compound. Each augmented row carries the
same pEC50 and pEC50_counter as the original. All weights are set to 1.0.

This increases the effective count of inactives from 959 to ~3836
(959 original + 959*3 augmented), bringing the active/inactive ratio from
3.3x down to ~0.8x (slightly over-representing inactives to compensate for
their higher experimental noise).

Input  : train_cliff_error_weighted_median.csv
Output : train_cliff_error_augmented.csv
"""

import random

import numpy as np
import pandas as pd
from rdkit import Chem

INPUT_CSV  = "/home/spal/dockerimages/QSARTuna/PXR/train_cliff_error_weighted_median.csv"
OUTPUT_CSV = "/home/spal/dockerimages/QSARTuna/PXR/train_cliff_error_augmented.csv"

INACTIVE_THRESHOLD = 3.5
N_AUGMENTATIONS    = 2


def randomize_smiles(mol, seed):
    random.seed(seed)
    order = list(range(mol.GetNumAtoms()))
    random.shuffle(order)
    return Chem.MolToSmiles(Chem.RenumberAtoms(mol, order), canonical=False)


df = pd.read_csv(INPUT_CSV)
df["weight"] = 1.0   # remove weighting — augmentation handles imbalance

inactive_mask = df["pEC50"] <= INACTIVE_THRESHOLD
inactive_df   = df[inactive_mask].copy()
active_df     = df[~inactive_mask].copy()

print(f"Loaded {len(df)} compounds")
print(f"  Active   (pEC50 > {INACTIVE_THRESHOLD}) : {len(active_df)}")
print(f"  Inactive (pEC50 <= {INACTIVE_THRESHOLD}) : {len(inactive_df)}")

augmented_rows = []
failed = 0

for idx, (_, row) in enumerate(inactive_df.iterrows()):
    mol = Chem.MolFromSmiles(row["SMILES"])
    if mol is None:
        failed += 1
        continue

    seen = {row["SMILES"]}
    aug_count = 0
    attempts  = 0

    while aug_count < N_AUGMENTATIONS and attempts < N_AUGMENTATIONS * 10:
        rand_smi = randomize_smiles(mol, seed=idx * 100 + attempts)
        attempts += 1
        if rand_smi not in seen:
            seen.add(rand_smi)
            new_row = row.copy()
            new_row["SMILES"] = rand_smi
            augmented_rows.append(new_row)
            aug_count += 1

aug_df = pd.DataFrame(augmented_rows)

out = pd.concat([df, aug_df], ignore_index=True)
out = out[["Molecule Name", "SMILES", "pEC50", "pEC50_counter", "weight"]]
out.to_csv(OUTPUT_CSV, index=False)

print(f"\nAugmented rows added : {len(aug_df)}")
print(f"  ({failed} inactive compounds failed SMILES parsing)")
print(f"\nOutput rows          : {len(out)}  (was {len(df)})")
print(f"  Active             : {(out['pEC50'] > INACTIVE_THRESHOLD).sum()}")
print(f"  Inactive (all)     : {(out['pEC50'] <= INACTIVE_THRESHOLD).sum()}")
print(f"  Active/inactive ratio: {(out['pEC50'] > INACTIVE_THRESHOLD).sum() / (out['pEC50'] <= INACTIVE_THRESHOLD).sum():.2f}x")
print(f"\nSaved: {OUTPUT_CSV}")
