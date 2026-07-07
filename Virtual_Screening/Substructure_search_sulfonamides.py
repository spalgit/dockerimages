import os
import sys
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd

df = pd.read_csv("combined_Reduced_Graph_Enamine_Mcule_ALL.csv")

sulfonamide = Chem.MolFromSmarts("NS(=O)(=O)")

df['sulfonamide'] = df['SMILES'].apply(lambda x: Chem.MolFromSmiles(x).HasSubstructMatch(sulfonamide))

df[df['sulfonamide']==True].to_csv("check.csv",index=False)
