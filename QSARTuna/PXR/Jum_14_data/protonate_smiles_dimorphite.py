import dimorphite_dl
import numpy as np
import pandas as pd
from rdkit.Chem import PandasTools
from rdkit import Chem

def _protonate_mol(smi: str , ph: float = 7.4) -> Chem.Mol:
    """Protonate a molecule at the given pH using dimorphite_dl.

    Converts to SMILES, protonates, converts back to mol.
    Falls back to the original mol if dimorphite_dl fails or returns an
    unparseable SMILES.
    """
    try:
        variants = dimorphite_dl.protonate_smiles(
            smi, ph_min=ph, ph_max=ph, max_variants=1, validate_output=True
        )
        prot_smi = variants[0] if variants else smi
    except Exception:
        prot_smi = smi
        print("Sandeep")
    prot_mol = Chem.MolFromSmiles(prot_smi)
    prot_mol = Chem.AddHs(prot_mol)
    return prot_mol if prot_mol is not None else mol


df = pd.read_csv("trainining_set_AND_phase_one_results_4393.csv")
df['ROMol'] = df['SMILES'].apply(lambda x: _protonate_mol(x))

for _,x in df.iterrows():
    id_ = x['Molecule Name']
    x['ROMol'].SetProp("_Name",id_)


PandasTools.WriteSDF(
    df,
    "trainining_set_AND_phase_one_results_4393_dimorphite.sdf",
    molColName="ROMol",
    idName=None,
    properties=list(df.columns)
)
