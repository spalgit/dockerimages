import os
from rdkit import Chem
import pandas as pd
import json
from openbabel import pybel
import pandas as pd
from rdkit import Chem
from strain_relief import compute_strain
from rdkit.Chem import PandasTools, rdMolDescriptors
import yaml
from strain_relief.compute_strain import compute_strain
from omegaconf import OmegaConf
import pickle
from hydra import compose, initialize
from omegaconf import OmegaConf
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds, AllChem
from rdkit.Chem.Draw import IPythonConsole
IPythonConsole.ipython_3d = True
import pandas as pd
import subprocess
import sys
from hydra import initialize_config_dir, compose


input_directory = '/home/spal/Ache/VRK1/Enumerated_comps'
output_file = f'{input_directory}/prolif_REF.sdf'
input_parque = f'{input_directory}/input_strain_relief_file.parque'
output_parque = f'{input_directory}/output_strain_relief_file.parque'
output_csv = f'{input_directory}/output_strain_relief_file.csv'
sdf_output = f'{input_directory}/prolif_strain_annotated.sdf'

# Calcualte Ligand strain energies

df_sdf = PandasTools.LoadSDF(output_file,
                        smilesName='SMILES',      # Add SMILES column
                        molColName='ROMol',    # Rename mol column
                        embedProps=True,
                            removeHs=False)


boron_query = Chem.MolFromSmarts('[#5]')

df_sdf['Has_Boron'] = df_sdf['ROMol'].apply(lambda x: x.HasSubstructMatch(boron_query))

df_sdf = df_sdf[df_sdf['Has_Boron']==False]


with initialize_config_dir(
    version_base="1.1",
    config_dir="/home/spal/StrainRelief/src/strain_relief/hydra_config",
):
    cfg = compose(config_name="default")


df_strain = pd.DataFrame([{"mol_bytes": mol.ToBinary(), **mol.GetPropsAsDict()} for mol in df_sdf['ROMol']])
df_strain = df_strain.reset_index(drop=False, names='id')
#df_strain['ID'] = df_sdf['ID']

for col in df_sdf.columns:
    if(col != 'ROMol'):
        df_strain[col] = df_sdf[col]

#print(df_strain)

df_strain.to_parquet(input_parque)

cmd = [
    'strain-relief',
    'seed=123',
    'experiment=mace',           # ✅ Fixed: added =
    f'conformers.numConfs=20',    # ✅ f-string for number
    f'io.input.parquet_path={input_parque}',
    f'io.output.parquet_path={output_parque}',
    'hydra.verbose=true'
]

#cmd = [
#    'strain-relief',
#    'experiment=mace',           # ✅ Fixed: added =
#    f'conformers.numConfs=50',    # ✅ f-string for number
#    'minimisation@global_min=mmff94s',
#    'minimisation@local_min=mmff94s',
#    'energy_eval=mace',
#    'model=mace',
#    f'io.input.parquet_path={input_parque}',
#    f'io.output.parquet_path={output_parque}',
#    'hydra.verbose=true'
#]


print(f"Running StrainRelief on {input_parque}")
result = subprocess.run(cmd, capture_output=True, text=True)

df_lig = pd.read_parquet(output_parque)

all_cols = [x for x in df_strain.columns]

df_f_strain = pd.merge(df_strain[all_cols],df_lig[['ID','formal_charge', 'spin_multiplicity',
     'global_min_e',
       'ligand_strain', 'passes_strain_filter', 'nconfs_converged']],on='ID')


#df_f_strain = pd.merge(df_strain[['ID', 'meeko','mol_bytes']],df_lig[['ID','formal_charge', 'spin_multiplicity',
#     'global_min_e',
#       'ligand_strain', 'passes_strain_filter', 'nconfs_converged']],on='ID')

df_f_strain['ROMol'] = df_f_strain['mol_bytes'].apply(lambda x: Chem.Mol(x))

PandasTools.WriteSDF(
    df_f_strain, 
    sdf_output, 
    molColName='ROMol',
    properties=[x for x in df_f_strain.columns if x not in ['mol_bytes','meeko']], 
    idName='ID'
)

df_f_strain.to_csv(output_csv,index=False)
