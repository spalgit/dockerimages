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


input_directory = 'Temp_dir'
output_file = f'{input_directory}/combined_docking_Reference_actives.sdf'
output_mol2 = f'{input_directory}/combined_docking_Reference_actives.mol2'
input_parque = f'{input_directory}/combined_docking_Reference_actives.parque'
output_parque = f'{input_directory}/combined_docking_Reference_actives_output.parque'




def concatenate_sdf_files(input_dir, output_sdf):
    # Create a writer object for the output SDF
    writer = Chem.SDWriter(output_sdf)
    output_filename = os.path.basename(output_sdf)

    # Loop through all files in the input directory, sorted by name
    for filename in sorted(os.listdir(input_dir)):
        if filename.endswith('.sdf') and filename != output_filename:
            file_path = os.path.join(input_dir, filename)
            suppl = Chem.SDMolSupplier(file_path,removeHs=False)
            if suppl is None:
                print(f"Warning: could not read {file_path}")
                continue
            for mol in suppl:
                if mol is not None:
                    writer.write(mol)

    writer.close()

# Example usage:
concatenate_sdf_files(input_directory, output_file)

# Calcualte Ligand strain energies

df_sdf = PandasTools.LoadSDF(output_file,
                        smilesName='SMILES',      # Add SMILES column
                        molColName='ROMol',    # Rename mol column
                        embedProps=True,
                            removeHs=False)


#with initialize(version_base="1.1", config_path="/home/spal/StrainRelief/src/strain_relief/hydra_config"):
#    cfg = compose(
#        config_name="default", 
#        overrides=["experiment=mmff94s",]
#    )

with initialize_config_dir(
    version_base="1.1",
    config_dir="/home/spal/StrainRelief/src/strain_relief/hydra_config",
):
    cfg = compose(config_name="default")


df_strain = pd.DataFrame([{"mol_bytes": mol.ToBinary(), **mol.GetPropsAsDict()} for mol in df_sdf['ROMol']])
df_strain = df_strain.reset_index(drop=False, names='id')
df_strain['ID'] = df_sdf['ID']

df_strain.to_parquet(input_parque)

cmd = [
    'strain-relief',
    f'io.input.parquet_path={input_parque}',
    f'io.output.parquet_path={output_parque}',
    'experiment=mmff94s'
]

print(f"Running StrainRelief on {input_parque}")
result = subprocess.run(cmd, capture_output=True, text=True)

df_lig = pd.read_parquet(output_parque)

df_f_strain = pd.merge(df_strain[['id', 'meeko']],df_lig[['id','formal_charge', 'spin_multiplicity',
     'global_min_e',
       'ligand_strain', 'passes_strain_filter', 'nconfs_converged']],on='id')

# Extract properties to DataFrame
SDF = Chem.SDMolSupplier(output_file,removeHs=False)
mols = [x for x in SDF if x is not None]
ids = [x.GetProp("_Name") for x in mols]
json_list = [x.GetProp("meeko") for x in mols]
data = [json.loads(item) for item in json_list]
df = pd.DataFrame(data)
df['ID'] = ids
df = df.reset_index(drop=False, names='id')
df = pd.merge(df,df_f_strain,on=['id'])


print(df.shape[0])
print(df.columns)


# Build proper pivot table
df_pivot = pd.pivot_table(
    df, index='ID',
    values=['free_energy', 'intermolecular_energy', 'internal_energy','ligand_strain'],
    aggfunc=['mean', 'std']
).reset_index()

# Flatten multi-index columns
df_pivot.columns = [
    'ID',
    'mean_free_energy', 'mean_intermolecular_energy', 'mean_internal_energy','mean_ligand_strain',
    'std_free_energy', 'std_intermolecular_energy', 'std_internal_energy','std_ligand_strain'
]
df_pivot.sort_values(by=['mean_free_energy']).to_csv(
    f'{input_directory}/Docking_scores.csv', index=False
)

# Prepare mol2 output with headers
mols_pybel = list(pybel.readfile("sdf", output_file))

with open(output_mol2, "w") as out:
    for i, mol in enumerate(mols_pybel, start=1):
        if i-1 >= len(df):
            continue  # Prevent index error if there are more molecules than DataFrame rows
        row = df.iloc[i-1]
        mean_intermolecular_energy = row["intermolecular_energy"]
        mean_free_energy = row["free_energy"]
        mean_internal_energy = row["internal_energy"]
        mean_ligand_strain = row["ligand_strain"]
        id_ = row['ID']

        comment_block = f"""########## Number      : {i}
########## Name        : {id_}
########## Description : Docking_Pose
########## Reflect     : 0
########## Energy score                             :     {mean_intermolecular_energy}
##########   intermolecular van der Waals                 {mean_free_energy}
##########   intermolecular electrostatic                 {mean_internal_energy}
##########   ligand strain                                {mean_ligand_strain}

"""

        out.write(comment_block)
        mol_block = mol.write("mol2")
        out.write(mol_block)
        out.write("\n\n")

print("Written all molecules with header to output.mol2")

