import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdReducedGraphs
from rdkit.DataStructs import TanimotoSimilarity
from scipy.spatial.distance import cosine
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import os
import glob
import multiprocessing
from rdkit.DataStructs import TanimotoSimilarity

# Load reference file from csv
df = pd.read_csv("Find_close_analogues_from_CDD_and_Article.csv")
ref_mols = [Chem.MolFromSmiles(x) for x in df['smiles']]
ids = [x for x in df['ID']]

# Get reduced graph fingerprints for the molecules
ref_fps = []
for mol in ref_mols:
    try:
        rg_fp = rdReducedGraphs.GetErGFingerprint(mol)
        ref_fps.append(rg_fp)
    except Exception as e:
        print(f"Skipping reference molecule due to error: {e}")

df_ref = pd.DataFrame(list(zip([x for x in df['smiles']], ids, ref_fps)), columns=['smiles', 'ID', 'FPS'])

# === Function to process one SDF file and write results ===
def process_sdf_file(sdf_path):
    sdf_name = os.path.splitext(os.path.basename(sdf_path))[0]
    print(f"Processing: {sdf_name}")

    supplier = Chem.SDMolSupplier(sdf_path)
 #   mols = [mol for mol in supplier if mol is not None]
    mols = [mol for mol in supplier]
    print("Total mols = {}, sdf_name = {}".format(len(mols),sdf_name))

    output_dir = "CDD_Article_Reduced_Graph_Mcule_1/"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{sdf_name}_processed.csv")

    #For each molecule in Enamine find the similarity with each compound in reference
    # Take the maximum similairty and if it is greater than 0.6 then retain save the Enamine mol in a list
    # Finally we need to output a dataframe for each Supplier.sdf from Enamine containing information about
    # SMILES, Enamine ID and the reference ChemBL ID and Reference ChemBL Smiles and reduced graph similarity.

    smiles_enamine = []
    id_enamine = []
    chembl_smi = []
    chembl_id= []
    similarity_ = []

    for enamine_mol in mols:
        query_fp = rdReducedGraphs.GetErGFingerprint(enamine_mol)
        similarities = []
        for ref_fp in ref_fps:
            sim = 1 - cosine(query_fp, ref_fp)
#            sim = TanimotoSimilarity(query_fp, ref_fp)
#            print(sim)
            if np.isnan(sim):
                sim = 0.0
            similarities.append(sim)
        max_idx = np.argmax(similarities)
        similarity_max = similarities[max_idx]
        if(similarity_max>=0.8):
            smiles_enamine.append(Chem.MolToSmiles(enamine_mol))
   #         id_enamine.append(enamine_mol.GetProp("catalog_id"))
            id_enamine.append(enamine_mol.GetProp("catalog_id"))
            chembl_smi.append(df_ref.iloc[max_idx]['smiles'])
            chembl_id.append(df_ref.iloc[max_idx]['ID'])
            similarity_.append(similarity_max)

    # Placeholder for processing logic
    # Here you would compute fingerprints and similarities and store them
    # For now, just save molecule count
    print(len(id_enamine))
    df_out = pd.DataFrame(list(zip(smiles_enamine,id_enamine,chembl_smi,chembl_id,similarity_)),columns=['SMILES_Enamine','ID_Enamine','ChemBL_Smiles','ChemBl_ID','Similarity'])
#    df_out = pd.DataFrame(list(zip(smiles_enamine,id_enamine,chembl_smi,chembl_id,similarity_)),columns=['SMILES_Mcule','ID_Mcule','ChemBL_Smiles','ChemBl_ID','Similarity'])
    print("Number of hit = {}".format(df_out.shape[0]))
    df_out.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")

# === Run all SDF files concurrently using all processors ===
if __name__ == "__main__":
#    sdf_folder = "/home/spal/Enamine_Screening_collection/Scrub_4/"
    sdf_folder = "/home/spal/Mcule_Screening_collection/Chunk1/"
    sdf_files = sorted(glob.glob(os.path.join(sdf_folder, "scrubbed_*.sdf")))

    num_cpus = multiprocessing.cpu_count()
    num_cpus = 6 
    print(f"Using {num_cpus} CPU cores.")

    print(sdf_files)

    with ProcessPoolExecutor(max_workers=num_cpus) as executor:
#        executor.map(process_sdf_file, sdf_files)
        executor.map(process_sdf_file, sdf_files)

    print("All files processed.")
