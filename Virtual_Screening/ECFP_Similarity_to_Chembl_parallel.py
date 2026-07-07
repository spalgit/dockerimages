import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import TanimotoSimilarity
from concurrent.futures import ProcessPoolExecutor
import os
import glob
import multiprocessing

# Load reference file from csv
df = pd.read_csv("Find_Close_Analogues_for_inhouse_comps.csv")
ref_mols = [Chem.MolFromSmiles(x) for x in df['smiles']]
ids = [x for x in df['ID']]

# Get ECFP4 (Morgan) fingerprints for the reference molecules
ref_fps = []
for mol in ref_mols:
    try:
        ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        ref_fps.append(ecfp)
    except Exception as e:
        print(f"Skipping reference molecule due to error: {e}")

df_ref = pd.DataFrame(list(zip([x for x in df['smiles']], ids, ref_fps)), columns=['smiles', 'ID', 'FPS'])

# === Function to process one SDF file and write results ===
def process_sdf_file(sdf_path):
    sdf_name = os.path.splitext(os.path.basename(sdf_path))[0]
    print(f"Processing: {sdf_name}")

    supplier = Chem.SDMolSupplier(sdf_path)
    mols = [mol for mol in supplier if mol is not None]
    print(f"Total mols = {len(mols)}, sdf_name = {sdf_name}")

    output_dir = "ECFP_Results_4"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{sdf_name}_processed.csv")

    smiles_enamine = []
    id_enamine = []
    chembl_smi = []
    chembl_id = []
    similarity_ = []

    for enamine_mol in mols:
        try:
            query_fp = AllChem.GetMorganFingerprintAsBitVect(enamine_mol, radius=2, nBits=2048)
        except Exception as e:
            print(f"Skipping molecule due to fingerprint error: {e}")
            continue

        similarities = []
        for ref_fp in ref_fps:
            sim = TanimotoSimilarity(query_fp, ref_fp)
            similarities.append(sim)

        max_idx = int(similarities.index(max(similarities)))
        similarity_max = similarities[max_idx]

        if similarity_max >= 0.6:
            try:
                smiles_enamine.append(Chem.MolToSmiles(enamine_mol))
 #               id_enamine.append(enamine_mol.GetProp("catalog_id"))
                id_enamine.append(enamine_mol.GetProp("catalog_id"))
                chembl_smi.append(df_ref.iloc[max_idx]['smiles'])
                chembl_id.append(df_ref.iloc[max_idx]['ID'])
                similarity_.append(similarity_max)
            except Exception as e:
                print(f"Error extracting properties: {e}")
                continue

    df_out = pd.DataFrame(list(zip(smiles_enamine, id_enamine, chembl_smi, chembl_id, similarity_)),
                          columns=['SMILES_Enamine', 'ID_Enamine', 'ChemBL_Smiles', 'ChemBl_ID', 'Similarity'])
    df_out.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")

# === Run all SDF files concurrently using limited processors ===
if __name__ == "__main__":
#    sdf_folder = "/home/spal/Enamine_Screening_collection/Scrub_4/"
    sdf_folder = "/home/spal/Enamine_Screening_collection/Scrub_4"
    sdf_files = sorted(glob.glob(os.path.join(sdf_folder, "scrubbed_*.sdf")))

    num_cpus = 2
    print(f"Using {num_cpus} CPU cores.")

    with ProcessPoolExecutor(max_workers=num_cpus) as executor:
        executor.map(process_sdf_file, sdf_files)

    print("All files processed.")

