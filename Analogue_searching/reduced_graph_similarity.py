import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdReducedGraphs
from scipy.spatial.distance import cosine
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

# Load reference molecules from SDF
sdf_file = 'mdm2_crystal_structures.sdf'
ref_supp = Chem.SDMolSupplier(sdf_file)
ref_mols = [mol for mol in ref_supp if mol is not None]

# Create Reduced Graph fingerprints and names for reference molecules
ref_fps = []
ref_names = []
for mol in ref_mols:
    try:
        rg_fp = rdReducedGraphs.GetErGFingerprint(mol)
        ref_fps.append(rg_fp)
        name = mol.GetProp('_Name') if mol.HasProp('_Name') else "Unknown"
        ref_names.append(name)
    except Exception as e:
        print(f"Skipping reference molecule due to error: {e}")

# Load query molecules from CSV
csv_file = 'Murcko_Clustered_And_Averaged.csv'
df = pd.read_csv(csv_file)

# Function to compute similarity for a single query molecule row
def compute_max_similarity(row):
    smiles = row['SMILES']
    idx = row.name  # preserve index for result insertion
    query_mol = Chem.MolFromSmiles(smiles)
    if query_mol is None:
        print(f"Invalid SMILES at row {idx}: {smiles}")
        return idx, None, None

    try:
        query_fp = rdReducedGraphs.GetErGFingerprint(query_mol)
    except Exception as e:
        print(f"Skipping SMILES {smiles} at row {idx} due to error in RG generation: {e}")
        return idx, None, None

    similarities = []
    for ref_fp in ref_fps:
        try:
            sim = 1 - cosine(query_fp, ref_fp)
            if np.isnan(sim):
                sim = 0.0
        except:
            sim = 0.0
        similarities.append(sim)

    if similarities:
        max_idx = np.argmax(similarities)
        return idx, similarities[max_idx], ref_names[max_idx]
    else:
        return idx, None, None

# Prepare list of rows for parallel processing
rows = [row for _, row in df.iterrows()]

# Use ProcessPoolExecutor for parallel processing
results = []
with ProcessPoolExecutor() as executor:
    futures = [executor.submit(compute_max_similarity, row) for row in rows]
    for future in as_completed(futures):
        results.append(future.result())

# Put results back into dataframe
for idx, max_sim, best_name in results:
    df.at[idx, 'Max_Similarity'] = max_sim
    df.at[idx, 'Best_Match_Name'] = best_name

# Save to CSV
output_file = 'reduced_graph_similarity_results.csv'
df.to_csv(output_file, index=False)
print(f"Results with all original properties saved to {output_file}")
