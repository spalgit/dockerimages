"""
Add pEC50, pEC50_counter, pEC50_std.error to merged SDF:
  - Base molecules (OADMET IDs): all three from f1_plus_htchem_train_4827.sdf
  - ChEMBL molecules: pEC50 only from ChEMBL_PXR_confirmed_in_ChEMBL.csv
Output: Chembl_pxr_ligprepped_cleaned_merged_pEC50.sdf
"""
import pandas as pd
from rdkit import Chem

DIR = "/home/spal/dockerimages/QSARTuna/PXR/June_19"

IN_SDF     = f"{DIR}/Chembl_pxr_ligprepped_cleaned_merged.sdf"
TRAIN_SDF  = f"{DIR}/f1_plus_htchem_train_4827.sdf"
CHEMBL_CSV = "/home/spal/dockerimages/QSARTuna/PXR_For_QSARTuna_Modelling/ChEMBL_PXR_confirmed_in_ChEMBL.csv"
OUT_SDF    = f"{DIR}/Chembl_pxr_ligprepped_cleaned_merged_pEC50.sdf"

# ── Lookup from training SDF ──────────────────────────────────────────────────
print("Building lookup from f1_plus_htchem_train_4827.sdf ...")
train_lookup = {}
for mol in Chem.SDMolSupplier(TRAIN_SDF, removeHs=False):
    if mol is None:
        continue
    name = mol.GetProp("_Name")
    train_lookup[name] = {
        "pEC50":           mol.GetProp("pEC50")           if mol.HasProp("pEC50")           else None,
        "pEC50_counter":   mol.GetProp("pEC50_counter")   if mol.HasProp("pEC50_counter")   else None,
        "pEC50_std.error": mol.GetProp("pEC50_std.error") if mol.HasProp("pEC50_std.error") else None,
        "source":          mol.GetProp("source")          if mol.HasProp("source")          else None,
    }
print(f"  {len(train_lookup)} IDs loaded")

# ── Lookup from ChEMBL CSV ────────────────────────────────────────────────────
chembl_df = pd.read_csv(CHEMBL_CSV)
chembl_pec50  = dict(zip(chembl_df["ID"], chembl_df["pEC50"].astype(str)))
chembl_source = dict(zip(chembl_df["ID"], chembl_df["source"]))
print(f"  {len(chembl_pec50)} ChEMBL IDs loaded")

# ── Write output SDF ──────────────────────────────────────────────────────────
writer = Chem.SDWriter(OUT_SDF)
stats = {"base": 0, "chembl": 0, "missing": 0, "errors": 0}

for mol in Chem.SDMolSupplier(IN_SDF, removeHs=False):
    if mol is None:
        stats["errors"] += 1
        continue
    name = mol.GetProp("_Name")

    if name in train_lookup:
        p = train_lookup[name]
        if p["pEC50"]           is not None: mol.SetProp("pEC50",           p["pEC50"])
        if p["pEC50_counter"]   is not None: mol.SetProp("pEC50_counter",   p["pEC50_counter"])
        if p["pEC50_std.error"] is not None: mol.SetProp("pEC50_std.error", p["pEC50_std.error"])
        if p["source"]          is not None: mol.SetProp("source",          p["source"])
        stats["base"] += 1

    elif name in chembl_pec50:
        mol.SetProp("pEC50",  chembl_pec50[name])
        mol.SetProp("source", chembl_source.get(name, "ChEMBL"))
        stats["chembl"] += 1

    else:
        stats["missing"] += 1
        print(f"  WARNING: no pEC50 for {name}")

    writer.write(mol)

writer.close()

print(f"\nBase molecules (pEC50 + counter + std.error) : {stats['base']}")
print(f"ChEMBL molecules (pEC50 only)                : {stats['chembl']}")
print(f"Missing                                       : {stats['missing']}")
print(f"Output: {OUT_SDF}")
