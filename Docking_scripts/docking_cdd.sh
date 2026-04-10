#!/bin/bash
#
#Prepare ligand file

#rosetta_lig_dir="temp_pdb_dirs"
rosetta_lig_dir="Unique_Designs_from_Rosetta_OOP"
ligand_output_file="PDB_from_Rosetta_output.sdf"
scrubbed_file="scrubbed.sdf"
#mdm2_Ala_file="Lig.sdf"
mdm2_Ala_file="BTK_Non_Cov_New_ref_from_Cov.sdf"
tethered_file="tethered.sdf"
untethered_file="untethered.sdf"
prepared_lig="ligs_prepped"
PDB_DIR="AD_Prots_temp"
prep_PDB_DIR="Proteins_prepared"
#sdf_file_for_dock="Unique_OOPs_And_Sabine_designs.sdf"
sdf_file_for_dock="Structures.sdf"


# Convert Rosetta OOP PDB files to sdf fiels

#python3 convert_pdb_to_sdf.py $rosetta_lig_dir $ligand_output_file

# Protonate the OOps

#scrub.py $ligand_output_file -o $scrubbed_file --skip_tautomers --ph_low 7.0 --ph_high 7.4 --name_from_prop "_Name" --cpu 20
scrub.py $sdf_file_for_dock -o $scrubbed_file --skip_tautomers --ph_low 8.0 --ph_high 8.4 --name_from_prop "_Name" --cpu 20

# Align the Protonated compounds to Alanized p53

#python3 tetheredMinimization.py  $mdm2_Ala_file $scrubbed_file $tethered_file $untethered_file
#python3 tetheredMinimization.py  $mdm2_Ala_file $sdf_file_for_dock $tethered_file $untethered_file

# Prepare ligands for docking. Rigidify back bone amides.

#mk_prepare_ligand.py -i $sdf_file_for_dock --multimol_outdir ligs_prepped \
#        --name_from_prop "_Name" 


mk_prepare_ligand.py -i $scrubbed_file --multimol_outdir ligs_prepped \
        --name_from_prop "_Name" 


#  Prepare the receptor


mkdir $prep_PDB_DIR
for pdb_file in "$PDB_DIR"/*.pdb; do
    echo "Processing: $pdb_file"
python3 - <<EOF   
from prody import parsePDB, writePDB
import subprocess, os
pdb_token ="$pdb_file"
atoms_from_pdb = parsePDB(pdb_token)
receptor_selection = "chain A and not water and not hetero and not resname AMP and not element H"
receptor_atoms = atoms_from_pdb.select(receptor_selection)
pdbfile=pdb_token[:-4].split("/")[1]
file="$prep_PDB_DIR/"+pdb_token[:-4].split("/")[1]
recfile=pdb_token[:-4].split("/")[1]
prody_receptorPDB = f"{file}_receptor_atoms.pdb"
prody_rec = f"{file}_recceptorH.pdb"

writePDB(prody_receptorPDB, receptor_atoms)

lig_selection = "chain A and hetero and not water"
ligand_atoms = atoms_from_pdb.select(lig_selection)
prody_ligandPDB = f"{file}_ligand_atoms.pdb"
writePDB(prody_ligandPDB, ligand_atoms)

file_rec=f"{file}_rec"

cmd = [ 
 
    "mk_prepare_receptor.py",
    "--read_pdb", prody_receptorPDB,
    "-o", file_rec,
    "-p",
    "-g",
    "--box_enveloping", prody_ligandPDB,
    "--padding", "5"
]

subprocess.run(cmd)

file_rec=f"{recfile}_rec"
file_gpf = f"{file_rec}.gpf"

cmd = ["autogrid4", "-p", file_gpf]

result=subprocess.run(cmd,cwd="$prep_PDB_DIR",capture_output=True,text=True)

EOF
done
