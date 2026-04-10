#!/bin/bash
#
#Prepare ligand file

rosetta_lig_dir="ligs_prepped"
#rosetta_lig_dir="Unique_Designs_from_Rosetta_OOP"
ligand_output_file="PDB_from_Rosetta_output.sdf"
scrubbed_file="scrubbed.sdf"
mdm2_Ala_file="mdm2-ligand-ALANIZED.sdf"
tethered_file="tethered.sdf"
untethered_file="untethered.sdf"
prepared_lig="ligs_prepped"
PDB_DIR="AD_Proteins"
prep_PDB_DIR="Proteins_prepared"
docking_output="Ache_Dockings_3"


mkdir $docking_output

for fld_file in "$prep_PDB_DIR"/*.fld; do
    echo "Processing: $fld_file"
    resname="${fld_file::-4}"
#    ~/AutoDock-GPU/bin/autodock_gpu_64wi  --filelist $prepared_lig --ffile $fld_file --resnam $resname
#    ~/AutoDock-GPU/bin/autodock_gpu_64wi  --filelist $prepared_lig --ffile $fld_file &> autodock.log&
    ~/AutoDock-GPU/bin/autodock_gpu_64wi  --filelist $prepared_lig --ffile $fld_file
    echo "Finished: $fld_file"
done


for dlg_file in "$prepared_lig"/*.dlg; do
	sdfile="${dlg_file::-4}".sdf
	mk_export.py $dlg_file -s $sdfile --all_dlg_poses 
done

mv $prepared_lig/*.sdf $docking_output

