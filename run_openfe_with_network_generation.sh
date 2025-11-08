#!/bin/sh
# Automatic run of openfe
# ONLY USE IT WITH MACHINES WITH 2 GPUs and slurm installed

#source ~/.bashrc
conda activate /home/spal/miniforge3/envs/openfe_env

# Read input arguments
pdb_file="$1"
sdf_file="$2"
edge="$3"

if [ -z "$pdb_file" ] || [ -z "$sdf_file" ] || [ -z "$edge" ]; then
  echo "Usage: $0 <pdb_file> <sdf_file> <edge_file>"
  exit 1
fi

echo "Using PDB file: $pdb_file"
echo "Using SDF file: $sdf_file"
echo "Using edge file: $edge"

edge_file=${PWD}/$edge


cat <<EOF >| create_network.py
#!/usr/bin/env python
import os
import glob
import sys
import openfe
from rdkit import Chem
from openfe.protocols.openmm_rfe import RelativeHybridTopologyProtocol
from kartograf import KartografAtomMapper
import pathlib

supp = Chem.SDMolSupplier('$sdf_file', removeHs=False)
ligands = [openfe.SmallMoleculeComponent.from_rdkit(mol) for mol in supp]

mapper = KartografAtomMapper(atom_map_hydrogens=True)
scorer = openfe.lomap_scorers.default_lomap_score
network_planner = openfe.ligand_network_planning.generate_maximal_network

ligand_network = network_planner(
        ligands=ligands,
        mappers=[mapper],
        scorer=scorer
        )

solvent = openfe.SolventComponent()
protein = openfe.ProteinComponent.from_pdb_file('$pdb_file')

transformations = []

for mapping in ligand_network.edges:
    for leg in ['solvent', 'complex']:
        sysA_dict = {'ligand': mapping.componentA,
                      'solvent': solvent}
        sysB_dict = {'ligand': mapping.componentB,
                      'solvent': solvent}

        if leg == 'complex':
            sysA_dict['protein'] = protein
            sysB_dict['protein'] = protein

        sysA = openfe.ChemicalSystem(sysA_dict, name=f"{mapping.componentA.name}_{leg}")
        sysB = openfe.ChemicalSystem(sysB_dict, name=f"{mapping.componentB.name}_{leg}")

        prefix = "easy_rbfe_"

        settings = RelativeHybridTopologyProtocol.default_settings()
        settings.protocol_repeats = 1
        protocol = RelativeHybridTopologyProtocol(settings)

        transformation = openfe.Transformation(
                stateA=sysA,
                stateB=sysB,
                mapping={'ligand': mapping},
                protocol = protocol,
                name=f"{prefix}{sysA.name}_{sysB.name}"
        )

        transformations.append(transformation)

network = openfe.AlchemicalNetwork(transformations)

transformation_dir = pathlib.Path("network_setup_exhaustive")
transformation_dir.mkdir(exist_ok=True)

for transformation in network.edges:
    transformation.dump(transformation_dir / f"{transformation.name}.json")

EOF

python create_network.py

mkdir -p network_setup/transformations

cat <<EOF >| copy_files.py
#!/usr/bin/env python
import os
import glob

with open("$edge_file") as file:

    for line in file:
        cmd = "cp "+"../../network_setup_exhaustive/"+"easy_rbfe_"+line.split()[2]+"_complex_"+line.split()[4]+"_complex"+".json ."
        os.system(cmd)
        cmd = "cp "+"../../network_setup_exhaustive/"+"easy_rbfe_"+line.split()[2]+"_solvent_"+line.split()[4]+"_solvent"+".json ."
        os.system(cmd)
        cmd = "cp "+"../../network_setup_exhaustive/"+"easy_rbfe_"+line.split()[4]+"_complex_"+line.split()[2]+"_complex"+".json ."
        os.system(cmd)
        cmd = "cp "+"../../network_setup_exhaustive/"+"easy_rbfe_"+line.split()[4]+"_solvent_"+line.split()[2]+"_solvent"+".json ."
        os.system(cmd)
EOF

cd network_setup/transformations
python3 ../../copy_files.py
cd ../../

mkdir -p results

job_count=0
max_jobs=2

files=($(ls network_setup/transformations/*.json))

for ((i=0; i<${#files[@]}; i+=4)); do
  batch=("${files[@]:i:4}")
  jobpath="network_setup/transformations/batch_$((i/4)).job"

  echo -e "#!/usr/bin/env bash
#SBATCH --job-name=openfe_batch_$((i/4))
#SBATCH -N 1
#SBATCH --partition=LocalQ
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=results/batch_$((i/4)).out
#SBATCH --error=results/batch_$((i/4)).err

source ~/.bashrc
conda activate /home/spal/miniforge3/envs/openfe_env
" > "$jobpath"

  for j in "${!batch[@]}"; do
    gpu_id=$(( j / 2 ))
    input_file="${batch[$j]}"
    relpath=${input_file#network_setup/transformations/}
    dirpath=${relpath%.json}
    outdir="results/$dirpath"

    echo "CUDA_VISIBLE_DEVICES=$gpu_id openfe quickrun $input_file -o $relpath -d $outdir &" >> "$jobpath"
  done

  echo "wait" >> "$jobpath"

  sbatch "$jobpath"
  job_count=$((job_count + 1))

  if [[ $job_count -ge max_jobs ]]; then
    echo "Waiting for $max_jobs jobs to finish..."
    while [[ $(squeue -u $USER -r | grep -c openfe_env) -ge $max_jobs ]]; do
      sleep 10
    done
  fi
done

