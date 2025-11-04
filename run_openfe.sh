#!/bin/sh
# Automatic run of openfe
# 1. sdf file with ligands docked
# 2. PDB file with protein
# 3. edges file which the user wants

source ~/.bashrc
conda activate /home/spal/miniforge3/envs/openfe_env

mkdir -p results

job_count=0
max_jobs=2   # Submit 2 SLURM jobs max (each job runs 4 parallel processes using 2 GPUs)

# Group input JSON files in batches of 4
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
#SBATCH --gres=gpu:2
#SBATCH --output=results/batch_$((i/4)).out
#SBATCH --error=results/batch_$((i/4)).err

source ~/.bashrc
conda activate openfe_env

" > "$jobpath"

  # Launch up to 4 processes with CUDA_VISIBLE_DEVICES set per GPU
  for j in "${!batch[@]}"; do
    gpu_id=$(( j / 2 ))   # 2 processes per GPU: IDs 0,0,1,1
    input_file="${batch[$j]}"
    relpath=${input_file#network_setup/transformations/}
    dirpath=${relpath%.json}
    outdir="results/$dirpath"

    echo "CUDA_VISIBLE_DEVICES=$gpu_id openfe quickrun $input_file -o $outdir -d $outdir &" >> "$jobpath"
  done

  echo "wait" >> "$jobpath"

  sbatch "$jobpath"
  job_count=$((job_count + 1))

  # Limit the number of concurrent SLURM jobs
  if [[ $job_count -ge max_jobs ]]; then
    echo "Waiting for $max_jobs jobs to finish..."
    while [[ $(squeue -u $USER -r | grep -c openfe_env) -ge $max_jobs ]]; do
      sleep 10
    done
  fi
done

