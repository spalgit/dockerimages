#!/bin/bash
#SBATCH --job-name=qsartuna_optimize
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --partition=LocalQ # Adjust to your partition (cpu, gpu, etc.)
#SBATCH --output=qsartuna_optimize_%j.out
#SBATCH --error=qsartuna_optimize_%j.err

# Load modules if needed (adjust for your cluster)
# module purge
# module load miniconda3  # or python/3.10, etc.

# Activate conda environment (adjust name/path)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qsartuna  # Replace with your env name, e.g., cheminfo or qsartuna-env

# Create output directory
mkdir -p Output_dir
chmod 755 Output_dir

# Change to working directory (optional)
#cd ${SLURM_SUBMIT_DIR}

# Run QSARtuna optimize
qsartuna-optimize \
  --config regression.json \
  --best-buildconfig-outpath Output_dir/best_config.json \
  --best-model-outpath Output_dir/best_model.pkl \
  --merged-model-outpath Output_dir/merged.pkl

# Optional: Print completion
echo "QSARtuna optimization completed at $(date)"
echo "Best config: Output_dir/best_config.json"
echo "Best model: Output_dir/best_model.pkl"
echo "Merged model: Output_dir/merged.pkl"
