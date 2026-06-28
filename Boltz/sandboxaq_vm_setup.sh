#!/usr/bin/env bash
# =============================================================================
# sandboxaq VM setup script
# Sets up the conda environment for AQAffinity (OpenFold3 + SandboxAQ affinity head)
#
# Requirements:
#   - NVIDIA GPU with CUDA 12.x driver (check: nvidia-smi)
#   - Miniconda or Anaconda installed
#   - ~30 GB free disk space (packages + model weights)
#   - HuggingFace account with access to SandboxAQ/aqaffinity model
#     (request access at: https://huggingface.co/SandboxAQ/aqaffinity)
# =============================================================================

set -e

echo "=== Step 1: Create conda environment ==="
conda env create -f sandboxaq_environment.yml

echo "=== Step 2: Activate environment ==="
# Note: 'conda activate' doesn't work in scripts; use 'conda run' or source first
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sandboxaq

echo "=== Step 3: Download OpenFold3 model weights ==="
# Weights are downloaded automatically on first run via huggingface-hub.
# To pre-download them manually:
mkdir -p ~/.openfold3

python - <<'EOF'
from huggingface_hub import hf_hub_download
import shutil, os

cache = os.path.expanduser("~/.openfold3")

# OpenFold3 base model (fine-tuned v1)
print("Downloading of3_ft3_v1.pt ...")
path = hf_hub_download(
    repo_id="OpenFold/openfold3",
    filename="of3_ft3_v1.pt",
    local_dir=cache,
)
print(f"  -> {path}")
EOF

echo ""
echo "=== Step 4: Download AQAffinity binding-head weights ==="
python - <<'EOF'
from huggingface_hub import hf_hub_download
import os

cache = os.path.expanduser("~/.openfold3")

print("Downloading AQAffinity model_weights_only.pt ...")
path = hf_hub_download(
    repo_id="SandboxAQ/aqaffinity",
    filename="model_weights/model_weights_only.pt",
    local_dir=cache,
)
print(f"  -> {path}")
EOF

echo ""
echo "=== Step 5: Verify installation ==="
conda run -n sandboxaq python - <<'EOF'
import torch
print(f"PyTorch      : {torch.__version__}")
print(f"CUDA avail   : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU          : {torch.cuda.get_device_name(0)}")

import openfold3
print(f"OpenFold3    : {openfold3.__version__}")

import aqaffinity
print(f"AQAffinity   : OK")

import kalign
print(f"kalign2      : OK")
EOF

echo ""
echo "=== Setup complete ==="
echo ""
echo "To run PXR predictions:"
echo "  conda activate sandboxaq"
echo "  cd <your_working_dir>"
echo "  python run_aqaffinity_PXR.py input_ligands_Batch_1.csv --out_dir pxr_aqaffinity_results"
echo ""
echo "Model weight paths expected by the run script:"
echo "  OF3 checkpoint   : ~/.openfold3/of3_ft3_v1.pt"
echo "  Affinity weights : ~/.openfold3/model_weights/model_weights_only.pt"
echo ""
echo "NOTE: Update OF3_CKPT and AFFINITY_CKPT paths in run_aqaffinity_PXR.py"
echo "      if you place the weights in a different location."
