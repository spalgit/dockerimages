#!/usr/bin/env bash
# =============================================================================
# sandboxaq VM setup script
# Sets up the conda environment for AQAffinity (OpenFold3 + SandboxAQ affinity head)
#
# All data, weights, and the conda environment are installed under /mnt/data/sandeep
#
# Requirements:
#   - NVIDIA GPU with CUDA 12.x driver (check: nvidia-smi)
#   - Miniconda or Anaconda installed
#   - ~30 GB free disk space on /mnt/data/sandeep
#   - HuggingFace account with access to SandboxAQ/aqaffinity
#     (request access at: https://huggingface.co/SandboxAQ/aqaffinity)
#   - Log in before running: hf auth login
# =============================================================================

set -e

BASE=/mnt/data/sandeep
ENV_PREFIX=$BASE/conda/envs/sandboxaq
WEIGHTS_DIR=$BASE/openfold3_weights

echo "=== Paths ==="
echo "  Conda env  : $ENV_PREFIX"
echo "  Weights    : $WEIGHTS_DIR"
echo ""

# ---------------------------------------------------------------------------
echo "=== Step 1: Create conda environment at $ENV_PREFIX ==="
# Using --prefix installs into /mnt/data/sandeep instead of ~/miniconda3/envs/
conda env create \
    -f sandboxaq_environment.yml \
    --prefix "$ENV_PREFIX"

# ---------------------------------------------------------------------------
echo ""
echo "=== Step 2: Configure conda to recognise the prefix-based env ==="
# Add the envs dir so 'conda activate sandboxaq' works by name
conda config --append envs_dirs "$BASE/conda/envs"

# ---------------------------------------------------------------------------
echo ""
echo "=== Step 2b: Download and install aqaffinity (+ openfold3) ==="
# git clone over HTTPS requires token auth; use 'hf download' instead
AQAFFINITY_DIR=$BASE/aqaffinity
hf download SandboxAQ/aqaffinity --local-dir "$AQAFFINITY_DIR"
conda run --prefix "$ENV_PREFIX" pip install "$AQAFFINITY_DIR"

# ---------------------------------------------------------------------------
echo ""
echo "=== Step 3: Download OpenFold3 base model weights ==="
mkdir -p "$WEIGHTS_DIR"

conda run --prefix "$ENV_PREFIX" python - <<EOF
from huggingface_hub import hf_hub_download

weights_dir = "$WEIGHTS_DIR"

print("Downloading of3_ft3_v1.pt  (OpenFold3 base model) ...")
path = hf_hub_download(
    repo_id="OpenFold/openfold3",
    filename="of3_ft3_v1.pt",
    local_dir=weights_dir,
)
print(f"  -> {path}")
EOF

# ---------------------------------------------------------------------------
echo ""
echo "=== Step 4: Download AQAffinity binding-head weights ==="

conda run --prefix "$ENV_PREFIX" python - <<EOF
from huggingface_hub import hf_hub_download

weights_dir = "$WEIGHTS_DIR"

print("Downloading model_weights_only.pt  (AQAffinity binding head) ...")
path = hf_hub_download(
    repo_id="SandboxAQ/aqaffinity",
    filename="model_weights/model_weights_only.pt",
    local_dir=weights_dir,
)
print(f"  -> {path}")
EOF

# ---------------------------------------------------------------------------
echo ""
echo "=== Step 5: Verify installation ==="

conda run --prefix "$ENV_PREFIX" python - <<'EOF'
import torch
print(f"PyTorch      : {torch.__version__}")
print(f"CUDA avail   : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU          : {torch.cuda.get_device_name(0)}")

import openfold3
print(f"OpenFold3    : {openfold3.__version__}")

import aqaffinity
print("AQAffinity   : OK")

from pathlib import Path
import subprocess
result = subprocess.run(["kalign", "--version"], capture_output=True, text=True)
print(f"kalign2      : {result.stdout.strip() or result.stderr.strip()}")
EOF

# ---------------------------------------------------------------------------
echo ""
echo "=== Setup complete ==="
echo ""
echo "Activate the environment:"
echo "  conda activate $ENV_PREFIX"
echo "  # or, after 'conda config --append envs_dirs $BASE/conda/envs':"
echo "  conda activate sandboxaq"
echo ""
echo "Run PXR predictions:"
echo "  cd $BASE"
echo "  python run_aqaffinity_PXR.py input_ligands_Batch_1.csv --out_dir pxr_aqaffinity_results"
echo ""
echo "Weight paths (already configured in run_aqaffinity_PXR.py):"
echo "  OF3 checkpoint   : $WEIGHTS_DIR/of3_ft3_v1.pt"
echo "  Affinity weights : $WEIGHTS_DIR/model_weights/model_weights_only.pt"
