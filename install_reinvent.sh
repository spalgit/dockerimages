#!/bin/bash
set -e

ENV_NAME="reinvent4_env"

# Load conda function
source $(conda info --base)/etc/profile.d/conda.sh

# Create and activate REINVENT4 conda environment with Python 3.10
conda create -n $ENV_NAME python=3.10 -y
conda activate $ENV_NAME

# Clone and install REINVENT4
git clone https://github.com/MolecularAI/REINVENT4.git
cd REINVENT4

# Install REINVENT4 for CPU (modify if GPU CUDA/ROCm available)
python install.py all
cd ..

# Clone and install DockStream using its environment.yml
git clone https://github.com/MolecularAI/DockStream.git
cd DockStream
conda env create -f environment.yml
cd ..

echo "Installation completed!"
echo "To use REINVENT4: conda activate $ENV_NAME"
echo "AutoDock Vina installed at $HOME/bin/vina (run 'vina' after restarting shell or 'source ~/.bashrc')"
echo "To use DockStream: conda activate dockstream"

