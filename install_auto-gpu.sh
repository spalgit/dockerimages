#!/bin/bash

# Update and install prerequisites
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential gcc g++ make cmake git wget curl

sudo apt install linux-headers-$(uname -r)

wget https://download.nvidia.com/XFree86/Linux-x86_64/550.54.14/NVIDIA-Linux-x86_64-550.54.14.run

sudo ./NVIDIA-Linux-x86_64-550.54.14.run

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-4

echo "export PATH=/usr/local/cuda-12.4/bin${PATH:+:\${PATH}}" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64\${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" >> ~/.bashrc

# Export CUDA environment variables for current session & bashrc
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Verify CUDA installation
#nvcc --version

# Install AutoDock-GPU dependencies
sudo apt install -y ocl-icd-opencl-dev ocl-icd-libopencl1 opencl-headers

# Clone AutoDock-GPU source
cd ~
if [ ! -d "AutoDock-GPU" ]; then
  git clone https://github.com/ccsb-scripps/AutoDock-GPU.git
fi
cd AutoDock-GPU

# Build AutoDock-GPU (default make will use CUDA if available)
make DEVICE=CUDA NUMWI=128

# Verify AutoDock-GPU binary
ls -l bin/

# Install Miniconda if not installed
if ! command -v conda &> /dev/null; then
  echo "Miniconda not found, installing..."
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
  bash ~/miniconda.sh -b -p $HOME/miniconda
  eval "$($HOME/miniconda/bin/conda shell.bash hook)"
  conda init
fi

# Create or activate environment for Meeko
conda create -n meeko_env python=3.9 -y
conda activate meeko_env

# Install Meeko from conda-forge channel
conda install -c conda-forge meeko -y

echo "Installation complete. Remember to source ~/.bashrc or re-login to refresh environment variables."

