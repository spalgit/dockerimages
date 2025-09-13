#!/bin/sh

sudo apt update
sudo apt-get update && sudo apt-get install -y build-essential
sudo apt install -y git
sudo apt install -y docker.io
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER
sudo apt update
NVIDIA_DRIVER_VERSION=$(sudo apt-cache search 'linux-modules-nvidia-[0-9]+-gcp$' | awk '{print $1}' | sort | tail -n 1 | awk -F"-" '{print $4}')
sudo apt install -y linux-modules-nvidia-${NVIDIA_DRIVER_VERSION}-gcp nvidia-driver-${NVIDIA_DRIVER_VERSION}
sudo modprobe nvidia
sudo systemctl stop google-cloud-ops-agent
NVIDIA_DRIVER_VERSION=$(sudo apt-cache search 'linux-modules-nvidia-[0-9]+-gcp$' | awk '{print $1}' | sort | tail -n 1 | awk -F"-" '{print $4}')
echo $NVIDIA_DRIVER_VERSION
sudo apt update
sudo apt install -y linux-modules-nvidia-${NVIDIA_DRIVER_VERSION}-gcp nvidia-driver-${NVIDIA_DRIVER_VERSION}
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update
sudo apt install -y nvidia-docker2
sudo systemctl restart docker
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/miniconda3/bin/activate
conda init
conda create --name my_pmx python=3.9
conda activate my_pmx
conda install conda-forge::acpype
pip install ipython
conda install conda-forge::rdkit
git clone git@github.com:spalgit/pmx_scripts.git
cd pmx_scripts
pip install .
chmod g+rx /home/spal
sudo reboot

