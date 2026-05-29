#!/bin/bash
set -e

echo "=== [1/6] System packages ==="
sudo apt update
# Headers must arrive first so any pending NVIDIA dpkg config can link .ko files
sudo apt-get install -y linux-headers-azure
# Fix any packages left in broken state from a previous partial run
sudo apt-get -f install -y
sudo apt-get install -y build-essential git docker.io

sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker "$USER"

echo "=== [2/6] NVIDIA driver (Azure kernel) ==="
NVIDIA_DRIVER_VERSION=$(sudo apt-cache search 'linux-modules-nvidia-[0-9]+-azure$' \
    | awk '{print $1}' | sort | tail -n 1 | awk -F"-" '{print $4}')
echo "Detected driver version: ${NVIDIA_DRIVER_VERSION}"
sudo apt install -y \
    linux-headers-azure \
    linux-modules-nvidia-${NVIDIA_DRIVER_VERSION}-azure \
    nvidia-driver-${NVIDIA_DRIVER_VERSION}

echo "=== [3/6] NVIDIA Container Toolkit ==="
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

# Use the generic stable/deb URL — distribution-specific URLs return 404 on Ubuntu 24.04
curl -sL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

echo "=== [4/6] Miniforge (includes mamba) ==="
curl -LO https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p "$HOME/miniforge3"
rm Miniforge3-Linux-x86_64.sh

. "$HOME/miniforge3/etc/profile.d/conda.sh"
conda init bash
conda activate base

echo "=== [5/6] Permissions ==="
chmod g+rx /home/spal

echo "=== [6/6] Setup complete — rebooting to load NVIDIA kernel module ==="
sudo reboot
