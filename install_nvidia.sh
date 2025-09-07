#!/bin/sh

sudo apt update
sudo apt install build-essential
sudo apt install linux-headers-$(uname -r)

wget https://download.nvidia.com/XFree86/Linux-x86_64/550.54.14/NVIDIA-Linux-x86_64-550.54.14.run

sudo ./NVIDIA-Linux-x86_64-550.54.14.run

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-4

echo "export PATH=/usr/local/cuda-12.4/bin${PATH:+:\${PATH}}" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64\${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" >> ~/.bashrc

source ~/.bashrc

