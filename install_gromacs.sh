#!/bin/sh

sudo apt update
wget ftp://ftp.gromacs.org/gromacs/gromacs-2024.3.tar.gz
tar xfz gromacs-2024.3.tar.gz
cd gromacs-2024.3
mkdir build
cd build
sudo apt  install cmake
sudo apt update
sudo apt-get install mpich
sudo apt update
cmake .. -DGMX_BUILD_OWN_FFTW=ON -DREGRESSIONTEST_DOWNLOAD=ON -DGMX_MPI=on -DGMX_GPU=CUDA
make
make check
sudo make install
echo "export PATH=/usr/local/gromacs/bin/GMXRC:\$PATH" >> ~/.bashrc
echo "export PATH=/usr/local/gromacs/bin/:\$PATH" >> ~/.bashrc


source ~/.bashrc

