#!/bin/bash
set -e

git clone https://github.com/MolecularAI/DockStream.git
cd DockStream
conda env create -f environment.yml
cd ..

echo "Installation completed!"
echo "To use REINVENT4: conda activate $ENV_NAME"
echo "AutoDock Vina installed at $HOME/bin/vina (run 'vina' after restarting shell or 'source ~/.bashrc')"
echo "To use DockStream: conda activate dockstream"

