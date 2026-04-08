#!/usr/bin/env bash

set -euo pipefail

# Prefer conda‑forge when listed; otherwise use pip.
# Note: run from inside the activated env, e.g.
#   conda activate openadmet-models
#   bash install_openadmet_deps.sh

echo "Installing OpenADMET-related dependencies..."

# Click, intake, loguru, pydantic, phx‑class‑registry, uncertainty_toolbox, catboost, lightgbm, wandb, MDAnalysis, splito, useful_rdkit_utils, zarr
python -m pip install \
    click \
    intake \
    loguru \
    pydantic \
    phx-class-registry \
    uncertainty-toolbox \
    catboost \
    lightgbm \
    wandb \
    MDAnalysis \
    splito \
    "useful-rdkit-utils" \
    zarr \
    xgboost

# Email‑specific extras for pydantic
python -m pip install "pydantic[email]"

# mtenn from OpenADMET
python -m pip install "git+https://github.com/OpenADMET/mtenn.git@main"

# tabpfn and tabpfn‑extensions (all submodules)
python -m pip install tabpfn
python -m pip install "tabpfn-extensions[all]@git+https://github.com/PriorLabs/tabpfn-extensions.git"

# molfeat (conda‑forge, but you can use pip if you prefer)
# Uncomment one of these:
# mamba install -c conda-forge molfeat  # conda
python -m pip install molfeat           # or pip

echo "Done installing OpenADMET dependencies."
echo "Run 'openadmet anvil -h' again to verify."
