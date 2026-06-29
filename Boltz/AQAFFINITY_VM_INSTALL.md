# AQAffinity + OpenFold3 VM Installation Notes

**VM:** `boltzadmin@RPI-BOLTZ01`  
**Conda env:** `sandboxaq`  
**Weights directory:** `/mnt/data/sandeep/openfold3_weights/`

---

## Step 1 — Create the conda environment

The environment definition is in `sandboxaq_environment.yml`. Install it at a custom prefix
so everything lives on `/mnt/data/sandeep` rather than the default home:

```bash
conda env create -f sandboxaq_environment.yml \
    --prefix /mnt/data/sandeep/conda/envs/sandboxaq

conda config --append envs_dirs /mnt/data/sandeep/conda/envs
conda activate sandboxaq
```

---

## Step 2 — Download and install AQAffinity

The package is hosted on HuggingFace (`SandboxAQ/aqaffinity`). You must be logged in:

```bash
hf auth login   # or: huggingface-cli login
```

Download and pip-install:

```bash
hf download SandboxAQ/aqaffinity \
    --local-dir /mnt/data/sandeep/aqaffinity

pip install /mnt/data/sandeep/aqaffinity
```

This pulls in `openfold3` automatically as a dependency. Verify versions:

```bash
pip show openfold3 aqaffinity | grep -E "Name|Version"
# Expected: openfold3 0.4.1, aqaffinity 0.0.1
```

---

## Step 3 — Download model weights

The `OpenFold/openfold3` repo is **gated** — you must first visit
`https://huggingface.co/OpenFold/OpenFold3` while logged in and request access.
Test access with:

```bash
python -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download(repo_id='OpenFold/openfold3', filename='README.md', local_dir='/tmp/of3_test')
print('ACCESS OK')
"
```

Once access is confirmed, download both checkpoints:

```bash
python - <<'EOF'
from huggingface_hub import hf_hub_download

weights_dir = "/mnt/data/sandeep/openfold3_weights"

# OpenFold3 base model (note: now lives under checkpoints/ in the HF repo)
hf_hub_download(
    repo_id="OpenFold/openfold3",
    filename="checkpoints/of3_ft3_v1.pt",
    local_dir=weights_dir,
)

# AQAffinity binding-head weights
hf_hub_download(
    repo_id="SandboxAQ/aqaffinity",
    filename="model_weights/model_weights_only.pt",
    local_dir=weights_dir,
)
print("Done.")
EOF
```

Final weight locations on disk:
```
/mnt/data/sandeep/openfold3_weights/checkpoints/of3_ft3_v1.pt
/mnt/data/sandeep/openfold3_weights/model_weights/model_weights_only.pt
```

---

## Step 4 — Copy the prediction script

Copy `run_aqaffinity_PXR.py` from the WSL machine to the VM working directory:

```bash
# Run on WSL
scp ~/dockerimages/Boltz/run_aqaffinity_PXR.py \
    boltzadmin@<VM_IP>:~/Boltz/Sandbox/
```

The script resolves checkpoint paths automatically via `_find_path()`, checking
the new `checkpoints/` layout first, then the old flat layout, then the local WSL path.

---

## Step 5 — Run predictions

```bash
conda activate sandboxaq
cd ~/Boltz/Sandbox

python run_aqaffinity_PXR.py batch1.csv --out_dir PXR_Sandbox_Batch1
python run_aqaffinity_PXR.py batch2.csv --out_dir PXR_Sandbox_Batch2
```

Results are written to `<out_dir>/pxr_aqaffinity_predictions.csv`.
Completed compounds are skipped automatically on re-run.

---

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `Path '...of3_ft3_v1.pt' does not exist` | Weights not downloaded | Run Step 3 |
| `KeyError: 'experiment_settings'` | aqaffinity predict called without `--runner_yaml` | Script now writes `runner.yaml` automatically per compound |
| `size mismatch for fourier_emb.w` / `layer_norm_z.weight` | Checkpoint from openfold3 0.3.1 used with 0.4.1 code | Download fresh weights per Step 3 (repo updated for 0.4.1) |
| `403 Forbidden` on download | HuggingFace access not granted | Visit HF page and request access; verify with the README.md test above |
| `huggingface-cli: command not found` | CLI not on PATH | Use `hf auth login` or `python -m huggingface_hub.commands.huggingface_cli login` |
