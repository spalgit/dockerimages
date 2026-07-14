#!/usr/bin/env bash
# End-to-end Boltz-2 finetuning pipeline on internal ITK structures.
# Implements every step in FINETUNING.md in this directory — read that file for the
# verified reasoning behind each choice below (checkpoint hyperparameter mismatches,
# the DateFilter metadata gotcha, the --clusters flag name, etc).
#
# Usage:
#   ./run_finetune_pipeline.sh                 # full pipeline + debug run + real training
#   ./run_finetune_pipeline.sh --prep-only      # data prep only, skip training
#   ./run_finetune_pipeline.sh --skip-prep      # regenerate config + train only, assumes prep done
#
set -euo pipefail

# ============================== configuration ================================
CONDA_ENV="${CONDA_ENV:-boltz2}"
BOLTZ_REPO="${BOLTZ_REPO:-$HOME/boltz}"
PDB_SRC="${PDB_SRC:-$HOME/Recludix/ITK/PDB_structure}"
PRETRAINED_CKPT="${PRETRAINED_CKPT:-$HOME/.boltz/boltz2_conf.ckpt}"
PIPELINE_DIR="${PIPELINE_DIR:-$HOME/Recludix/ITK/Boltz_retrain/pipeline_work}"
MSA_SERVER="${MSA_SERVER:-https://api.colabfold.com}"
REDIS_PORT="${REDIS_PORT:-7777}"

# Training knobs — tune for your GPU / how long you want to run.
TRAINER_DEVICES="${TRAINER_DEVICES:-1}"
ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-4}"
MAX_STEPS="${MAX_STEPS:-500}"
MAX_TOKENS="${MAX_TOKENS:-512}"     # drop to 256/384 (and MAX_ATOMS to 2304/3456) on GPU OOM
MAX_ATOMS="${MAX_ATOMS:-4608}"
MAX_LR="${MAX_LR:-0.0002}"

RUN_PREP=1
RUN_TRAIN=1
for arg in "$@"; do
  case "$arg" in
    --prep-only) RUN_TRAIN=0 ;;
    --skip-prep) RUN_PREP=0 ;;
    *) echo "Unknown argument: $arg" >&2; exit 1 ;;
  esac
done

# Derived paths
CIF_DIR="$PIPELINE_DIR/cif"
RESOURCES_DIR="$PIPELINE_DIR/resources"
CLUSTER_DIR="$PIPELINE_DIR/clustering"
RAW_MSA_DIR="$PIPELINE_DIR/raw_msa"
MSA_OUT_DIR="$PIPELINE_DIR/msa_processed"
TARGET_OUT_DIR="$PIPELINE_DIR/targets"
CONFIG_DIR="$PIPELINE_DIR/config"
TRAIN_OUTPUT_DIR="$PIPELINE_DIR/output"
SEQ_FASTA="$PIPELINE_DIR/chain_sequences.fasta"
FINETUNE_YAML="$CONFIG_DIR/finetune_internal.yaml"

log() { echo -e "\n=== $* ===\n"; }

# taxonomy.rdb is ~7.3GB on disk (~29GB once loaded into memory), so redis
# answers PING well before it has finished loading the dataset — commands
# like GET fail with BusyLoadingError until loading is actually complete.
# Poll INFO persistence's `loading` flag instead of just PING.
wait_redis_ready() {
  local port="$1"
  until redis-cli -p "$port" ping >/dev/null 2>&1; do sleep 0.5; done
  while true; do
    loading=$(redis-cli -p "$port" info persistence 2>/dev/null | tr -d '\r' | awk -F: '/^loading:/{print $2}')
    [ "$loading" = "0" ] && break
    sleep 2
  done
}

mkdir -p "$PIPELINE_DIR" "$CIF_DIR" "$RESOURCES_DIR" "$CLUSTER_DIR" "$RAW_MSA_DIR" \
         "$MSA_OUT_DIR" "$TARGET_OUT_DIR" "$CONFIG_DIR" "$TRAIN_OUTPUT_DIR"

# ============================== environment setup =============================
log "Activating conda env '$CONDA_ENV'"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

if ! python -c "import boltz" 2>/dev/null; then
  log "boltz not importable in '$CONDA_ENV' — installing editable from $BOLTZ_REPO"
  (cd "$BOLTZ_REPO" && pip install -e .)
fi
pip install -q -r "$BOLTZ_REPO/scripts/process/requirements.txt"

if ! command -v mmseqs >/dev/null 2>&1; then
  log "mmseqs not on PATH — installing into '$CONDA_ENV' via conda-forge/bioconda"
  conda install -n "$CONDA_ENV" -c conda-forge -c bioconda mmseqs2 -y
fi
MMSEQS_BIN="$(command -v mmseqs)"

if ! command -v redis-server >/dev/null 2>&1; then
  log "redis-server not on PATH — installing into '$CONDA_ENV' via conda-forge"
  conda install -n "$CONDA_ENV" -c conda-forge redis-server -y
fi

if [ "$RUN_PREP" -eq 1 ]; then

# ============================== 1. PDB -> mmCIF ================================
log "Step 1/7: converting legacy PDB files to mmCIF"
python - "$PDB_SRC" "$CIF_DIR" <<'PYEOF'
import sys, pathlib
from Bio.PDB import PDBParser, MMCIFIO

src, dst = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)
pdb_files = sorted(src.glob("*.pdb"))
print(f"Found {len(pdb_files)} .pdb files in {src}")
for f in pdb_files:
    out = dst / f.with_suffix(".cif").name
    if out.exists():
        continue
    s = PDBParser(QUIET=True).get_structure(f.stem, f)
    io = MMCIFIO()
    io.set_structure(s)
    io.save(str(out))
    print(f"  {f.name} -> {out.name}")
PYEOF

# ============================== 2. Extract chain sequences ======================
log "Step 2/7: extracting per-chain polymer sequences to FASTA"
python - "$CIF_DIR" "$SEQ_FASTA" <<'PYEOF'
import sys, pathlib
from Bio.PDB import MMCIFParser
from Bio.PDB.Polypeptide import PPBuilder

cif_dir, out_fasta = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
parser = MMCIFParser(QUIET=True)
builder = PPBuilder()
records = []
for f in sorted(cif_dir.glob("*.cif")):
    structure = parser.get_structure(f.stem, f)
    for model in structure:
        for chain in model:
            seq = "".join(str(pp.get_sequence()) for pp in builder.build_peptides(chain))
            if len(seq) >= 4:
                records.append((f"{f.stem}_{chain.id}", seq))
        break  # first model only
with out_fasta.open("w") as fh:
    for name, seq in records:
        fh.write(f">{name}\n{seq}\n")
print(f"Wrote {len(records)} chain sequences to {out_fasta}")
PYEOF

# ============================== 3. CCD dictionary ================================
log "Step 3/7: fetching CCD dictionary + symmetry file"
cd "$RESOURCES_DIR"
[ -f ccd.pkl ]      || wget -q https://boltz1.s3.us-east-2.amazonaws.com/ccd.pkl
[ -f ccd.rdb ]      || wget -q https://boltz1.s3.us-east-2.amazonaws.com/ccd.rdb
[ -f taxonomy.rdb ] || wget -q https://boltz1.s3.us-east-2.amazonaws.com/taxonomy.rdb
[ -f symmetry.pkl ] || wget -q https://boltz1.s3.us-east-2.amazonaws.com/symmetry.pkl

# ============================== 4. Cluster sequences =============================
log "Step 4/7: clustering chain sequences"
if [ ! -f "$CLUSTER_DIR/clustering.json" ]; then
  python "$BOLTZ_REPO/scripts/process/cluster.py" \
      --ccd "$RESOURCES_DIR/ccd.pkl" \
      --sequences "$SEQ_FASTA" \
      --mmseqs "$MMSEQS_BIN" \
      --outdir "$CLUSTER_DIR"
else
  echo "clustering.json already exists, skipping"
fi

# ============================== 5. Generate MSAs =================================
log "Step 5/7: generating MSAs via $MSA_SERVER"
python - "$SEQ_FASTA" "$RAW_MSA_DIR" "$MSA_SERVER" <<'PYEOF'
import hashlib, pathlib, sys
from Bio import SeqIO
from boltz.data.msa.mmseqs2 import run_mmseqs2

seq_fasta, raw_msa_dir, host_url = sys.argv[1], pathlib.Path(sys.argv[2]), sys.argv[3]
raw_msa_dir.mkdir(parents=True, exist_ok=True)

def hash_sequence(seq: str) -> str:
    return hashlib.sha256(seq.encode()).hexdigest()

seqs = sorted({str(r.seq).strip() for r in SeqIO.parse(seq_fasta, "fasta")})
todo = [s for s in seqs if not (raw_msa_dir / f"{hash_sequence(s)}.a3m").exists()]
print(f"{len(seqs)} unique sequences, {len(todo)} need MSAs")
if todo:
    # NB: run_mmseqs2 downloads the raw ColabFold tarball into
    # "<prefix>_<mode>" (e.g. "..._env") and leaves the extracted
    # uniref.a3m / bfd...a3m files there, still containing literal
    # NUL bytes it only strips from its *in-memory* return value.
    # That scratch dir must live outside raw_msa_dir, otherwise
    # msa.py's rglob("*.a3m*") in step 6 picks up those raw files
    # and parse_a3m() dies with KeyError: '\x00'.
    scratch_dir = raw_msa_dir.parent / "msa_scratch"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    a3m_lines = run_mmseqs2(
        todo, str(scratch_dir / "mmseqs_tmp"),
        use_env=True, use_pairing=False, host_url=host_url,
    )
    for seq, a3m in zip(todo, a3m_lines):
        (raw_msa_dir / f"{hash_sequence(seq)}.a3m").write_text(a3m)
        print(f"  wrote {hash_sequence(seq)}.a3m ({len(a3m.splitlines())} lines)")
PYEOF

# ============================== 6. Process MSAs (needs redis) ====================
log "Step 6/7: processing MSAs (redis + taxonomy.rdb on port $REDIS_PORT)"
redis-server --dir "$RESOURCES_DIR" --dbfilename taxonomy.rdb --port "$REDIS_PORT" --daemonize no &
REDIS_PID=$!
wait_redis_ready "$REDIS_PORT"
python "$BOLTZ_REPO/scripts/process/msa.py" \
    --msadir "$RAW_MSA_DIR" --outdir "$MSA_OUT_DIR" --redis-port "$REDIS_PORT" \
    --num-processes 1
redis-cli -p "$REDIS_PORT" shutdown nosave >/dev/null 2>&1 || true
wait "$REDIS_PID" 2>/dev/null || true

# ============================== 7. Process structures (needs redis) ==============
log "Step 7/7: processing structures (redis + ccd.rdb on port $REDIS_PORT)"
redis-server --dir "$RESOURCES_DIR" --dbfilename ccd.rdb --port "$REDIS_PORT" --daemonize no &
REDIS_PID=$!
wait_redis_ready "$REDIS_PORT"
python "$BOLTZ_REPO/scripts/process/rcsb.py" \
    --datadir "$CIF_DIR" \
    --clusters "$CLUSTER_DIR/clustering.json" \
    --outdir "$TARGET_OUT_DIR" \
    --use-assembly --redis-port "$REDIS_PORT" \
    --num-processes 1
redis-cli -p "$REDIS_PORT" shutdown nosave >/dev/null 2>&1 || true
wait "$REDIS_PID" 2>/dev/null || true

fi  # RUN_PREP

# ============================== write finetune config =============================
log "Writing finetune config to $FINETUNE_YAML"
cat > "$FINETUNE_YAML" <<YAMLEOF
trainer:
  accelerator: gpu
  devices: $TRAINER_DEVICES
  precision: 32
  gradient_clip_val: 10.0
  max_epochs: -1
  max_steps: $MAX_STEPS
  accumulate_grad_batches: $ACCUMULATE_GRAD_BATCHES

output: $TRAIN_OUTPUT_DIR
pretrained: $PRETRAINED_CKPT
resume: null
disable_checkpoint: false
matmul_precision: null
save_top_k: 3

data:
  datasets:
    - _target_: boltz.data.module.training.DatasetConfig
      target_dir: $TARGET_OUT_DIR
      msa_dir: $MSA_OUT_DIR
      prob: 1.0
      sampler:
        _target_: boltz.data.sample.cluster.ClusterSampler
      cropper:
        _target_: boltz.data.crop.boltz.BoltzCropper
        min_neighborhood: 0
        max_neighborhood: 40
      split: null

  filters: []   # deliberately empty — structure.yaml's DateFilter would reject all internal
                # structures, which have no deposition/release metadata (see FINETUNING.md caveat b)
  tokenizer:
    _target_: boltz.data.tokenize.boltz.BoltzTokenizer
  featurizer:
    _target_: boltz.data.feature.featurizer.BoltzFeaturizer

  symmetries: $RESOURCES_DIR/symmetry.pkl
  max_tokens: $MAX_TOKENS
  max_atoms: $MAX_ATOMS
  max_seqs: 2048
  pad_to_max_tokens: true
  pad_to_max_atoms: true
  pad_to_max_seqs: true
  samples_per_epoch: 100000
  batch_size: 1
  num_workers: 4
  random_seed: 42
  pin_memory: true
  atoms_per_window_queries: 32
  min_dist: 2.0
  max_dist: 22.0
  num_bins: 64

model:
  _target_: boltz.model.models.boltz2.Boltz2
  atom_s: 128
  atom_z: 16
  token_s: 384
  token_z: 128
  num_bins: 64
  atom_feature_dim: 388
  atoms_per_window_queries: 32
  atoms_per_window_keys: 128

  embedder_args:
    atom_encoder_depth: 3
    atom_encoder_heads: 4
    add_mol_type_feat: true
    add_method_conditioning: true
    add_modified_flag: true
    add_cyclic_flag: true

  msa_args:
    msa_s: 64
    msa_blocks: 4
    msa_dropout: 0.15
    z_dropout: 0.25
    pairwise_head_width: 32
    pairwise_num_heads: 4
    activation_checkpointing: true
    use_paired_feature: true

  pairformer_args:
    num_blocks: 64
    num_heads: 16
    dropout: 0.25
    pairwise_head_width: 32
    pairwise_num_heads: 4
    post_layer_norm: false
    activation_checkpointing: true

  score_model_args:
    sigma_data: 16
    dim_fourier: 256
    atom_encoder_depth: 3
    atom_encoder_heads: 4
    token_transformer_depth: 24
    token_transformer_heads: 16
    atom_decoder_depth: 3
    atom_decoder_heads: 4
    conditioning_transition_layers: 2
    transformer_post_ln: false
    activation_checkpointing: true

  diffusion_process_args:
    sigma_min: 0.0004
    sigma_max: 160.0
    sigma_data: 16.0
    rho: 7
    P_mean: -1.2
    P_std: 1.5
    gamma_0: 0.8
    gamma_min: 1.0
    noise_scale: 1.0
    step_scale: 1.0
    coordinate_augmentation: true
    alignment_reverse_diff: true
    synchronize_sigmas: false

  diffusion_loss_args:
    add_smooth_lddt_loss: true
    nucleotide_loss_weight: 5.0
    ligand_loss_weight: 10.0
    filter_by_plddt: 0.0

  training_args:
    recycling_steps: 3
    sampling_steps: 20
    diffusion_multiplicity: 16
    diffusion_samples: 2
    confidence_loss_weight: 1e-4
    diffusion_loss_weight: 4.0
    distogram_loss_weight: 3e-2
    adam_beta_1: 0.9
    adam_beta_2: 0.95
    adam_eps: 0.00000001
    lr_scheduler: af3
    base_lr: 0.0
    max_lr: $MAX_LR
    lr_warmup_no_steps: 1000
    lr_start_decay_after_n_steps: 50000
    lr_decay_every_n_steps: 50000
    lr_decay_factor: 0.95

  validation_args:
    recycling_steps: 3
    sampling_steps: 200
    diffusion_samples: 5
    symmetry_correction: true
    run_confidence_sequentially: false

  ema: true
  ema_decay: 0.999
  bond_type_feature: true
  use_no_atom_char: false
  use_atom_backbone_feat: false
  use_residue_feats_atoms: false
  cyclic_pos_enc: true
  fix_sym_check: true

  confidence_prediction: false
  affinity_prediction: false
  use_templates: false
  structure_prediction_training: true
  compile_pairformer: false
  compile_structure: false
  compile_confidence: false
  compile_msa: false
YAMLEOF

if [ "$RUN_TRAIN" -eq 0 ]; then
  log "Data prep complete, --prep-only given, skipping training. Config at $FINETUNE_YAML"
  exit 0
fi

# ============================== debug + real run ==================================
log "Debug run (single process, surfaces config errors fast)"
(cd "$BOLTZ_REPO" && python scripts/train/train.py "$FINETUNE_YAML" debug=1)

log "Launching real training run"
(cd "$BOLTZ_REPO" && python scripts/train/train.py "$FINETUNE_YAML")

log "Done. Checkpoints under $TRAIN_OUTPUT_DIR/*/checkpoints/"
