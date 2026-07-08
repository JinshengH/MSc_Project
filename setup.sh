#!/bin/bash
# One-time Aire login-node setup.
# Run after syncing the project to Aire:
#   cd ~/msc_project && bash setup.sh
set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "Do not run setup.sh inside a Slurm job."
    echo "Run it once on the login node before submitting train.slurm."
    exit 1
fi

if [[ -z "${SCRATCH:-}" ]]; then
    echo "SCRATCH is not set. Run this on Aire, not on your local Mac."
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$SCRIPT_DIR}"
ENV_NAME="${ENV_NAME:-msc}"
MODULE_NAME="${MODULE_NAME:-miniforge}"
SCRATCH_PROJECT="${SCRATCH_PROJECT:-$SCRATCH/msc_project}"
HF_CACHE_DIR="${HF_HOME:-$SCRATCH/hf_cache}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_CACHE_DIR/datasets}"
DATASET_NAME="${DATASET_NAME:-sedrickkeh/snli-ve}"
DATA_DIR="$SCRATCH_PROJECT/data/snli-ve"

cd "$PROJECT_DIR"

if [[ ! -f "$PROJECT_DIR/environment.yml" ]]; then
    echo "Missing environment.yml in $PROJECT_DIR"
    exit 1
fi

echo "Project dir: $PROJECT_DIR"
echo "Scratch project dir: $SCRATCH_PROJECT"
echo "HF cache dir: $HF_CACHE_DIR"
echo "HF datasets cache dir: $HF_DATASETS_CACHE"
echo "Dataset: $DATASET_NAME"

mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$SCRATCH_PROJECT/data" "$SCRATCH_PROJECT/results" \
    "$HF_CACHE_DIR" "$HF_DATASETS_CACHE"

if [[ -f "$HOME/.bashrc" ]] && ! grep -q 'HF_HOME=.*hf_cache' "$HOME/.bashrc"; then
    {
        echo ""
        echo "# MSc project Hugging Face cache"
        echo 'export HF_HOME="$SCRATCH/hf_cache"'
    } >> "$HOME/.bashrc"
    echo "Added HF_HOME to $HOME/.bashrc"
fi

export HF_HOME="$HF_CACHE_DIR"
export HF_DATASETS_CACHE

ENV_FILE="$PROJECT_DIR/.env"
if [[ ! -f "$ENV_FILE" ]]; then
    printf 'HF_TOKEN=\n' > "$ENV_FILE"
    chmod 600 "$ENV_FILE"
    echo "Created $ENV_FILE placeholder. Fill HF_TOKEN there if needed."
else
    chmod 600 "$ENV_FILE"
    echo "Using existing $ENV_FILE"
fi

if ! command -v module >/dev/null 2>&1; then
    echo "The module command is not available. Are you on an Aire login node?"
    exit 1
fi

module load "$MODULE_NAME"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda is not available after module load $MODULE_NAME"
    exit 1
fi

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Updating conda environment: $ENV_NAME"
    conda env update -n "$ENV_NAME" -f "$PROJECT_DIR/environment.yml" --prune
else
    echo "Creating conda environment from environment.yml"
    conda env create -f "$PROJECT_DIR/environment.yml"
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
# Some conda activation hooks read optional variables such as MKL_INTERFACE_LAYER.
# Keep nounset off only while conda initializes and activates the env.
set +u
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
set -u

python -m py_compile \
    src/train.py \
    src/configs/base.py \
    src/configs/camc_seed42.py \
    src/utils/env.py \
    src/utils/paths.py

if [[ -f "$DATA_DIR/dataset_dict.json" ]]; then
    echo "Found SNLI-VE dataset: $DATA_DIR"
else
    echo "SNLI-VE dataset not found at $DATA_DIR"
    echo "Downloading $DATASET_NAME and saving to scratch..."
    DATASET_NAME="$DATASET_NAME" DATA_DIR="$DATA_DIR" \
        PROJECT_DIR="$PROJECT_DIR" python - <<'PY'
import os

from datasets import load_dataset
from dotenv import load_dotenv

project_dir = os.environ["PROJECT_DIR"]
dataset_name = os.environ["DATASET_NAME"]
data_dir = os.environ["DATA_DIR"]

load_dotenv(os.path.join(project_dir, ".env"))
token = os.getenv("HF_TOKEN") or None

kwargs = {"cache_dir": os.environ.get("HF_DATASETS_CACHE")}
if token:
    kwargs["token"] = token

ds = load_dataset(dataset_name, **kwargs)
ds.save_to_disk(data_dir)
print(f"Saved {dataset_name} to {data_dir}")
PY
fi

echo "Setup complete."
echo "Next:"
echo "  mkdir -p logs"
echo "  sbatch train.slurm"
