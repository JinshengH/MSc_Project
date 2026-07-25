#!/bin/bash
#SBATCH --job-name=late_fusion_s44     # Job name (override: sbatch -J <name>)
#SBATCH --time=03:00:00            # 5 epochs; encoders dominate runtime (~camc pace)
#SBATCH --partition=gpu            # Request GPU partition
#SBATCH --gres=gpu:1               # Request 1 GPU
#SBATCH --cpus-per-task=8          # Request 8 CPU cores
#SBATCH --mem-per-cpu=8G           # Request 8GB memory per CPU core
#SBATCH --output=logs/%x_%j.out    # stdout log
#SBATCH --error=logs/%x_%j.err     # stderr log

# Load GPU environment (create the env once on the login node first:
# conda env create -f environment.yml)
set -e
module load miniforge
conda activate msc

# Keep HF cache on scratch. HF_TOKEN is loaded from ~/msc_project/.env
# by src/utils/env.py (copy .env there once; it is gitignored).
export HF_HOME="$SCRATCH/hf_cache"

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
echo "Flash storage path: $TMP_SHARED"
nvidia-smi

# Copy input data to flash storage before training starts
# (target name must match staged_data_name in the experiment config)
cp -r "$SCRATCH/msc_project/data/snli-ve" "$TMP_SHARED/snli-ve"

# This experiment config is also the Python entrypoint.
cd "$HOME/msc_project"
python -u -m src.configs.late_fusion_seed44

echo "Finished $(date)"

# Flash storage ($TMP_SHARED) is automatically cleaned after the job ends
