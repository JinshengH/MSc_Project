#!/bin/bash
#SBATCH --job-name=msc_ai_train
#SBATCH --output=../logs/%x_%j.out
#SBATCH --error=../logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G

module load miniforge
conda activate msc_ai

cd ~/msc_project/code

echo "Job ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"

python -c "import torch; print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"

python train.py \
  --data_dir ~/msc_project/data \
  --output_dir ~/msc_project/outputs \
  --checkpoint_dir ~/msc_project/checkpoints \
  --epochs 20 \
  --batch_size 32