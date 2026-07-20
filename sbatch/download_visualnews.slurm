#!/bin/bash
#SBATCH --job-name=visualnews_download
#SBATCH --time=30:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

URL="https://www.cs.rice.edu/~vo9/visualnews/origin.tar"

mkdir -p "$SCRATCH/msc_project/data"
cd "$SCRATCH/msc_project/data"

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
echo "Downloading $URL"
echo "Destination: $SCRATCH/msc_project/data/origin.tar"

# -c resumes the existing origin.tar after a timeout or connection failure.
# Unlimited retries keep transient network errors from ending the allocation.
wget \
    --continue \
    --retry-connrefused \
    --waitretry=10 \
    --read-timeout=60 \
    --timeout=60 \
    --tries=0 \
    --progress=dot:giga \
    "$URL"

echo "Download complete: $(du -h origin.tar | cut -f1)"
echo "Finished $(date)"
