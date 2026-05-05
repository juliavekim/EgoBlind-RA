#!/bin/bash
#SBATCH --job-name=egoblind_sft
#SBATCH --partition=pi_dbertsim
#SBATCH --account=mit_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/home/julia225/egoblind-ra/logs/sft_%j.out
#SBATCH --error=/home/julia225/egoblind-ra/logs/sft_%j.err
# ============================================================
# 02_sft.sh
# LoRA SFT on the combined urgent+not_urgent tagged dataset.
# Prerequisite: 01_prep_data.sh complete.
# Runtime: ~3-4 hrs on A100.
#
# Submit: sbatch ~/egoblind-ra/scripts/02_sft.sh
# ============================================================

set -e

PROJECT_DIR=/home/julia225/egoblind-ra
LLAMA_DIR=/home/julia225/LLaMA-Factory

source /home/software/anaconda3/2023.07/etc/profile.d/conda.sh
conda activate egoblind

echo "=== EgoBlind SFT ==="
echo "Job:     $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "Started: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# LLaMA-Factory reads dataset_info.json from dataset_dir (set in yaml).
# Data is already written there by prep job — nothing to copy.

cd $LLAMA_DIR
llamafactory-cli train $PROJECT_DIR/configs/sft_unified.yaml

echo ""
echo "=== SFT complete: $(date) ==="
ls -lh $PROJECT_DIR/output/sft_unified/
