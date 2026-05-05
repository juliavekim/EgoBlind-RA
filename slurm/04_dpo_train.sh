#!/bin/bash
#SBATCH --job-name=egoblind_dpo_train
#SBATCH --partition=pi_dbertsim
#SBATCH --account=mit_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00
#SBATCH --output=/home/julia225/egoblind-ra/logs/dpo_train_%j.out
#SBATCH --error=/home/julia225/egoblind-ra/logs/dpo_train_%j.err
# ============================================================
# 04_dpo_train.sh
# DPO training on top of SFT checkpoint.
# Prerequisite: 03_dpo_gen.sh complete.
# Runtime: ~1-2 hrs on A100.
#
# Submit: sbatch ~/egoblind-ra/scripts/04_dpo_train.sh
# ============================================================

set -e

PROJECT_DIR=/home/julia225/egoblind-ra
LLAMA_DIR=/home/julia225/LLaMA-Factory

source /home/software/anaconda3/2023.07/etc/profile.d/conda.sh
conda activate egoblind

echo "=== EgoBlind DPO Training ==="
echo "Job:     $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "Started: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# dpo_unified.yaml points adapter_name_or_path at checkpoint-final.
# If that doesn't exist, patch the yaml on the fly with the latest checkpoint.
SFT_CKPT=$PROJECT_DIR/output/sft_unified/checkpoint-final
if [ ! -d "$SFT_CKPT" ]; then
    SFT_CKPT=$(ls -d $PROJECT_DIR/output/sft_unified/checkpoint-* 2>/dev/null | sort -V | tail -1)
    echo "Patching adapter path → $SFT_CKPT"
    sed -i "s|adapter_name_or_path:.*|adapter_name_or_path: $SFT_CKPT|" \
        $PROJECT_DIR/configs/dpo_unified.yaml
fi

cd $LLAMA_DIR
llamafactory-cli train $PROJECT_DIR/configs/dpo_unified.yaml

echo ""
echo "=== DPO training complete: $(date) ==="
ls -lh $PROJECT_DIR/output/dpo_unified/
