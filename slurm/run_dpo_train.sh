#!/bin/bash
#SBATCH --job-name=dpo_train
#SBATCH --partition=mit_normal_gpu
#SBATCH -G 1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/julia225/egoblind-ra/logs/dpo_train_%j.out
#SBATCH --error=/home/julia225/egoblind-ra/logs/dpo_train_%j.err

set -e

source ~/.bashrc
conda activate egoblind

export WANDB_DISABLED=true
export DISABLE_VERSION_CHECK=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=$HOME/hf_cache

cd /orcd/home/002/julia225/LLaMA-Factory

echo "================================================================"
echo "DPO training start: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Job: $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "================================================================"

python src/train.py ~/egoblind-ra/configs/dpo_vision.yaml

echo "================================================================"
echo "DPO training done: $(date)"
echo "================================================================"
