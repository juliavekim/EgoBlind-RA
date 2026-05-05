#!/bin/bash
#SBATCH --job-name=dpo_eval
#SBATCH --partition=mit_normal_gpu
#SBATCH -G 1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/home/julia225/egoblind-ra/logs/dpo_eval_%j.out
#SBATCH --error=/home/julia225/egoblind-ra/logs/dpo_eval_%j.err

set -e
source ~/.bashrc
conda activate egoblind

export WANDB_DISABLED=true
export DISABLE_VERSION_CHECK=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=$HOME/hf_cache

echo "DPO eval start: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

python ~/egoblind-ra/scripts/run_dpo_eval.py

echo "DPO eval done: $(date)"
