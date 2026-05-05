#!/bin/bash
#SBATCH --job-name=dpo_f1_eval
#SBATCH --partition=mit_normal_gpu
#SBATCH -G 1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/julia225/egoblind-ra/logs/dpo_f1_eval_%j.out
#SBATCH --error=/home/julia225/egoblind-ra/logs/dpo_f1_eval_%j.err

# Usage: sbatch run_dpo_f1_eval.sh sft|baseline

set -e
source ~/.bashrc
conda activate egoblind

export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=$HOME/hf_cache

INIT=${1:-sft}

echo "DPO_F1 eval ($INIT init): $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Job: $SLURM_JOB_ID on $SLURMD_NODENAME"

python ~/egoblind-ra/scripts/run_dpo_f1_eval.py $INIT

echo "Done: $(date)"
