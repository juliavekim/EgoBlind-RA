#!/bin/bash
#SBATCH --job-name=dpo_f1
#SBATCH --partition=mit_normal_gpu
#SBATCH -G 1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/home/julia225/egoblind-ra/logs/dpo_f1_%j.out
#SBATCH --error=/home/julia225/egoblind-ra/logs/dpo_f1_%j.err

# Usage: sbatch run_dpo_f1.sh sft|baseline

set -e
source ~/.bashrc
conda activate egoblind

export WANDB_DISABLED=true
export DISABLE_VERSION_CHECK=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=$HOME/hf_cache

INIT=${1:-sft}

if [ "$INIT" = "sft" ]; then
    YAML=$HOME/egoblind-ra/configs/dpo_f1_sft.yaml
elif [ "$INIT" = "baseline" ]; then
    YAML=$HOME/egoblind-ra/configs/dpo_f1_baseline.yaml
else
    echo "Usage: $0 [sft|baseline]"; exit 1
fi

cd /orcd/home/002/julia225/LLaMA-Factory

echo "DPO F1 training start ($INIT init): $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Job: $SLURM_JOB_ID on $SLURMD_NODENAME"

python src/train.py $YAML

echo "DPO F1 training done: $(date)"
