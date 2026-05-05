#!/bin/bash
#SBATCH --job-name=dpo_gen_f1
#SBATCH --partition=mit_normal_gpu
#SBATCH -G 1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/julia225/egoblind-ra/logs/dpo_gen_f1_%j.out
#SBATCH --error=/home/julia225/egoblind-ra/logs/dpo_gen_f1_%j.err

# Usage:
#   sbatch run_dpo_gen_f1.sh baseline    # baseline init
#   sbatch run_dpo_gen_f1.sh sft         # SFT init
#
# Defaults to baseline if no arg passed.

set -e
source ~/.bashrc
conda activate egoblind

export WANDB_DISABLED=true
export DISABLE_VERSION_CHECK=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=$HOME/hf_cache

INIT=${1:-baseline}

if [ "$INIT" = "sft" ]; then
    OUT=$HOME/egoblind-ra/output/dpo_pairs_f1_sft.json
    EXTRA="--sft_adapter $HOME/egoblind-ra/output/sft_vision_v1"
elif [ "$INIT" = "baseline" ]; then
    OUT=$HOME/egoblind-ra/output/dpo_pairs_f1_baseline.json
    EXTRA=""
else
    echo "Usage: $0 [baseline|sft]"; exit 1
fi

echo "================================================================"
echo "F1 pair gen ($INIT init): $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Job: $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "Output: $OUT"
echo "================================================================"

python ~/egoblind-ra/scripts/generate_dpo_pairs_f1.py \
    --train_data   $HOME/egoblind-ra/data/grpo_train.jsonl \
    --frames_root  $HOME/egoblind-ra/data/baseline_frames/train \
    --output       $OUT \
    --n_candidates 4 \
    --temperature  0.9 \
    --save_every   25 \
    $EXTRA

echo "================================================================"
echo "Done: $(date)"
echo "================================================================"
