#!/bin/bash
#SBATCH --job-name=dpo_gen
#SBATCH --partition=mit_normal_gpu
#SBATCH -G 1
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/julia225/egoblind-ra/logs/dpo_gen_%j.out
#SBATCH --error=/home/julia225/egoblind-ra/logs/dpo_gen_%j.err

# Setup
source ~/.bashrc
conda activate egoblind
export HF_HOME=$HOME/hf_cache

echo "=== Job started: $(date) ==="
echo "Node: $(hostname)"
nvidia-smi
echo ""

# Run generation
python ~/egoblind-ra/scripts/generate_dpo_pairs.py \
    --adapter_path ~/egoblind-ra/output/sft_vision_v1 \
    --data_path ~/egoblind-ra/data/sft_best_reference.jsonl \
    --frames_dir ~/egoblind-ra/data/baseline_frames/train \
    --output_path ~/egoblind-ra/output/dpo_pairs_vision.json \
    --num_candidates 4 \
    --min_loss_gap 0.2

echo ""
echo "=== Job finished: $(date) ==="
