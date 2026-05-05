#!/bin/bash
#SBATCH --job-name=egoblind_eval
#SBATCH --partition=pi_dbertsim
#SBATCH --account=mit_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --output=/home/julia225/egoblind-ra/logs/eval_%j.out
#SBATCH --error=/home/julia225/egoblind-ra/logs/eval_%j.err
# ============================================================
# 05_eval.sh
# Inference + scoring on the test set with the final DPO adapter.
# Uses ground-truth urgency labels (--use_gt_urgency) so results
# measure model quality, not classifier quality.
# Prerequisite: 04_dpo_train.sh complete.
# Runtime: ~30-60 min on A100.
#
# Submit: sbatch ~/egoblind-ra/scripts/05_eval.sh
# ============================================================

set -e

PROJECT_DIR=/home/julia225/egoblind-ra

source /home/software/anaconda3/2023.07/etc/profile.d/conda.sh
conda activate egoblind

echo "=== EgoBlind Evaluation ==="
echo "Job:     $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "Started: $(date)"

DPO_CKPT=$PROJECT_DIR/output/dpo_unified/checkpoint-final
if [ ! -d "$DPO_CKPT" ]; then
    DPO_CKPT=$(ls -d $PROJECT_DIR/output/dpo_unified/checkpoint-* 2>/dev/null | sort -V | tail -1)
fi
echo "DPO adapter: $DPO_CKPT"

python $PROJECT_DIR/scripts/inference.py \
    --base_model   moonshotai/Kimi-VL-A3B-Instruct \
    --adapter_path "$DPO_CKPT" \
    --test_data    $PROJECT_DIR/data/egoblind_test.json \
    --output_path  $PROJECT_DIR/output/predictions_unified.json \
    --best_of_k    5 \
    --use_gt_urgency

echo ""
echo "=== Eval complete: $(date) ==="
