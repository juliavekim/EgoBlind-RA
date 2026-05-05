#!/bin/bash
#SBATCH --job-name=egoblind_dpo_gen
#SBATCH --partition=pi_dbertsim
#SBATCH --account=mit_general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=/home/julia225/egoblind-ra/logs/dpo_gen_%j.out
#SBATCH --error=/home/julia225/egoblind-ra/logs/dpo_gen_%j.err
# ============================================================
# 03_dpo_gen.sh
# Generate DPO preference pairs from the SFT checkpoint.
# Prerequisite: 02_sft.sh complete.
# Runtime: ~1-2 hrs on A100.
#
# Submit: sbatch ~/egoblind-ra/scripts/03_dpo_gen.sh
# ============================================================

set -e

PROJECT_DIR=/home/julia225/egoblind-ra

source /home/software/anaconda3/2023.07/etc/profile.d/conda.sh
conda activate egoblind

echo "=== EgoBlind DPO Pair Generation ==="
echo "Job:     $SLURM_JOB_ID on $SLURMD_NODENAME"
echo "Started: $(date)"

# Find the latest SFT checkpoint
SFT_CKPT=$PROJECT_DIR/output/sft_unified/checkpoint-final
if [ ! -d "$SFT_CKPT" ]; then
    SFT_CKPT=$(ls -d $PROJECT_DIR/output/sft_unified/checkpoint-* 2>/dev/null | sort -V | tail -1)
fi
echo "SFT checkpoint: $SFT_CKPT"

python $PROJECT_DIR/scripts/generate_dpo_pairs.py \
    --model_path     moonshotai/Kimi-VL-A3B-Instruct \
    --adapter_path   "$SFT_CKPT" \
    --data_path      $PROJECT_DIR/data/egoblind_urgent_for_dpo.json \
    --output_path    $PROJECT_DIR/data/egoblind_unified_dpo.json \
    --num_candidates 8 \
    --alpha 0.4 --beta 0.3 --gamma 0.3

echo ""
echo "=== DPO gen complete: $(date) ==="
python -c "
import json
with open('$PROJECT_DIR/data/egoblind_unified_dpo.json') as f:
    d = json.load(f)
print(f'{len(d)} DPO pairs generated')
"
