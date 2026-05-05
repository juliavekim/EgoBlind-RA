#!/bin/bash
# ============================================================
# 00_setup_env.sh
# Run ONCE interactively (not sbatch) on Engaging.
# Sets up the conda env and installs LLaMA-Factory.
#
# Usage (from a login or interactive node):
#   bash ~/egoblind-ra/scripts/00_setup_env.sh
# ============================================================

set -e

PROJECT_DIR=/home/julia225/egoblind-ra
CONDA_BASE=/home/software/anaconda3/2023.07
ENV_NAME=egoblind
LLAMA_DIR=/home/julia225/LLaMA-Factory

echo "=== Initialising conda ==="
source "$CONDA_BASE/etc/profile.d/conda.sh"

if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Env '$ENV_NAME' already exists — skipping creation"
else
    conda create -y -n "$ENV_NAME" python=3.11
fi
conda activate "$ENV_NAME"

echo ""
echo "=== Installing PyTorch (CUDA 12.1) ==="
pip install torch==2.5.1 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "=== Cloning LLaMA-Factory ==="
if [ ! -d "$LLAMA_DIR" ]; then
    git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git "$LLAMA_DIR"
fi
cd "$LLAMA_DIR"
pip install -e ".[torch,metrics]"

echo ""
echo "=== Installing other dependencies ==="
pip install \
    transformers==4.51.3 \
    peft \
    accelerate \
    bitsandbytes \
    opencv-python \
    openai

echo ""
echo "=== Installing flash-attn (takes a few minutes) ==="
pip install flash-attn --no-build-isolation

echo ""
echo "=== Creating project directories ==="
mkdir -p "$PROJECT_DIR"/{data,output,logs}

echo ""
echo "=== Done ==="
echo "Next: huggingface-cli login   (need HF token for Kimi-VL)"
echo "Then: sbatch ~/egoblind-ra/scripts/01_prep_data.sh"
