#!/bin/bash
# ============================================================================
# Install LLaMA-Factory + dependencies on MIT Engaging
# Run once on a login node. Conda env `egoblind` should already exist.
# ============================================================================
set -e

echo "[1/5] Activating egoblind env..."
source ~/.bashrc
conda activate egoblind

echo "[2/5] Cloning LLaMA-Factory..."
mkdir -p ~/egoblind-ra/code
cd ~/egoblind-ra/code
if [ -d LLaMA-Factory ]; then
    echo "  already cloned, pulling latest"
    cd LLaMA-Factory && git pull && cd ..
else
    git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
fi

echo "[3/5] Verifying pinned versions are still installed..."
# These should match what was pinned for SFT — DO NOT let pip resolve fresh deps.
python -c "
import transformers, peft, trl, accelerate, datasets, tokenizers, huggingface_hub
print(f'transformers:    {transformers.__version__}     (expected 4.56.0)')
print(f'tokenizers:      {tokenizers.__version__}     (expected 0.22.0)')
print(f'peft:            {peft.__version__}      (expected 0.13.0)')
print(f'trl:             {trl.__version__}      (expected 0.24.0)')
print(f'accelerate:      {accelerate.__version__}     (expected 1.11.0)')
print(f'datasets:        {datasets.__version__}      (expected 4.0.0)')
print(f'huggingface_hub: {huggingface_hub.__version__}     (expected 0.35.0)')
"

echo "[4/5] Installing LLaMA-Factory in editable mode (no deps)..."
pip install -e ~/egoblind-ra/code/LLaMA-Factory --no-deps -q

echo "[5/5] Re-applying kimi_vl_nothink template registration..."
python ~/egoblind-ra/scripts/register_template.py

echo ""
echo "================================================================"
echo "DONE. LLaMA-Factory installed at:"
echo "  ~/egoblind-ra/code/LLaMA-Factory"
echo ""
echo "Next: convert DPO pairs → ShareGPT, register dataset, launch."
echo "================================================================"
