# Trained Adapters

All LoRA adapter weights are hosted on Hugging Face (each ~280MB,
too large to commit directly to GitHub).

| Method | Hugging Face | Local results |
|---|---|---|
| SFT (vision) | [julia225/egoblind-ra-sft-vision](https://huggingface.co/julia225/egoblind-ra-sft-vision) | `results/sft_vision/` |
| SFT → DPO (composite loss) | [julia225/egoblind-ra-dpo-composite](https://huggingface.co/julia225/egoblind-ra-dpo-composite) | `results/dpo_composite/` |
| SFT → DPO (token-overlap F1) | [julia225/egoblind-ra-dpo-f1-sft](https://huggingface.co/julia225/egoblind-ra-dpo-f1-sft) | `results/dpo_f1_sft/` |
| Baseline → DPO (token-overlap F1) | [julia225/egoblind-ra-dpo-f1-baseline](https://huggingface.co/julia225/egoblind-ra-dpo-f1-baseline) | `results/dpo_f1_baseline/` |
| CLIP urgency classifier | [julia225/egoblind-ra-clip-urgency](https://huggingface.co/julia225/egoblind-ra-clip-urgency) | `models/clip_urgency_classifier/` |

## Loading an adapter

```python
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(
    "moonshotai/Kimi-VL-A3B-Instruct",
    trust_remote_code=True,
)
adapter_dir = snapshot_download("julia225/egoblind-ra-sft-vision")
model = PeftModel.from_pretrained(base, adapter_dir)
```

## Final composite-loss results

| Method | Composite loss | vs SFT | Mean words | Similarity |
|---|---|---|---|---|
| Baseline (uniform-policy Kimi-VL) | 0.6067 | +61.6% | 84.1 | 0.350 |
| **SFT (vision LoRA)** | **0.3754** | — | 2.2 | 0.589 |
| SFT → DPO (composite) | 0.4334 | +15.5% | 7.3 | 0.522 |
| SFT → DPO (F1) | 0.5259 | +40.1% | 18.2 | 0.424 |
| Baseline → DPO (F1) | 0.5430 | +44.7% | 25.2 | 0.407 |

Test set: n = 1,283 (subset with full prediction coverage across all methods).
