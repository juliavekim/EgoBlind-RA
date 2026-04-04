
# Multimodal LLMs & Vision-Language Model Fine-Tuning

Introduce Vision-Language Models (VLMs), covering baseline inference, prompt engineering, LoRA fine-tuning, and post-training evaluation to a curated image dataset from CIFAR-10-style . The model used throughout is **Qwen2.5-VL-3B-Instruct**, a 3-billion-parameter vision-language model loaded via HuggingFace Transformers and fine-tuned with **LoRA (Low-Rank Adaptation)** using the PEFT library. 

> **GPU required.** This notebook is configured for an **A100 GPU** on Google Colab Pro.


## Dataset

CIFAR-10 is used as the image dataset for this homework. CIFAR-10 consists of 32×32 colour images across 10 object classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. Images are stored and annotated in the following structure: 

```
mmai-data/
├── images/
│   ├── image_01.jpg
│   ├── image_02.jpg
│   └── ...
└── data.jsonl
```

Each line of `data.jsonl` is a training example:

```json
{
  "image": "images/image_01.jpg",
  "question": "What object is shown in this image? Answer with a single label.",
  "answer": "deer"
}
```

The dataset is split into a **training set** and a **held-out test set** (4+ images reserved for evaluation, never seen during fine-tuning). An example CIFAR-10-style dataset is provided as a scaffold.

## Model

| Property | Value |
|---|---|
| Base model | `Qwen2.5-VL-3B-Instruct` |
| Fine-tuning method | LoRA (via HuggingFace PEFT) |
| Precision | `float16` |
| Framework | HuggingFace Transformers + TRL |
| GPU | A100 (40 GB) |


## Dependencies

```
torch
torchvision
transformers
peft
trl
accelerate
Pillow
```

Install with:

```bash
pip install torch torchvision transformers peft trl accelerate Pillow
```

## Key Results

| Stage | Accuracy (4 held-out images) |
|---|---|
| Baseline (pre-trained, no fine-tuning) | 1/4 |
| Prompt engineering (label-constrained) | 2/4 |
| LoRA fine-tuning | 4/4 |

- **Prompt engineering:** Explicitly listing the 10 CIFAR-10 labels was the most effective prompt strategy. Few-shot prompting matched this but added no further gain. Chain-of-thought prompting performed worst, introducing more hallucinations.
- **LoRA fine-tuning:** Best config — 3 epochs, LR = 1×10⁻⁴, rank = 4, alpha = 8. Increasing epochs or rank slightly worsened performance, likely due to overfitting on the small dataset. Final training loss: **1.3076**.
- **Post-fine-tuning:** All three baseline errors corrected (deer, frog, automobile). No new errors introduced. 
