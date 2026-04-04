
# Multimodal AI — MAS.S60 / 6.S985 (MIT, Spring 2026)

Julia's coursework for the Multimodal AI course at MIT. Each folder contains a Google Colab notebook with/and associated writeup for one homework assignment.

## Repository Structure

```
├── hw1/    # Multimodal Data Preprocessing & Visualization
├── hw2/    # Multimodal Fusion and Alignment
└── hw3/    # Multimodal LLMs & VLM Fine-Tuning
```


## HW1 — Multimodal Data Preprocessing & Visualization

Extracts and visualizes via t-SNE two modalities from the **RAVDESS** dataset (360 video clips, 8 emotion classes, 3 actors). 

## HW2 — Multimodal Fusion and Alignment

Builds on HW1 using two datasets — **AV-MNIST** for fusion experiments and **RAVDESS** for contrastive alignment. Key experiments:

- **Unimodal baselines (AV-MNIST):** Image CNN ~64.6% vs. Audio CNN ~41.4% test accuracy
- **Late fusion:** Concatenation of image + audio encoders via MLP head outperforms both unimodal baselines
- **Contrastive alignment (RAVDESS):** NT-Xent loss applied to waveform vs. mel-spectrogram views; partial alignment achieved, with high-intensity emotion clips aligning better than neutral ones

## HW3 — Multimodal LLMs & VLM Fine-Tuning

Fine-tunes **Qwen2.5-VL-3B-Instruct** on **CIFAR-10** (10-class image classification) using LoRA. Key results:

| Stage | Accuracy (4 held-out images) |
|---|---|
| Baseline — pre-trained, no fine-tuning | 1/4 |
| Prompt engineering — label-constrained | 2/4 |
| LoRA fine-tuning | 4/4 |

## Requirements

See the individual `README.md` in each homework folder for dependencies and setup instructions. All notebooks are designed to run on **Google Colab** with GPU acceleration (T4 for HW2, A100 for HW3).
