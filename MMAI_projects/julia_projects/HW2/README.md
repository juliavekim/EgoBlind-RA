# Multimodal Fusion and Alignment

This assignment builds on HW1's preprocessing pipeline and introduces core techniques in multimodal fusion and cross-modal alignment, spanning tensor operations, unimodal baselines, late fusion, efficiency analysis, and contrastive learning.

> **GPU required.** The notebook is configured for a T4 GPU on Google Colab.

## Datasets

**AV-MNIST** — Audio-Visual MNIST
- Paired image (28×28 handwritten digit) and audio (112×112 spectrogram) samples
- 10-class digit classification task
- Downloaded via `gdown` from Google Drive (~1.6 GB)

**RAVDESS** — Ryerson Audio-Visual Database of Emotional Speech and Song
- 360 video clips from 3 actors, 8 emotion classes
- Used for contrastive alignment experiments (waveform vs. mel-spectrogram)
- Continued from HW1

## Dependencies

```
torch==2.3.1
torchvision==0.18.1
torchaudio==2.3.1
torchtext==0.18.0
opencv-python
librosa
gdown
memory_profiler
matplotlib
numpy
Pillow
```

Install with:

```bash
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 torchtext==0.18.0
pip install opencv-python librosa gdown memory_profiler matplotlib numpy Pillow
```

The MultiBench library (for `unimodals`, `fusions`, and `training_structures` modules) must also be cloned separately:

```bash
git clone https://github.com/pliang279/MultiBench
cd MultiBench
```

## Key Results Summary

| Model | Test Accuracy |
|---|---|
| Image-only (CNN) | ~64.6% |
| Audio-only (CNN) | ~41.4% |
| Late Fusion (Concat MLP) | > unimodal baselines |
| Contrastive Alignment (RAVDESS) | Partial — loss ~3.1–3.4 |

Image substantially outperforms audio under the current architectures, suggesting inductive bias and modality structure are the dominant performance drivers rather than hyperparameter choices. Late fusion provides gains by combining complementary signals, though the audio encoder's weakness limits the ceiling.
