# CLIP Urgency Classifier

Binary urgency classifier for egocentric BLV queries. Predicts whether a
(video, question) pair is *urgent* (safety-critical) or *non-urgent*.

## Trained checkpoint

The trained weights (580MB) are hosted on Hugging Face:
**https://huggingface.co/julia225/egoblind-ra-clip-urgency**

```python
from huggingface_hub import hf_hub_download
ckpt = hf_hub_download(
    repo_id="julia225/egoblind-ra-clip-urgency",
    filename="final_model.pt",
)
```

## Architecture

CLIP ViT-B/32 (frozen) + 2-layer MLP head. 4 frames sampled from a ±2s window
around the query timestamp, mean-pooled, concatenated with CLIP text embedding.

## Performance

| Metric    | Validation | Test  |
|-----------|------------|-------|
| Accuracy  | 0.863      | 0.798 |
| Precision | 0.930      | 0.879 |
| Recall    | 0.807      | 0.695 |
| F1        | 0.864      | 0.777 |
| ROC-AUC   | 0.938      | 0.905 |

## Training

See `../clip_urgency_classifier.ipynb` for the full training notebook.
Trained on EgoBlind with GPT-5.2-generated urgency labels, BCE loss, AdamW
(lr=1e-4), 5 epochs, single NVIDIA L40S GPU.
