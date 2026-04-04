# Multimodal Data Preprocessing & Visualization

End-to-end multimodal data preprocessing, feature extraction, and visualization using the [RAVDESS](https://zenodo.org/record/1188976) emotional speech/video dataset.

## Files
- `MMAI_HW1.py`: preprocessing, implementation and results  
- `MMAI_HW1.pdf`: write-up
  
## Dataset
**RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song)

- 360 video clips (`.mp4`) from 3 actors
- 8 emotion classes: neutral, calm, happy, sad, angry, fearful, disgust, surprised
- 

### Modalities Extracted

#### 1. Visual — ResNet18 Embeddings (512-dim)
- The **middle frame** of each video is extracted using OpenCV as a proxy for the peak expression
- Frames are preprocessed (resized to 224×224, normalized) and passed through a **pretrained ResNet18** with the classification head removed
- Output: a 512-dimensional embedding per video

#### 2. Audio — MFCC Features (13-dim)
- Audio is loaded directly from each `.mp4` using `librosa`
- **13 Mel-Frequency Cepstral Coefficients (MFCCs)** are extracted and averaged across time to produce a fixed-length feature vector
- Output: a 13-dimensional vector per video

#### 3. Multimodal Fusion
- Visual and audio features are concatenated into a single **525-dimensional** feature vector (`512 + 13`) per sample

## Dependencies

```
opencv-python
librosa
torch
torchvision
scikit-learn
matplotlib
seaborn
pandas
numpy
Pillow
```

Install with:

```bash
pip install opencv-python librosa torch torchvision scikit-learn matplotlib seaborn pandas numpy Pillow
```

## Key Results

- **Visual features (t-SNE):** Even without training, ResNet embeddings show partial cluster separation — emotions like *happy* and *angry* form distinct regions, while *calm* and *neutral* overlap
- **Multimodal shape:** `(360, 525)` — 360 samples × 525 features (512 visual + 13 audio)
- **Bonus — Activity classification:** Multimodal fusion (accelerometer + gyroscope) achieves **93.6% accuracy** vs. **93.2%** for accelerometer-only, with improved sitting/standing disambiguation
