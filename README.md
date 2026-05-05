# EgoBlind-RA

**Risk-adaptive egocentric visual assistance for blind and low-vision (BLV) users.**

Classifies queries by urgency and routes them to response policies matched to safety stakes. Built on [Kimi-VL-A3B-Instruct](https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct) and evaluated on the [EgoBlind benchmark](https://arxiv.org/abs/2503.08221).

**Julia Kim & Xander Backus** · MIT · Equal contribution

---

## Headline results

We compare a uniform-policy baseline against four fine-tuned configurations on the EgoBlind test set (n = 1,283), scored by composite loss (lower is better):

| Method | Composite loss | vs SFT | Mean words | Similarity |
|---|---|---|---|---|
| Baseline (uniform-policy Kimi-VL) | 0.6067 | +61.6% | 84.1 | 0.350 |
| **SFT (vision LoRA)** | **0.3754** | — | 2.2 | 0.589 |
| SFT → DPO with composite-loss pairs | 0.4334 | +15.5% | 7.3 | 0.522 |
| SFT → DPO with token-overlap F1 pairs | 0.5259 | +40.1% | 18.2 | 0.424 |
| Baseline → DPO with token-overlap F1 pairs | 0.5430 | +44.7% | 25.2 | 0.407 |

**Key findings.** SFT with shortest-correct-reference targets reduces composite loss by 38% over baseline. Every DPO variant regresses SFT — across two pair-scoring methods (composite loss vs token-overlap F1) and two initialization points (SFT-init vs baseline-init). Failure analysis traces the regression to a uniform verbosity drift across all question types and an internal conflict between the similarity and latency components of the composite reward (see `scripts/analysis/`).

The CLIP-based urgency classifier achieves **0.905 ROC-AUC** on the held-out test set.

---

## Architecture

We instantiate **Approach 2** from our paper: a unified prompt-conditioned LoRA adapter trained on the full dataset, with the predicted urgency injected as a control tag (`[URGENT]` / `[NON-URGENT]`) prepended to the system prompt. The composite loss switches per example based on the prepended tag — urgent examples are optimized with a latency penalty against the *shortest* correct reference; non-urgent are optimized without latency against the *longest* reference.

```
EgoBlind query (video + question)
            │
            ▼
   CLIP urgency classifier  ──── 0.905 ROC-AUC
            │
       [URGENT] | [NON-URGENT] tag
            │
            ▼
  Kimi-VL-A3B + unified LoRA adapter
   (composite loss switched per tag)
            │
            ▼
  Token-level F1 + composite loss eval
```

A modular hard-routing variant (Approach 1) with two specialized adapters was explored as an earlier design and is not part of the final pipeline.

---

## Trained models

All adapters are hosted on Hugging Face. See [`results/ADAPTERS.md`](results/ADAPTERS.md) for loading instructions.

| Model | Hugging Face |
|---|---|
| CLIP urgency classifier | [julia225/egoblind-ra-clip-urgency](https://huggingface.co/julia225/egoblind-ra-clip-urgency) |
| SFT (vision) | [julia225/egoblind-ra-sft-vision](https://huggingface.co/julia225/egoblind-ra-sft-vision) |
| SFT → DPO (composite-loss pairs) | [julia225/egoblind-ra-dpo-composite](https://huggingface.co/julia225/egoblind-ra-dpo-composite) |
| SFT → DPO (token-overlap F1 pairs) | [julia225/egoblind-ra-dpo-f1-sft](https://huggingface.co/julia225/egoblind-ra-dpo-f1-sft) |
| Baseline → DPO (token-overlap F1 pairs) | [julia225/egoblind-ra-dpo-f1-baseline](https://huggingface.co/julia225/egoblind-ra-dpo-f1-baseline) |

---

## Repo structure

```
EgoBlind-RA/
├── data/
├── docs/
├── models/
│   ├── clip_urgency_classifier/
│   └── prompt_conditioned_unified_model.ipynb
├── configs/
├── slurm/
├── scripts/
│   ├── analysis/
│   └── legacy/
├── kimi_vl_patches/
├── results/
│   ├── baseline/
│   ├── sft_vision/
│   ├── dpo_composite/
│   ├── dpo_f1_sft/
│   ├── dpo_f1_baseline/
│   ├── dpo_pairs/
│   ├── analysis/
│   ├── final/
│   └── ADAPTERS.md
├── legacy_adapter_configs/
├── LICENSE
└── README.md
```

- `data/` — EgoBlind splits with urgency labels, plus baseline frames.
- `docs/` — project proposal, midterm report, final paper.
- `models/clip_urgency_classifier/` — CLIP fusion-head classifier (notebook + Hugging Face link).
- `configs/` — LLaMA-Factory configs (SFT + 3 DPO variants).
- `slurm/` — Engaging cluster SLURM submission scripts.
- `scripts/` — data prep, baseline, DPO pair generation, eval.
- `scripts/analysis/` — five DPO failure analyses (1 and 2 already run; outputs in `results/analysis/`).
- `scripts/legacy/` — abandoned AWS pipeline; kept for reference only.
- `kimi_vl_patches/` — compatibility patches for Kimi-VL × `transformers >= 4.50`.
- `results/` — predictions, training logs, and analysis JSONs for every method.
- `legacy_adapter_configs/` — Approach 1 (bifurcated, abandoned) adapter configs.

---

## Reproducing the pipeline

### Environment

Engaging or any SLURM cluster with A100 / L40S:

```bash
bash slurm/00_setup_env.sh
conda activate egoblind
huggingface-cli login
python kimi_vl_patches/setup_patches.py    # one-time, idempotent
```

Pinned versions: `transformers==4.56.0`, `peft==0.13.0`, `trl==0.24.0`, `accelerate==1.11.0`, `datasets==4.0.0`, `tokenizers==0.22.0`, `torch==2.5.1+cu121`.

### Pipeline

```bash
# 1. Urgency labels (already in data/, regenerate with:)
python scripts/classify_urgency.py     # uses GPT-5.2 to label train + test

# 2. CLIP urgency classifier (notebook)
jupyter notebook models/clip_urgency_classifier/clip_urgency_classifier.ipynb

# 3. Uniform-policy baseline (Kimi API)
python scripts/kimi_api_baseline.py

# 4. Vision SFT (~4 h on A100)
sbatch slurm/02_sft.sh

# 5a. DPO with composite-loss-scored pairs (~4 h gen + 1.5 h train)
sbatch slurm/run_dpo_gen.sh
sbatch slurm/run_dpo_train.sh

# 5b. DPO with token-overlap F1 pairs
sbatch slurm/run_dpo_gen_f1.sh
sbatch slurm/run_dpo_f1.sh

# 6. Score every method against the composite-loss leaderboard
sbatch slurm/run_score.sh
```

### Failure analyses

```bash
python scripts/analysis/analysis_1_wins_losses.py            # already run
python scripts/analysis/analysis_2_verbosity.py              # already run
python scripts/analysis/analysis_3_generalization_gap.py
python scripts/analysis/analysis_4_pair_case_correlation.py
python scripts/analysis/analysis_5_reward_decomposition.py
```

---

## Author contributions

- **Julia Kim** — CLIP urgency classifier (architecture, training, evaluation), composite loss design and grid search, three-filter DPO pair construction, DPO failure analyses (1–5), Kimi-VL × transformers compatibility patches.
- **Xander Backus** — GPT-5.2-based urgency labeling pipeline, F1-scored DPO pair construction, Engaging cluster setup and LLaMA-Factory integration.
- **Joint** — problem framing, dataset preparation, paper writing.

---

## References

- Xiao et al., *EgoBlind: Towards Egocentric Visual Assistance for the Blind*, 2025. [arXiv:2503.08221](https://arxiv.org/abs/2503.08221)
- Radford et al., *Learning Transferable Visual Models From Natural Language Supervision*, 2021. [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, 2021. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- Rafailov et al., *Direct Preference Optimization*, 2023. [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)
- Bigham et al., *VizWiz: Nearly Real-time Answers to Visual Questions*, 2010.
- Grauman et al., *Ego4D: Around the World in 3,000 Hours of Egocentric Video*, 2021. [arXiv:2110.07058](https://arxiv.org/abs/2110.07058)
