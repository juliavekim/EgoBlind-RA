# EgoBlind-RA

**Risk-adaptive egocentric visual assistance for blind and low-vision (BLV) users.**

Classifies queries by urgency and routes them to response policies matched to safety stakes. Built on [Kimi-VL-A3B-Instruct](https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct) and evaluated on the [EgoBlind benchmark](https://arxiv.org/abs/2503.08221).

**Julia Kim & Xander Backus** · MIT · Equal contribution

---

## Headline results

We compare two urgency-conditioned architectures against a uniform-policy baseline on the EgoBlind test set (n = 1,283), scored by composite loss (lower is better).

**Approach 2 (unified adapter, Julia):**

| Method | Composite loss | vs SFT | Mean words | Similarity |
|---|---|---|---|---|
| Baseline (uniform-policy Kimi-VL) | 0.607 | +61.6% | 84.1 | 0.350 |
| **SFT (vision LoRA)** | **0.375** | — | 2.2 | 0.589 |
| SFT → DPO with composite-loss pairs | 0.433 | +15.5% | 7.3 | 0.522 |
| SFT → DPO with token-overlap F1 pairs | 0.526 | +40.1% | 18.2 | 0.424 |
| Baseline → DPO with token-overlap F1 pairs | 0.543 | +44.7% | 25.2 | 0.407 |

**Approach 1 (bifurcated adapters, Xander):**

| Method | Urgent sim | Non-urg. sim | Overall sim |
|---|---|---|---|
| Baseline (uniform policy) | 0.124 | 0.080 | 0.103 |
| SFT only | 0.505 | 0.268 | 0.391 |
| DPO only (from base) | 0.277 | 0.270 | 0.273 |
| SFT → DPO | 0.459 | 0.298 | 0.381 |
| **Best mix (SFT urgent + DPO non-urgent)** | **0.507** | **0.298** | **0.407** |

**End-to-end pipeline (Approach 1 with CLIP routing):**

| Routing strategy | Loss | Similarity | Mean words |
|---|---|---|---|
| Baseline (uniform) | 0.607 | 0.350 | 84.1 |
| Zero-shot (prompted Kimi-VL) | 0.597 | 0.346 | 5.1 |
| **CLIP classifier** | **0.556** | **0.393** | 4.5 |
| Oracle (ground-truth) | 0.549 | 0.399 | 4.0 |

**Key findings:**
- SFT with shortest/longest-correct-reference targets reduces composite loss by 38% over baseline in both approaches.
- DPO's effect depends sharply on architecture: it regresses SFT uniformly in the unified adapter (Approach 2), but helps the non-urgent adapter in the bifurcated design (Approach 1). The explanation: urgent references are 1–3 words, so preference pairs lack variation; non-urgent references are longer and give DPO real signal.
- The unified adapter is robust to CLIP routing noise at deployment (matches oracle loss despite 21.5% tag flips), while the bifurcated design depends on classifier quality. The CLIP classifier closes 88% of the oracle gap.
- The CLIP-based urgency classifier achieves **0.905 ROC-AUC** on the held-out test set.

---

## Architecture

We investigate two architectures for urgency-conditioned response generation, both using the same CLIP urgency classifier and composite loss.

**Approach 1 (bifurcated adapters):** Two separate LoRA adapters — one for urgent queries (SFT, brevity-optimized), one for non-urgent (SFT+DPO, detail-optimized). The classifier hard-routes between them. Higher ceiling under oracle routing, but sensitive to classifier errors.

**Approach 2 (unified adapter):** A single LoRA adapter trained on the full dataset with `[URGENT]` / `[NON-URGENT]` tags prepended to the system prompt. Simpler deployment, robust to routing noise, but cannot specialize as deeply per regime.

```
EgoBlind query (video + question)
            │
            ▼
   CLIP urgency classifier  ──── 0.905 ROC-AUC
            │
            ├── Approach 1: hard-route to urgent OR non-urgent adapter
            │
            └── Approach 2: prepend [URGENT] or [NON-URGENT] tag
                             to single unified adapter
            │
            ▼
  Urgency-calibrated response
```

---

## Trained models

All adapters are hosted on Hugging Face.

**Approach 2 (unified):**

| Model | Hugging Face |
|---|---|
| CLIP urgency classifier | [julia225/egoblind-ra-clip-urgency](https://huggingface.co/julia225/egoblind-ra-clip-urgency) |
| SFT (vision) | [julia225/egoblind-ra-sft-vision](https://huggingface.co/julia225/egoblind-ra-sft-vision) |
| SFT → DPO (composite-loss pairs) | [julia225/egoblind-ra-dpo-composite](https://huggingface.co/julia225/egoblind-ra-dpo-composite) |
| SFT → DPO (token-overlap F1 pairs) | [julia225/egoblind-ra-dpo-f1-sft](https://huggingface.co/julia225/egoblind-ra-dpo-f1-sft) |
| Baseline → DPO (token-overlap F1 pairs) | [julia225/egoblind-ra-dpo-f1-baseline](https://huggingface.co/julia225/egoblind-ra-dpo-f1-baseline) |

**Approach 1 (bifurcated):**

| Model | Hugging Face |
|---|---|
| All 6 adapters (SFT + DPO, urgent + non-urgent) | [xabackus/egoblind-ra-adapters](https://huggingface.co/xabackus/egoblind-ra-adapters) |

See [`results/ADAPTERS.md`](results/ADAPTERS.md) for loading instructions, including DPO adapter stacking.

---

## Repo structure

```
EgoBlind-RA/
├── data/                           EgoBlind splits with urgency labels + frames
├── docs/                           Proposal, midterm, final paper, presentation
├── models/
│   ├── clip_urgency_classifier/    CLIP fusion-head classifier notebook
│   └── prompt_conditioned_unified_model.ipynb
├── configs/                        LLaMA-Factory YAML configs (both approaches)
├── slurm/                          Engaging cluster SLURM scripts (both approaches)
├── scripts/
│   ├── analysis/                   DPO failure analyses
│   └── legacy/                     Abandoned AWS pipeline (reference only)
├── kimi_vl_patches/                Kimi-VL × transformers >= 4.50 compatibility
├── results/
│   ├── baseline/
│   ├── sft_vision/
│   ├── dpo_composite/
│   ├── dpo_f1_sft/
│   ├── dpo_f1_baseline/
│   ├── dpo_pairs/
│   ├── analysis/
│   ├── final/                      Summary JSONs (both approaches)
│   └── ADAPTERS.md
├── legacy_adapter_configs/         Early Approach 1 adapter configs (pre-vision)
├── LICENSE
└── README.md
```

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

### Shared steps

```bash
# 1. Urgency labels (already in data/, regenerate with:)
python scripts/classify_urgency.py     # uses GPT-5.2 to label train + test

# 2. CLIP urgency classifier (notebook)
jupyter notebook models/clip_urgency_classifier/clip_urgency_classifier.ipynb

# 3. Uniform-policy baseline (Kimi API)
python scripts/kimi_api_baseline.py
```

### Approach 2 (unified adapter)

```bash
# 4. Vision SFT (~4 h on A100)
sbatch slurm/02_sft.sh

# 5a. DPO with composite-loss-scored pairs (~4 h gen + 1.5 h train)
sbatch slurm/run_dpo_gen.sh
sbatch slurm/run_dpo_train.sh

# 5b. DPO with token-overlap F1 pairs
sbatch slurm/run_dpo_gen_f1.sh
sbatch slurm/run_dpo_f1.sh

# 6. Score every method
sbatch slurm/run_score.sh
```

### Approach 1 (bifurcated adapters)

```bash
# 4. Prepare vision SFT data (splits by urgency, selects shortest/longest refs)
python scripts/prepare_vision_sft.py

# 5. Train urgent + non-urgent SFT adapters
#    Uses configs/sft_urgent_vision.yaml and configs/sft_nonurgent_vision_v2.yaml
sbatch slurm/run_eval_three_conditions.sh  # or submit via LLaMA-Factory CLI

# 6. Generate DPO pairs from each model
python scripts/generate_dpo_pairs_vision.py
python scripts/generate_dpo_pairs_vision_base.py
python scripts/generate_dpo_pairs_vision_base_nonurgent.py
python scripts/generate_dpo_pairs_vision_sft_nonurgent.py

# 7. Train DPO adapters
#    Uses configs/dpo_*.yaml

# 8. Evaluate: 3-condition eval (base, oracle, zero-shot pipeline)
python scripts/evaluate_three_conditions.py

# 9. Evaluate: full pipeline with CLIP classifier
python scripts/evaluate_pipeline_clip.py

# 10. Re-score with canonical paper params (no GPU needed)
python scripts/rescore_pipeline_canonical.py
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

- **Julia Kim** — Approach 2 (unified prompt-conditioned adapter): architecture, SFT and DPO training, end-to-end pipeline evaluation showing robustness to CLIP routing noise. CLIP urgency classifier (architecture, training, evaluation, HuggingFace deployment). Composite loss design and grid search. Three-filter DPO pair construction. DPO failure analyses (1–5). Kimi-VL × transformers compatibility patches.
- **Xander Backus** — Approach 1 (bifurcated adapters): architecture, vision-conditioned SFT training (including discovery that text-only training collapses to pattern matching), four DPO configurations, best-mix finding (SFT urgent + DPO non-urgent). End-to-end CLIP pipeline evaluation (88% oracle gap closure). GPT-5.2-based urgency labeling pipeline. Engaging cluster infrastructure and LLaMA-Factory integration.
- **Joint** — problem framing, dataset preparation, composite loss formulation, cross-architecture DPO analysis, paper writing, presentation.

---

## Citation

```bibtex
@misc{kim2026egoblindra,
  title  = {EgoBlind-RA: Towards Safer Egocentric Assistive AI for Blind Users via Risk-Adaptive Routing},
  author = {Kim, Julia and Backus, Xander},
  year   = {2026},
  url    = {https://github.com/juliavekim/EgoBlind-RA},
}
```

---

## References

- Xiao et al., *EgoBlind: Towards Egocentric Visual Assistance for the Blind*, 2025. [arXiv:2503.08221](https://arxiv.org/abs/2503.08221)
- Radford et al., *Learning Transferable Visual Models From Natural Language Supervision*, 2021. [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, 2021. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- Rafailov et al., *Direct Preference Optimization*, 2023. [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)
- Shao et al., *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*, 2024. [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
- Bigham et al., *VizWiz: Nearly Real-time Answers to Visual Questions*, 2010.
- Grauman et al., *Ego4D: Around the World in 3,000 Hours of Egocentric Video*, 2021. [arXiv:2110.07058](https://arxiv.org/abs/2110.07058)
