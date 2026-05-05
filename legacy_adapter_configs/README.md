# Legacy Adapter Configs

These configs are from an earlier exploration of **Approach 1** (modular
hard-routing — bifurcated adapters), which was abandoned in favor of the
unified prompt-conditioned design (Approach 2) used in the final paper.

They are kept here for reference only. The configs that produced the final
results live in `configs/`:

- `configs/sft_unified.yaml` — text-only SFT (midterm baseline, superseded)
- `configs/dpo_unified.yaml` — text-only DPO (midterm, superseded)
- `configs/dpo_vision.yaml` — DPO with composite-loss-scored pairs
- `configs/dpo_f1_sft.yaml` — DPO with token-overlap F1 pairs (SFT init)
- `configs/dpo_f1_baseline.yaml` — DPO with token-overlap F1 pairs (baseline init)

The vision SFT config used to produce `sft_vision_v1` was managed via
`slurm/02_sft.sh`.
