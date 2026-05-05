# DPO Failure Analyses

Five analyses dissecting why metric-based DPO regressed SFT performance.
All scripts read the SFT and DPO test predictions from `results/` and write
JSON outputs.

| # | Script | Question answered | Output | Run? |
|---|--------|-------------------|--------|------|
| 1 | `analysis_1_wins_losses.py` | Where did DPO regress? Failure-mode taxonomy of regressions. | `results/analysis/analysis_1_wins_losses.json` | ✓ |
| 2 | `analysis_2_verbosity.py` | Was verbosity drift uniform across question types or selective? | `results/analysis/analysis_2_verbosity.json` | ✓ |
| 3 | `analysis_3_generalization_gap.py` | Did DPO learn to discriminate but fail to generate? | not yet run | — |
| 4 | `analysis_4_pair_case_correlation.py` | Did high-gap training pairs drive worst test regressions? | not yet run | — |
| 5 | `analysis_5_reward_decomposition.py` | Did similarity and latency loss components fight? | not yet run | — |

## Headline findings (analyses 1 & 2)

- DPO regressed SFT on a substantial fraction of test cases, with the
  dominant failure mode being verbosity drift.
- Mean output length jumped from 2.2 (SFT) to 7.3 words (DPO) and was
  near-uniform across question types (yes/no +4.8, what +5.2, where +5.9,
  how +6.8), suggesting DPO learned a *global* "be wordy" preference rather
  than question-conditional length rules.

## Running the analyses

```bash
conda activate egoblind   # needs sentence_transformers
python scripts/analysis/analysis_1_wins_losses.py
python scripts/analysis/analysis_2_verbosity.py
# etc.
```
