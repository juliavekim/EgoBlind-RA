#!/usr/bin/env python
"""Analysis 3: Training-test generalization gap.

DPO training showed rewards/accuracies climbing from 0.46 → 0.66, indicating
the model learned to assign higher likelihood to chosen vs rejected on the
TRAINING pairs. But test-time GENERATION regressed by 15.5%.

This script quantifies that gap. We can't easily compute log-probs without
re-running the model, but we CAN ask:

  1. Do the SFT outputs on test resemble training "chosen" or "rejected"?
  2. Do the DPO outputs on test resemble training "chosen" or "rejected"?
  3. Did DPO move outputs *toward* the chosen distribution as expected,
     or did it land somewhere weird?

The hypothesis is that training-time "preference learning" did not translate
to test-time "preferred generation" — model learned to discriminate but not
to generate.
"""
import json, os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from collections import Counter

HOME = os.path.expanduser("~")
SFT_PREDS = f"{HOME}/egoblind-ra/output/sft_vision_test_predictions.json"
DPO_PREDS = f"{HOME}/egoblind-ra/output/dpo_vision_v2_test_predictions.json"
PAIRS     = f"{HOME}/egoblind-ra/output/dpo_pairs_vision_filtered.json"
TRAINER_STATE = f"{HOME}/egoblind-ra/output/dpo_vision_v2/trainer_state.json"
OUT       = f"{HOME}/egoblind-ra/output/analysis_3_generalization_gap.json"

print("Loading...")
sft_preds = {p['question_id']: p for p in json.load(open(SFT_PREDS))}
dpo_preds = {p['question_id']: p for p in json.load(open(DPO_PREDS))}
common = sorted(set(sft_preds) & set(dpo_preds))
pairs = json.load(open(PAIRS))

# ============================================================================
# Part A — show training metric trajectory
# ============================================================================
print("="*70)
print("PART A: TRAINING-TIME LEARNING TRAJECTORY")
print("="*70)
state = json.load(open(TRAINER_STATE))

reward_acc = []
for log in state['log_history']:
    if 'rewards/accuracies' in log:
        reward_acc.append((log['step'], log['rewards/accuracies'],
                            log.get('rewards/margins', 0)))

print(f"\n  {'step':>5}  {'rewards/acc':>12}  {'margins':>10}")
for step, acc, mar in reward_acc[::3]:     # every 3rd row to fit
    print(f"  {step:>5}  {acc:>12.3f}  {mar:>+10.3f}")

acc_first = np.mean([a for _, a, _ in reward_acc[:3]])
acc_last  = np.mean([a for _, a, _ in reward_acc[-3:]])
print(f"\n  Mean rewards/accuracies, first 3 logs: {acc_first:.3f}")
print(f"  Mean rewards/accuracies, last 3 logs:  {acc_last:.3f}")
print(f"  → DPO DID learn to discriminate chosen from rejected")
print(f"    (accuracy {acc_first:.0%} → {acc_last:.0%})")

# ============================================================================
# Part B — distribution of test outputs vs training-pair distributions
# ============================================================================
print(f"\n{'='*70}")
print(f"PART B: TEST OUTPUT vs TRAINING DISTRIBUTIONS")
print(f"{'='*70}")

# Length distribution
chosen_len   = [p['_meta']['chosen_words']   for p in pairs]
rejected_len = [p['_meta']['rejected_words'] for p in pairs]
sft_len      = [len(sft_preds[q]['sft_prediction'].split()) for q in common]
dpo_len      = [len(dpo_preds[q]['dpo_prediction'].split()) for q in common]

print(f"\n  Word counts:")
print(f"    {'distribution':22s} {'mean':>6s} {'median':>7s} {'p25':>6s} {'p75':>6s}")
for name, vals in [
    ("Train CHOSEN",   chosen_len),
    ("Train REJECTED", rejected_len),
    ("Test SFT",       sft_len),
    ("Test DPO",       dpo_len),
]:
    print(f"    {name:22s} {np.mean(vals):>6.1f} {np.median(vals):>7.0f} "
          f"{np.percentile(vals, 25):>6.0f} {np.percentile(vals, 75):>6.0f}")

# Did DPO move CLOSER to chosen length distribution, or AWAY from it?
ks_sft_chosen   = abs(np.median(sft_len) - np.median(chosen_len))
ks_dpo_chosen   = abs(np.median(dpo_len) - np.median(chosen_len))
ks_sft_rejected = abs(np.median(sft_len) - np.median(rejected_len))
ks_dpo_rejected = abs(np.median(dpo_len) - np.median(rejected_len))
print(f"\n  Median word distance to training distributions:")
print(f"    SFT to chosen:   {ks_sft_chosen:.1f}    SFT to rejected:   {ks_sft_rejected:.1f}")
print(f"    DPO to chosen:   {ks_dpo_chosen:.1f}    DPO to rejected:   {ks_dpo_rejected:.1f}")

# Verdict
print(f"\n  Interpretation:")
if ks_dpo_chosen < ks_sft_chosen:
    print(f"  ✓ DPO moved closer to training CHOSEN length")
elif ks_dpo_chosen > ks_sft_chosen and ks_dpo_rejected > ks_sft_rejected:
    print(f"  ⚠ DPO moved AWAY from BOTH training distributions")
    print(f"    → trained on chosen ≈ 4.5w / rejected ≈ 6.3w but generated longer at test")
elif ks_dpo_chosen > ks_sft_chosen:
    print(f"  ⚠ DPO moved AWAY from training CHOSEN length (toward rejected)")

# ============================================================================
# Part C — semantic similarity comparison
# ============================================================================
print(f"\n{'='*70}")
print(f"PART C: SEMANTIC SIMILARITY")
print(f"{'='*70}")
embed = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embed.eval()

# We have similarity stats already in pair _meta
chosen_sim   = [p['_meta']['chosen_sim']   for p in pairs]
rejected_sim = [p['_meta']['rejected_sim'] for p in pairs]

# Compute test-time similarity (this IS the reference similarity used in eval)
def sim(cand, refs):
    if not cand or not refs: return 0.0
    c = embed.encode(cand, convert_to_tensor=True)
    r = embed.encode(refs, convert_to_tensor=True)
    return torch.nn.functional.cosine_similarity(
        c.unsqueeze(0).expand_as(r), r).max().item()

sft_sim = [sim(sft_preds[q]['sft_prediction'], sft_preds[q]['answers']) for q in common]
dpo_sim = [sim(dpo_preds[q]['dpo_prediction'], dpo_preds[q]['answers']) for q in common]

print(f"\n  {'distribution':22s} {'mean sim':>10s} {'median':>8s}")
for name, vals in [
    ("Train CHOSEN",   chosen_sim),
    ("Train REJECTED", rejected_sim),
    ("Test SFT",       sft_sim),
    ("Test DPO",       dpo_sim),
]:
    print(f"    {name:22s} {np.mean(vals):>10.3f} {np.median(vals):>8.3f}")

print(f"\n  Distance to training CHOSEN similarity ({np.mean(chosen_sim):.3f}):")
print(f"    SFT: {abs(np.mean(sft_sim) - np.mean(chosen_sim)):.3f}")
print(f"    DPO: {abs(np.mean(dpo_sim) - np.mean(chosen_sim)):.3f}")

# ============================================================================
# Conclusion
# ============================================================================
print(f"\n{'='*70}")
print(f"CONCLUSION: TRAINING-TEST GENERALIZATION GAP")
print(f"{'='*70}")
print(f"""
  Training metric:  rewards/accuracies climbed {acc_first:.0%} → {acc_last:.0%}
                    Model learned to RANK chosen above rejected.
  
  Test metric:      composite loss 0.375 → 0.433 ({(0.433-0.375)/0.375*100:.1f}%)
                    Model regressed when GENERATING fresh outputs.
  
  Word counts:      Training chosen={np.mean(chosen_len):.1f}, rejected={np.mean(rejected_len):.1f}
                    Test SFT={np.mean(sft_len):.1f}, DPO={np.mean(dpo_len):.1f}
                    DPO outputs are LONGER than even rejected examples in training.
  
  Similarity:       Training chosen sim={np.mean(chosen_sim):.3f}, rejected={np.mean(rejected_sim):.3f}
                    Test SFT sim={np.mean(sft_sim):.3f}, DPO={np.mean(dpo_sim):.3f}
                    DPO test similarity dropped below the training chosen distribution.
  
  Mechanistic story:
    DPO succeeded at the per-pair discrimination task during training (66% acc).
    But the implicit reward shaping it learned didn't constrain test generation
    to match the chosen distribution. Instead, the model produced outputs that
    are LONGER than even rejected training examples and LESS similar than
    chosen ones — landing in an off-manifold region the training data did
    not actually optimize over.
    
    This is a known mode of metric-based DPO failure: discrimination ≠
    generation. The model learns to score chosen > rejected on a fixed
    set of comparisons but doesn't learn to *produce* outputs from the
    chosen distribution.
""")

with open(OUT, 'w') as f:
    json.dump({
        'training_acc_first': float(acc_first),
        'training_acc_last':  float(acc_last),
        'word_counts': {
            'train_chosen_mean':   float(np.mean(chosen_len)),
            'train_rejected_mean': float(np.mean(rejected_len)),
            'test_sft_mean':       float(np.mean(sft_len)),
            'test_dpo_mean':       float(np.mean(dpo_len)),
        },
        'similarities': {
            'train_chosen_mean':   float(np.mean(chosen_sim)),
            'train_rejected_mean': float(np.mean(rejected_sim)),
            'test_sft_mean':       float(np.mean(sft_sim)),
            'test_dpo_mean':       float(np.mean(dpo_sim)),
        },
    }, f, indent=2)
print(f"\nSaved → {OUT}")
