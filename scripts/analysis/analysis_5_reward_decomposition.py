#!/usr/bin/env python
"""Analysis 5: Decomposed reward — did similarity and latency fight each other?

Composite loss is a weighted sum of (1 - similarity) and a length penalty.
The training pairs had chosen with HIGH similarity AND fewer words.
But DPO at test time produces longer outputs with LOWER similarity.

This script shows the reward components moved in opposite directions:
   sim_loss component:    SFT good ─→ DPO worse (gave up similarity)
   latency component:     SFT good ─→ DPO worse (gave up brevity)
"""
import json, os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

HOME = os.path.expanduser("~")
SFT_PREDS = f"{HOME}/egoblind-ra/output/sft_vision_test_predictions.json"
DPO_PREDS = f"{HOME}/egoblind-ra/output/dpo_vision_v2_test_predictions.json"
PAIRS     = f"{HOME}/egoblind-ra/output/dpo_pairs_vision_filtered.json"
OUT       = f"{HOME}/egoblind-ra/output/analysis_5_reward_decomposition.json"

print("Loading...")
sft_preds = {p['question_id']: p for p in json.load(open(SFT_PREDS))}
dpo_preds = {p['question_id']: p for p in json.load(open(DPO_PREDS))}
common = sorted(set(sft_preds) & set(dpo_preds))

embed = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embed.eval()

def sim(cand, refs):
    if not cand or not refs: return 0.0
    c = embed.encode(cand, convert_to_tensor=True)
    r = embed.encode(refs, convert_to_tensor=True)
    return torch.nn.functional.cosine_similarity(
        c.unsqueeze(0).expand_as(r), r).max().item()

def latency_pen(text, tau_min=8, tau_max=40, p=3.0, kappa=0.3):
    n = len(text.split())
    if n <= tau_min: return 0.0
    if n >= tau_max: return kappa
    return kappa * ((n - tau_min) / (tau_max - tau_min)) ** p

# Decompose composite loss for SFT and DPO
def decompose(cand, refs, urgency):
    s = sim(cand, refs)
    sim_loss = 1 - s
    lat_loss = latency_pen(cand)
    a, b, g = (0.5, 0.3, 0.2) if urgency == 'urgent' else (0.6, 0.4, 0.0)
    sim_component = (a + b) * sim_loss     # acc + util both use (1 - sim)
    lat_component = g * lat_loss
    return sim_loss, lat_loss, sim_component, lat_component

print(f"\nDecomposing composite loss for {len(common)} cases...")
sft_sim, sft_lat, sft_sc, sft_lc = [], [], [], []
dpo_sim, dpo_lat, dpo_sc, dpo_lc = [], [], [], []
for qid in common:
    s = sft_preds[qid]
    d = dpo_preds[qid]
    refs = s['answers']
    u = s['urgency']
    a, b, c, dd = decompose(s['sft_prediction'], refs, u)
    sft_sim.append(a); sft_lat.append(b); sft_sc.append(c); sft_lc.append(dd)
    a, b, c, dd = decompose(d['dpo_prediction'], refs, u)
    dpo_sim.append(a); dpo_lat.append(b); dpo_sc.append(c); dpo_lc.append(dd)

print("="*70)
print("COMPOSITE LOSS DECOMPOSITION")
print("="*70)
print(f"  {'':30s} {'SFT':>10s} {'DPO':>10s} {'Δ (DPO-SFT)':>15s}")
print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*15}")

ssim, dsim = np.mean(sft_sim),  np.mean(dpo_sim)
slat, dlat = np.mean(sft_lat),  np.mean(dpo_lat)
ssc,  dsc  = np.mean(sft_sc),   np.mean(dpo_sc)
slc,  dlc  = np.mean(sft_lc),   np.mean(dpo_lc)
total_sft = ssc + slc
total_dpo = dsc + dlc

print(f"  Raw (1 - similarity):          {ssim:>10.4f} {dsim:>10.4f} {dsim-ssim:>+15.4f}")
print(f"  Raw latency loss:              {slat:>10.4f} {dlat:>10.4f} {dlat-slat:>+15.4f}")
print()
print(f"  Weighted similarity component: {ssc:>10.4f} {dsc:>10.4f} {dsc-ssc:>+15.4f}")
print(f"  Weighted latency component:    {slc:>10.4f} {dlc:>10.4f} {dlc-slc:>+15.4f}")
print(f"  Total composite loss:          {total_sft:>10.4f} {total_dpo:>10.4f} "
      f"{total_dpo-total_sft:>+15.4f}")

# Decomposition by urgency (latency only matters for urgent)
print(f"\n{'='*70}")
print(f"BY URGENCY")
print(f"{'='*70}")
for ui, ulabel in enumerate(['urgent', 'not_urgent']):
    idxs = [i for i, qid in enumerate(common) if sft_preds[qid]['urgency'] == ulabel]
    if not idxs: continue
    print(f"\n  {ulabel} (n={len(idxs)}):")
    print(f"    Sim component:  SFT {np.mean([sft_sc[i] for i in idxs]):.4f}  "
          f"→ DPO {np.mean([dpo_sc[i] for i in idxs]):.4f}  "
          f"(Δ {np.mean([dpo_sc[i] - sft_sc[i] for i in idxs]):+.4f})")
    print(f"    Lat component:  SFT {np.mean([sft_lc[i] for i in idxs]):.4f}  "
          f"→ DPO {np.mean([dpo_lc[i] for i in idxs]):.4f}  "
          f"(Δ {np.mean([dpo_lc[i] - sft_lc[i] for i in idxs]):+.4f})")

# What did the training pairs look like?
print(f"\n{'='*70}")
print(f"TRAINING PAIR REWARD PROFILE (the signal DPO was given)")
print(f"{'='*70}")
pairs = json.load(open(PAIRS))
chosen_sim   = [p['_meta']['chosen_sim']   for p in pairs]
rejected_sim = [p['_meta']['rejected_sim'] for p in pairs]
chosen_w     = [p['_meta']['chosen_words']   for p in pairs]
rejected_w   = [p['_meta']['rejected_words'] for p in pairs]
print(f"  Chosen   words mean: {np.mean(chosen_w):.1f}  similarity mean: {np.mean(chosen_sim):.3f}")
print(f"  Rejected words mean: {np.mean(rejected_w):.1f}  similarity mean: {np.mean(rejected_sim):.3f}")
print(f"\n  Training preference signal:")
print(f"    'Chosen' has both HIGHER similarity AND FEWER words (composite loss aligned)")
print(f"\n  Test-time DPO output (mean across {len(common)} cases):")
print(f"    DPO words: {np.mean([len(dpo_preds[qid]['dpo_prediction'].split()) for qid in common]):.1f}  "
      f"similarity: {1 - np.mean(dpo_sim):.3f}")
print(f"\n  ⚠ DPO drifted toward LONGER and LESS similar — opposite of training chosen.")

# Save
with open(OUT, 'w') as f:
    json.dump({
        'sft': {
            'sim_component_mean': float(ssc),
            'lat_component_mean': float(slc),
            'total_mean':         float(total_sft),
        },
        'dpo': {
            'sim_component_mean': float(dsc),
            'lat_component_mean': float(dlc),
            'total_mean':         float(total_dpo),
        },
        'training_pairs': {
            'chosen_words_mean':   float(np.mean(chosen_w)),
            'chosen_sim_mean':     float(np.mean(chosen_sim)),
            'rejected_words_mean': float(np.mean(rejected_w)),
            'rejected_sim_mean':   float(np.mean(rejected_sim)),
        },
    }, f, indent=2)
print(f"\nSaved → {OUT}")
