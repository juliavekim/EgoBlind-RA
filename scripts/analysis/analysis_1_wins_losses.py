#!/usr/bin/env python
"""Analysis 1: Where did DPO regress vs SFT?

Bucket each test example into Win/Tie/Loss based on composite loss delta,
then categorize losses by failure mode.
"""
import json, os
import numpy as np
import torch
from collections import Counter, defaultdict
from sentence_transformers import SentenceTransformer

HOME = os.path.expanduser("~")
SFT_PREDS = f"{HOME}/egoblind-ra/output/sft_vision_test_predictions.json"
DPO_PREDS = f"{HOME}/egoblind-ra/output/dpo_vision_v2_test_predictions.json"
OUT       = f"{HOME}/egoblind-ra/output/analysis_1_wins_losses.json"
TIE_THRESH = 0.05

print("Loading data and model...")
sft_preds = {p['question_id']: p for p in json.load(open(SFT_PREDS))}
dpo_preds = {p['question_id']: p for p in json.load(open(DPO_PREDS))}
common = sorted(set(sft_preds) & set(dpo_preds))

embed = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embed.eval()

def sim(cand, refs):
    if not cand or not refs: return 0.0
    c = embed.encode(cand, convert_to_tensor=True)
    r = embed.encode(refs, convert_to_tensor=True)
    return torch.nn.functional.cosine_similarity(c.unsqueeze(0).expand_as(r), r).max().item()

def latency(text, tau_min=8, tau_max=40, p=3.0, kappa=0.3):
    n = len(text.split())
    if n <= tau_min: return 0.0
    if n >= tau_max: return kappa
    return kappa * ((n - tau_min) / (tau_max - tau_min)) ** p

def composite(cand, refs, urgency):
    s = sim(cand, refs)
    a, b, g = (0.5, 0.3, 0.2) if urgency == 'urgent' else (0.6, 0.4, 0.0)
    return a*(1-s) + b*(1-s) + g*latency(cand)

def question_type(q):
    q = q.lower().strip()
    if q.startswith(('is ', 'are ', 'am ', 'do ', 'does ', 'did ', 'can ',
                      'could ', 'should ', 'would ', 'will ', 'has ', 'have ')):
        return 'yes_no'
    if q.startswith(('where', 'in which')):     return 'where'
    if q.startswith(('what')):                  return 'what'
    if q.startswith(('how many', 'how much')):  return 'how_many'
    if q.startswith(('how')):                   return 'how'
    if q.startswith(('who')):                   return 'who'
    if q.startswith(('when')):                  return 'when'
    if q.startswith(('why')):                   return 'why'
    return 'other'

def is_idk(s):
    s = s.lower()
    return any(t in s for t in ('know', "can't", 'cannot', 'unable',
                                  'unclear', 'not sure', 'sorry', 'no information'))

def is_yes_no_answer(s):
    s = s.lower().strip().rstrip('.,!')
    if s in ('yes','yeah','yep','no','nope','not'): return True
    if s.startswith('yes') or s.startswith('no '):  return True
    return False

print(f"Computing composite loss for {len(common)} cases...")
records = []
for qid in common:
    s = sft_preds[qid]
    d = dpo_preds[qid]
    sft_pred = s['sft_prediction']
    dpo_pred = d['dpo_prediction']
    refs = s['answers']
    urgency = s['urgency']
    sft_loss = composite(sft_pred, refs, urgency)
    dpo_loss = composite(dpo_pred, refs, urgency)
    delta = dpo_loss - sft_loss     # positive = DPO worse

    if delta > TIE_THRESH:
        bucket = 'loss'
    elif delta < -TIE_THRESH:
        bucket = 'win'
    else:
        bucket = 'tie'

    records.append({
        'qid': qid,
        'question': s['question'],
        'urgency': urgency,
        'qtype': question_type(s['question']),
        'sft': sft_pred,
        'dpo': dpo_pred,
        'refs': refs,
        'sft_loss': sft_loss,
        'dpo_loss': dpo_loss,
        'delta': delta,
        'bucket': bucket,
        'sft_words': len(sft_pred.split()),
        'dpo_words': len(dpo_pred.split()),
        'sft_idk': is_idk(sft_pred),
        'dpo_idk': is_idk(dpo_pred),
    })

# Summary
n = len(records)
buckets = Counter(r['bucket'] for r in records)
print(f"\n{'='*60}")
print(f"BUCKET COUNTS (tie threshold ±{TIE_THRESH})")
print(f"{'='*60}")
for b in ('win', 'tie', 'loss'):
    print(f"  {b:6s}: {buckets[b]:5d}  ({buckets[b]/n*100:.1f}%)")
print(f"\n  Net delta (DPO - SFT): mean {np.mean([r['delta'] for r in records]):+.4f}")

# Buckets by urgency
print(f"\n{'='*60}")
print(f"BY URGENCY")
print(f"{'='*60}")
for u in ('urgent', 'not_urgent'):
    sub = [r for r in records if r['urgency'] == u]
    sub_b = Counter(r['bucket'] for r in sub)
    print(f"  {u:10s}: win {sub_b['win']:4d}  tie {sub_b['tie']:4d}  loss {sub_b['loss']:4d}  "
          f"(net Δ {np.mean([r['delta'] for r in sub]):+.4f})")

# Buckets by question type
print(f"\n{'='*60}")
print(f"BY QUESTION TYPE")
print(f"{'='*60}")
for qt, _ in Counter(r['qtype'] for r in records).most_common():
    sub = [r for r in records if r['qtype'] == qt]
    sub_b = Counter(r['bucket'] for r in sub)
    print(f"  {qt:10s} (n={len(sub):4d}): win {sub_b['win']:4d}  tie {sub_b['tie']:4d}  "
          f"loss {sub_b['loss']:4d}  net Δ {np.mean([r['delta'] for r in sub]):+.4f}")

# Failure mode taxonomy among LOSSES
print(f"\n{'='*60}")
print(f"FAILURE MODES (DPO regressions, n={buckets['loss']})")
print(f"{'='*60}")
losses = [r for r in records if r['bucket'] == 'loss']

# IDK shifts
sft_idk_dpo_committed = sum(1 for r in losses if r['sft_idk'] and not r['dpo_idk'])
sft_committed_dpo_idk = sum(1 for r in losses if not r['sft_idk'] and r['dpo_idk'])

# Yes/no polarity flips
def polarity(s):
    s = s.lower().strip().rstrip('.,!')
    if s in ('yes','yeah','yep') or s.startswith('yes'): return 'pos'
    if s in ('no','nope','not')  or s.startswith('no '): return 'neg'
    return None

yn_flip = sum(1 for r in losses
              if polarity(r['sft']) and polarity(r['dpo'])
              and polarity(r['sft']) != polarity(r['dpo']))

# Verbosity blowup
verbose_blowup = sum(1 for r in losses if r['dpo_words'] > r['sft_words'] + 5)
similar_words  = sum(1 for r in losses if abs(r['dpo_words'] - r['sft_words']) <= 2)

print(f"  SFT said IDK, DPO committed:        {sft_idk_dpo_committed:5d}  "
      f"({sft_idk_dpo_committed/len(losses)*100:.1f}%)  ← over-commitment")
print(f"  SFT committed, DPO said IDK:        {sft_committed_dpo_idk:5d}  "
      f"({sft_committed_dpo_idk/len(losses)*100:.1f}%)  ← over-hedging")
print(f"  Yes/no polarity flipped:            {yn_flip:5d}  "
      f"({yn_flip/len(losses)*100:.1f}%)  ← polarity error")
print(f"  DPO output >5 words longer than SFT:{verbose_blowup:5d}  "
      f"({verbose_blowup/len(losses)*100:.1f}%)  ← verbosity drift")
print(f"  Similar word counts (±2):           {similar_words:5d}  "
      f"({similar_words/len(losses)*100:.1f}%)  ← content change")

# Save full records for further analysis
with open(OUT, 'w') as f:
    json.dump({
        'n_total':    n,
        'tie_thresh': TIE_THRESH,
        'buckets':    dict(buckets),
        'records':    records,
    }, f, indent=2)
print(f"\nSaved → {OUT}")
