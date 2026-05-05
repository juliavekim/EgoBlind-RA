#!/usr/bin/env python
"""Analysis 2: Verbosity drift, decomposed.

Did DPO uniformly inflate output length, or selectively expand on certain
question types? How does DPO output length relate to reference length?
"""
import json, os
import numpy as np
from collections import defaultdict

HOME = os.path.expanduser("~")
SFT_PREDS = f"{HOME}/egoblind-ra/output/sft_vision_test_predictions.json"
DPO_PREDS = f"{HOME}/egoblind-ra/output/dpo_vision_v2_test_predictions.json"
OUT       = f"{HOME}/egoblind-ra/output/analysis_2_verbosity.json"

print("Loading...")
sft_preds = {p['question_id']: p for p in json.load(open(SFT_PREDS))}
dpo_preds = {p['question_id']: p for p in json.load(open(DPO_PREDS))}
common = sorted(set(sft_preds) & set(dpo_preds))

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

# Build records
records = []
for qid in common:
    s = sft_preds[qid]
    d = dpo_preds[qid]
    sft_w = len(s['sft_prediction'].split())
    dpo_w = len(d['dpo_prediction'].split())
    ref_lens = [len(r.split()) for r in s['answers']]
    records.append({
        'urgency':  s['urgency'],
        'qtype':    question_type(s['question']),
        'sft_w':    sft_w,
        'dpo_w':    dpo_w,
        'delta_w':  dpo_w - sft_w,
        'ref_med':  np.median(ref_lens) if ref_lens else 0,
        'ref_max':  max(ref_lens) if ref_lens else 0,
        'ref_mean': np.mean(ref_lens) if ref_lens else 0,
    })

n = len(records)
print(f"n = {n} test cases\n")

# Overall
print("="*70)
print("OVERALL WORD COUNTS")
print("="*70)
print(f"  SFT:        mean {np.mean([r['sft_w'] for r in records]):5.1f}  "
      f"median {np.median([r['sft_w'] for r in records]):4.0f}  "
      f"p90 {np.percentile([r['sft_w'] for r in records], 90):4.0f}")
print(f"  DPO:        mean {np.mean([r['dpo_w'] for r in records]):5.1f}  "
      f"median {np.median([r['dpo_w'] for r in records]):4.0f}  "
      f"p90 {np.percentile([r['dpo_w'] for r in records], 90):4.0f}")
print(f"  References: mean {np.mean([r['ref_mean'] for r in records]):5.1f}  "
      f"median {np.median([r['ref_med'] for r in records]):4.0f}  "
      f"max {np.max([r['ref_max'] for r in records]):4.0f}")
print(f"  Δ words (DPO − SFT): mean {np.mean([r['delta_w'] for r in records]):+5.1f}  "
      f"median {np.median([r['delta_w'] for r in records]):+5.1f}")

# By urgency
print(f"\n{'='*70}")
print("WORD COUNTS BY URGENCY")
print("="*70)
for u in ('urgent', 'not_urgent'):
    sub = [r for r in records if r['urgency'] == u]
    print(f"  {u}:")
    print(f"    SFT mean {np.mean([r['sft_w'] for r in sub]):.1f}  → "
          f"DPO mean {np.mean([r['dpo_w'] for r in sub]):.1f}  "
          f"(Δ {np.mean([r['delta_w'] for r in sub]):+.1f})")
    print(f"    Ref mean {np.mean([r['ref_mean'] for r in sub]):.1f}")

# By question type
print(f"\n{'='*70}")
print("WORD COUNTS BY QUESTION TYPE")
print("="*70)
print(f"  {'qtype':10s} {'n':>5s}  {'sft_w':>7s} {'dpo_w':>7s} {'Δw':>6s}  "
      f"{'ref_w':>7s}  {'dpo-ref':>9s}")
qtype_counts = defaultdict(list)
for r in records:
    qtype_counts[r['qtype']].append(r)
for qt in sorted(qtype_counts.keys(), key=lambda x: -len(qtype_counts[x])):
    sub = qtype_counts[qt]
    sft_w   = np.mean([r['sft_w']    for r in sub])
    dpo_w   = np.mean([r['dpo_w']    for r in sub])
    delta_w = np.mean([r['delta_w']  for r in sub])
    ref_w   = np.mean([r['ref_mean'] for r in sub])
    print(f"  {qt:10s} {len(sub):5d}  {sft_w:7.1f} {dpo_w:7.1f} {delta_w:+6.1f}  "
          f"{ref_w:7.1f}  {dpo_w - ref_w:+9.1f}")

# Did DPO produce outputs LONGER than typical reference?
print(f"\n{'='*70}")
print("DOES DPO OVERSHOOT REFERENCE LENGTH?")
print("="*70)
overshoot = [r for r in records if r['dpo_w'] > r['ref_max']]
undershoot = [r for r in records if r['dpo_w'] < r['ref_med'] / 2]
in_range = [r for r in records
            if r['ref_med'] / 2 <= r['dpo_w'] <= r['ref_max']]
print(f"  DPO output > longest ref:        {len(overshoot):5d}  "
      f"({len(overshoot)/n*100:.1f}%) ← over-explanation")
print(f"  DPO output < half median ref:    {len(undershoot):5d}  "
      f"({len(undershoot)/n*100:.1f}%) ← extreme brevity")
print(f"  DPO output in [½median, max]:    {len(in_range):5d}  "
      f"({len(in_range)/n*100:.1f}%) ← appropriate range")

# Comparison: SFT did this much better
sft_overshoot = [r for r in records if r['sft_w'] > r['ref_max']]
sft_in_range  = [r for r in records
                 if r['ref_med'] / 2 <= r['sft_w'] <= r['ref_max']]
print(f"\n  For comparison, SFT:")
print(f"    SFT output > longest ref:      {len(sft_overshoot):5d}  "
      f"({len(sft_overshoot)/n*100:.1f}%)")
print(f"    SFT output in [½median, max]:  {len(sft_in_range):5d}  "
      f"({len(sft_in_range)/n*100:.1f}%)")

with open(OUT, 'w') as f:
    json.dump({
        'n': n,
        'overall': {
            'sft_mean':     float(np.mean([r['sft_w']    for r in records])),
            'dpo_mean':     float(np.mean([r['dpo_w']    for r in records])),
            'ref_mean':     float(np.mean([r['ref_mean'] for r in records])),
            'delta_w_mean': float(np.mean([r['delta_w']  for r in records])),
        },
        'by_qtype': {
            qt: {
                'n':       len(sub),
                'sft_w':   float(np.mean([r['sft_w']    for r in sub])),
                'dpo_w':   float(np.mean([r['dpo_w']    for r in sub])),
                'ref_w':   float(np.mean([r['ref_mean'] for r in sub])),
                'delta_w': float(np.mean([r['delta_w']  for r in sub])),
            } for qt, sub in qtype_counts.items()
        },
        'records': records,
    }, f, indent=2)
print(f"\nSaved → {OUT}")
