#!/usr/bin/env python
"""Analysis 4: Pair-level vs case-level correlation.

The 975 training pairs had varying loss gaps (median 0.436, range 0.20-1.05).
Were the pairs with biggest gaps the ones that drove the most aberrant
test-time behavior? Or did smaller-gap pairs accumulate similar effects?

Approach: link training pairs to test cases by question_id (when a question
appears in both). For test cases that ALSO appear in training pairs, see if
DPO regression correlates with training pair gap.

If YES: pairs with strong "preference signal" most disrupted generation.
If NO:  the regression is driven by aggregate behavior, not specific pairs.
"""
import json, os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from collections import defaultdict

HOME = os.path.expanduser("~")
SFT_PREDS = f"{HOME}/egoblind-ra/output/sft_vision_test_predictions.json"
DPO_PREDS = f"{HOME}/egoblind-ra/output/dpo_vision_v2_test_predictions.json"
PAIRS     = f"{HOME}/egoblind-ra/output/dpo_pairs_vision_filtered.json"
OUT       = f"{HOME}/egoblind-ra/output/analysis_4_pair_case_correlation.json"

print("Loading...")
sft_preds = {p['question_id']: p for p in json.load(open(SFT_PREDS))}
dpo_preds = {p['question_id']: p for p in json.load(open(DPO_PREDS))}
pairs = json.load(open(PAIRS))

# Note: training pair qids and test qids should be DIFFERENT (train/test split).
# But the SAME UNDERLYING VIDEO might appear in both with different question_ids.
# We can't directly link a training pair to a test case.
#
# Instead, we group training pairs by gap-magnitude bucket and look at
# DPO's overall test behavior. Then check whether question-type or
# urgency distributions in high-gap pairs predict where DPO regresses on test.

# ============================================================================
# Part A — distribution of training gaps
# ============================================================================
print("="*70)
print("PART A: TRAINING PAIR GAP DISTRIBUTION")
print("="*70)
gaps = np.array([p['_meta']['loss_gap'] for p in pairs])
print(f"\n  n pairs: {len(pairs)}")
print(f"  Gap percentiles:")
for q in [10, 25, 50, 75, 90, 99]:
    print(f"    p{q:2d}: {np.percentile(gaps, q):.3f}")
print(f"  Mean: {gaps.mean():.3f}  Std: {gaps.std():.3f}")

# Bucket pairs by gap
def gap_bucket(g):
    if g < 0.3:  return 'small (0.2–0.3)'
    if g < 0.5:  return 'medium (0.3–0.5)'
    if g < 0.7:  return 'large (0.5–0.7)'
    return 'huge (>0.7)'

bucket_pairs = defaultdict(list)
for p in pairs:
    bucket_pairs[gap_bucket(p['_meta']['loss_gap'])].append(p)

print(f"\n  Pairs by gap bucket:")
for b in ['small (0.2–0.3)', 'medium (0.3–0.5)', 'large (0.5–0.7)', 'huge (>0.7)']:
    sub = bucket_pairs[b]
    if not sub: continue
    cw = np.mean([p['_meta']['chosen_words']   for p in sub])
    rw = np.mean([p['_meta']['rejected_words'] for p in sub])
    cs = np.mean([p['_meta']['chosen_sim']     for p in sub])
    rs = np.mean([p['_meta']['rejected_sim']   for p in sub])
    print(f"    {b:22s}  n={len(sub):4d}  "
          f"chosen={cw:.1f}w sim={cs:.2f} | rejected={rw:.1f}w sim={rs:.2f}")

# ============================================================================
# Part B — what kinds of questions populated high-gap pairs?
# ============================================================================
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

# We need the question text — try to get it from pairs (conversation field)
def get_question(p):
    text = p['conversations'][0]['value']
    # strip "<image>[URGENT] " prefix
    for prefix in ['<image>[URGENT] ', '<image>[NON-URGENT] ', '[URGENT] ', '[NON-URGENT] ']:
        if text.startswith(prefix):
            return text[len(prefix):]
    return text

# Note: pairs store the prompt; question type can be inferred
print(f"\n{'='*70}")
print(f"PART B: QUESTION TYPES IN HIGH-GAP PAIRS")
print(f"{'='*70}")
print(f"  {'qtype':12s}  {'small':>6s}  {'medium':>6s}  {'large':>6s}  {'huge':>6s}")
qtype_x_bucket = defaultdict(lambda: defaultdict(int))
for p in pairs:
    q = get_question(p)
    qt = question_type(q)
    bk = gap_bucket(p['_meta']['loss_gap'])
    qtype_x_bucket[qt][bk] += 1

all_qts = sorted(qtype_x_bucket.keys(), key=lambda x: -sum(qtype_x_bucket[x].values()))
for qt in all_qts:
    sm = qtype_x_bucket[qt]['small (0.2–0.3)']
    md = qtype_x_bucket[qt]['medium (0.3–0.5)']
    lg = qtype_x_bucket[qt]['large (0.5–0.7)']
    hg = qtype_x_bucket[qt]['huge (>0.7)']
    print(f"  {qt:12s}  {sm:>6d}  {md:>6d}  {lg:>6d}  {hg:>6d}")

# ============================================================================
# Part C — does the qtype-level training distribution predict test regression?
# ============================================================================
print(f"\n{'='*70}")
print(f"PART C: TRAINING PAIR DISTRIBUTION vs TEST REGRESSION BY QTYPE")
print(f"{'='*70}")

embed = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embed.eval()

def sim(cand, refs):
    if not cand or not refs: return 0.0
    c = embed.encode(cand, convert_to_tensor=True)
    r = embed.encode(refs, convert_to_tensor=True)
    return torch.nn.functional.cosine_similarity(
        c.unsqueeze(0).expand_as(r), r).max().item()

def latency(text, tau_min=8, tau_max=40, p=3.0, kappa=0.3):
    n = len(text.split())
    if n <= tau_min: return 0.0
    if n >= tau_max: return kappa
    return kappa * ((n - tau_min) / (tau_max - tau_min)) ** p

def composite(cand, refs, urgency):
    s = sim(cand, refs)
    a, b, g = (0.5, 0.3, 0.2) if urgency == 'urgent' else (0.6, 0.4, 0.0)
    return a*(1-s) + b*(1-s) + g*latency(cand)

# For each qtype, compute (a) mean training pair gap, (b) mean test regression
qtype_train_gap     = defaultdict(list)
qtype_test_regress  = defaultdict(list)

print("  Computing per-qtype test regressions...")
common = sorted(set(sft_preds) & set(dpo_preds))
for qid in common:
    s = sft_preds[qid]
    d = dpo_preds[qid]
    qt = question_type(s['question'])
    sft_l = composite(s['sft_prediction'], s['answers'], s['urgency'])
    dpo_l = composite(d['dpo_prediction'], s['answers'], s['urgency'])
    qtype_test_regress[qt].append(dpo_l - sft_l)

for p in pairs:
    qt = question_type(get_question(p))
    qtype_train_gap[qt].append(p['_meta']['loss_gap'])

print(f"\n  {'qtype':12s}  {'n_train':>7s}  {'mean_gap':>9s}  "
      f"{'n_test':>6s}  {'mean_regress':>12s}")
correlation_data = []
for qt in sorted(set(qtype_train_gap) | set(qtype_test_regress),
                  key=lambda x: -len(qtype_test_regress.get(x, []))):
    n_train = len(qtype_train_gap.get(qt, []))
    n_test  = len(qtype_test_regress.get(qt, []))
    if n_train == 0 or n_test == 0: continue
    mean_gap     = np.mean(qtype_train_gap[qt])
    mean_regress = np.mean(qtype_test_regress[qt])
    print(f"  {qt:12s}  {n_train:>7d}  {mean_gap:>9.3f}  "
          f"{n_test:>6d}  {mean_regress:>+12.4f}")
    correlation_data.append((mean_gap, mean_regress, qt))

# Compute correlation
if len(correlation_data) >= 3:
    gaps_arr    = np.array([x[0] for x in correlation_data])
    regress_arr = np.array([x[1] for x in correlation_data])
    if gaps_arr.std() > 0 and regress_arr.std() > 0:
        corr = np.corrcoef(gaps_arr, regress_arr)[0, 1]
        print(f"\n  Pearson correlation (train_gap vs test_regression): {corr:+.3f}")
        if abs(corr) < 0.3:
            print(f"  → WEAK correlation: high-gap pairs did NOT specifically drive regressions.")
            print(f"    Implies regression is from aggregate signal, not 'bad pairs'.")
        elif corr > 0.3:
            print(f"  → POSITIVE correlation: question types with bigger training gaps")
            print(f"    saw worse test regression. High-gap pairs hurt generation.")
        elif corr < -0.3:
            print(f"  → NEGATIVE correlation: bigger training gaps actually helped.")

with open(OUT, 'w') as f:
    json.dump({
        'gap_distribution': {
            'mean':   float(gaps.mean()),
            'median': float(np.median(gaps)),
            'p90':    float(np.percentile(gaps, 90)),
            'max':    float(gaps.max()),
        },
        'qtype_x_bucket': {qt: dict(v) for qt, v in qtype_x_bucket.items()},
        'correlation_data': [
            {'qtype': qt, 'mean_train_gap': mg, 'mean_test_regress': mr}
            for mg, mr, qt in correlation_data
        ],
    }, f, indent=2)
print(f"\nSaved → {OUT}")
