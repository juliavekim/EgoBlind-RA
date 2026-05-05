#!/usr/bin/env python
"""Generate DPO pairs using token-overlap F1 against longest reference.

Matches partner's setup:
  - 4 candidates per prompt at temperature 0.9
  - Score each via token-overlap F1 vs the LONGEST reference answer
  - Chosen = highest F1, rejected = lowest F1
  - Skip only when all 4 candidates are identical
  - No min loss-gap filter

Use --sft_adapter to initialize from SFT, omit to use raw baseline.
"""
import argparse, json, os, re
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel


def tokenize(s):
    s = re.sub(r"[^\w\s]", " ", s.lower())
    return s.split()


def f1_overlap(pred, ref):
    p, r = set(tokenize(pred)), set(tokenize(ref))
    if not p or not r: return 0.0
    common = p & r
    if not common: return 0.0
    pre = len(common) / len(p)
    rec = len(common) / len(r)
    return 2 * pre * rec / (pre + rec)


def longest_ref(refs):
    return max(refs, key=lambda r: len(r.split()))


def generate_candidates(model, processor, image, text, n=4, temperature=0.9):
    messages = [{"role": "user", "content": text}]
    inputs_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=inputs_text, images=image,
                       return_tensors="pt").to(model.device)
    candidates = []
    for _ in range(n):
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=64,
                                  do_sample=True, temperature=temperature, top_p=0.9,
                                  use_cache=False)
        candidates.append(
            processor.decode(out[0][inputs.input_ids.shape[1]:],
                             skip_special_tokens=True).strip()
        )
    return candidates


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sft_adapter",  default=None,
                   help="Path to SFT adapter; omit for baseline")
    p.add_argument("--train_data",   required=True)
    p.add_argument("--frames_root",  required=True)
    p.add_argument("--output",       required=True)
    p.add_argument("--n_candidates", type=int, default=4)
    p.add_argument("--temperature",  type=float, default=0.9)
    p.add_argument("--save_every",   type=int, default=25)
    args = p.parse_args()

    pairs = []
    seen = set()
    if os.path.isfile(args.output):
        pairs = json.load(open(args.output))
        seen = {p["question_id"] for p in pairs}
        print(f"Resuming: {len(pairs)} pairs already saved")

    # Load model
    init = "SFT" if args.sft_adapter else "baseline"
    print(f"Loading Kimi-VL ({init} init) with 4-bit quantization...")
    bnb = BitsAndBytesConfig(load_in_4bit=True,
                              bnb_4bit_compute_dtype=torch.bfloat16,
                              bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(
        "moonshotai/Kimi-VL-A3B-Instruct",
        quantization_config=bnb, device_map="auto", trust_remote_code=True,
    )
    if args.sft_adapter:
        print(f"Loading SFT adapter from {args.sft_adapter}")
        model = PeftModel.from_pretrained(model, args.sft_adapter)
    model.eval()
    processor = AutoProcessor.from_pretrained("moonshotai/Kimi-VL-A3B-Instruct",
                                                trust_remote_code=True)
    print("Loaded.")

    skip_id = 0
    with open(args.train_data) as f:
        records = [json.loads(line) for line in f]

    pbar = tqdm(records, desc="Generating")
    for i, r in enumerate(pbar):
        if r["question_id"] in seen:
            continue
        frame_dir = f"{args.frames_root}/{r['question_id']}"
        if not os.path.isdir(frame_dir):
            continue
        frames = sorted(f for f in os.listdir(frame_dir) if f.endswith(".jpg"))
        if not frames:
            continue
        image = Image.open(os.path.join(frame_dir, frames[-1])).convert("RGB")
        tag = "[URGENT]" if r["urgency"] == "urgent" else "[NON-URGENT]"
        text = f"<|media_pad|>{tag} {r['question']}"
        cands = generate_candidates(model, processor, image, text,
                                     n=args.n_candidates,
                                     temperature=args.temperature)

        if len(set(cands)) == 1:
            skip_id += 1
            continue

        target_ref = longest_ref(r["refs"])
        scored = [{"text": c, "f1": f1_overlap(c, target_ref),
                    "words": len(c.split())} for c in cands]
        scored.sort(key=lambda s: s["f1"], reverse=True)
        chosen, rejected = scored[0], scored[-1]

        pairs.append({
            "question_id":   r["question_id"],
            "urgency":       r["urgency"],
            "conversations": [{"from": "human", "value": text}],
            "chosen":        {"from": "gpt", "value": chosen["text"]},
            "rejected":      {"from": "gpt", "value": rejected["text"]},
            "images":        [os.path.join(frame_dir, frames[-1])],
            "_meta": {
                "chosen_f1":      chosen["f1"],
                "chosen_words":   chosen["words"],
                "rejected_f1":    rejected["f1"],
                "rejected_words": rejected["words"],
                "f1_gap":         chosen["f1"] - rejected["f1"],
                "target_ref":     target_ref,
                "n_candidates":   args.n_candidates,
                "temperature":    args.temperature,
                "init":           init,
            },
        })

        if (i + 1) % args.save_every == 0:
            with open(args.output, "w") as f:
                json.dump(pairs, f, indent=2, ensure_ascii=False)
            pbar.set_postfix(saved=len(pairs), skip_id=skip_id)

    with open(args.output, "w") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)

    print(f"\n=== Done ({init} init) ===")
    print(f"Generated: {len(pairs)}")
    print(f"Skipped (all identical): {skip_id}")
    if pairs:
        import numpy as np
        gaps = [p["_meta"]["f1_gap"] for p in pairs]
        cw = [p["_meta"]["chosen_words"] for p in pairs]
        rw = [p["_meta"]["rejected_words"] for p in pairs]
        cf1 = [p["_meta"]["chosen_f1"] for p in pairs]
        rf1 = [p["_meta"]["rejected_f1"] for p in pairs]
        print(f"F1 gap median: {np.median(gaps):.3f}  mean: {np.mean(gaps):.3f}")
        print(f"Chosen   F1 mean: {np.mean(cf1):.3f}  words mean: {np.mean(cw):.1f}")
        print(f"Rejected F1 mean: {np.mean(rf1):.3f}  words mean: {np.mean(rw):.1f}")


if __name__ == "__main__":
    main()
