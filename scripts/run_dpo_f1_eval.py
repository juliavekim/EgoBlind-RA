#!/usr/bin/env python
"""Eval a DPO_F1 adapter on EgoBlind test set.

Usage:
    python run_dpo_f1_eval.py sft
    python run_dpo_f1_eval.py baseline
"""
import sys, json, os, gc, torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel

INIT = sys.argv[1] if len(sys.argv) > 1 else "sft"
assert INIT in ("sft", "baseline"), "Usage: python run_dpo_f1_eval.py [sft|baseline]"

HOME       = os.path.expanduser("~")
DATA_DIR   = f"{HOME}/egoblind-ra/data"
OUTPUT_DIR = f"{HOME}/egoblind-ra/output"
DRIVE_ROOT = f"{HOME}/egoblind-ra"

SFT_ADAPTER  = f"{OUTPUT_DIR}/sft_vision_v1"
DPO_ADAPTER  = f"{OUTPUT_DIR}/dpo_f1_{INIT}"
OUT_FILE     = f"{OUTPUT_DIR}/dpo_f1_{INIT}_test_predictions.json"
BASELINE_FILE = f"{DRIVE_ROOT}/baseline/baseline_predictions.json"


def main():
    print(f"[init={INIT}]")
    print("[load] base + processor")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
    )
    base = AutoModelForCausalLM.from_pretrained(
        "moonshotai/Kimi-VL-A3B-Instruct",
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(
        "moonshotai/Kimi-VL-A3B-Instruct", trust_remote_code=True
    )

    if INIT == "sft":
        print(f"[load] SFT adapter from {SFT_ADAPTER}")
        sft = PeftModel.from_pretrained(base, SFT_ADAPTER)
        sft = sft.merge_and_unload()
        print(f"[load] DPO_F1_SFT adapter from {DPO_ADAPTER}")
        model = PeftModel.from_pretrained(sft, DPO_ADAPTER)
    else:
        print(f"[load] DPO_F1_BASELINE adapter from {DPO_ADAPTER}")
        model = PeftModel.from_pretrained(base, DPO_ADAPTER)
    model.eval()

    print(f"[data] loading test prompts from {BASELINE_FILE}")
    with open(BASELINE_FILE) as f:
        baseline = json.load(f)
    baseline = [b for b in baseline if b.get("split") == "test"]
    print(f"[data] {len(baseline)} test cases")

    predictions = []
    seen = set()
    if os.path.isfile(OUT_FILE):
        predictions = json.load(open(OUT_FILE))
        seen = {p["question_id"] for p in predictions}
        print(f"[resume] {len(predictions)} already saved")

    pbar = tqdm(baseline, desc=f"Eval ({INIT})")
    for i, item in enumerate(pbar):
        if item["question_id"] in seen:
            continue
        frame_dir = f"{DATA_DIR}/baseline_frames/test/{item['question_id']}"
        if not os.path.isdir(frame_dir):
            continue
        frames = sorted(f for f in os.listdir(frame_dir) if f.endswith(".jpg"))
        if not frames:
            continue
        image = Image.open(os.path.join(frame_dir, frames[-1])).convert("RGB")

        tag = "[URGENT]" if item["urgency"] == "urgent" else "[NON-URGENT]"
        text = f"<|media_pad|>{tag} {item['question']}"
        messages = [{"role": "user", "content": text}]
        inputs_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=inputs_text, images=image,
                           return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=64,
                                 do_sample=False, use_cache=False)
        pred = processor.decode(
            out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        ).strip()

        predictions.append({
            "question_id":    item["question_id"],
            "urgency":        item["urgency"],
            "question":       item["question"],
            "answers":        item["answers"],
            f"dpo_f1_{INIT}_prediction": pred,
        })

        if (i + 1) % 50 == 0:
            with open(OUT_FILE, "w") as f:
                json.dump(predictions, f, indent=2, ensure_ascii=False)
            pbar.set_postfix(saved=len(predictions))

    with open(OUT_FILE, "w") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    print(f"\n[done] saved {len(predictions)} predictions to {OUT_FILE}")


if __name__ == "__main__":
    main()
