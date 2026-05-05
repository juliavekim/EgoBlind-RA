#!/usr/bin/env python
"""Run inference with SFT+DPO stack on EgoBlind test set."""
import json, os, time, torch, gc, glob, re
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel

HOME       = os.path.expanduser("~")
DATA_DIR   = f"{HOME}/egoblind-ra/data"
OUTPUT_DIR = f"{HOME}/egoblind-ra/output"
DRIVE_ROOT = f"{HOME}/egoblind-ra"

SFT_ADAPTER = f"{OUTPUT_DIR}/sft_vision_v1"
DPO_ADAPTER = f"{OUTPUT_DIR}/dpo_vision_v2"
OUT_FILE    = f"{OUTPUT_DIR}/dpo_vision_v2_test_predictions.json"

# Same baseline JSON structure as §6.5 SFT eval
BASELINE_FILE = f"{DRIVE_ROOT}/baseline/baseline_predictions.json"


def main():
    print(f"[load] base + processor")
    bnb = BitsAndBytesConfig(load_in_4bit=True,
                              bnb_4bit_compute_dtype=torch.bfloat16,
                              bnb_4bit_quant_type="nf4")
    base = AutoModelForCausalLM.from_pretrained(
        "moonshotai/Kimi-VL-A3B-Instruct",
        quantization_config=bnb, device_map="auto", trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained("moonshotai/Kimi-VL-A3B-Instruct",
                                                trust_remote_code=True)

    print(f"[load] SFT adapter from {SFT_ADAPTER}")
    sft = PeftModel.from_pretrained(base, SFT_ADAPTER)
    sft = sft.merge_and_unload()

    print(f"[load] DPO adapter from {DPO_ADAPTER}")
    model = PeftModel.from_pretrained(sft, DPO_ADAPTER)
    model.eval()
    print("[load] done\n")

    # Load test items
    with open(BASELINE_FILE) as f:
        baseline = json.load(f)
    test_items = [r for r in baseline if r['split'] == 'test']
    print(f"[data] {len(test_items)} test examples\n")

    def get_test_image_path(qid):
        frame_dir = f"{DATA_DIR}/baseline_frames/test/{qid}"
        if not os.path.isdir(frame_dir):
            return None
        frames = sorted(f for f in os.listdir(frame_dir) if f.endswith('.jpg'))
        return os.path.join(frame_dir, frames[-1]) if frames else None

    def generate(tag, question, image_path):
        image = Image.open(image_path).convert("RGB")
        text = f"<|media_pad|>{tag} {question}"
        messages = [{"role": "user", "content": text}]
        inputs_text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=inputs_text, images=image,
                           return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=32,
                                 do_sample=False, use_cache=False)
        return processor.decode(out[0][inputs['input_ids'].shape[1]:],
                                skip_special_tokens=True).strip()

    # Resume support — skip qids already in output
    predictions = []
    seen = set()
    if os.path.isfile(OUT_FILE):
        predictions = json.load(open(OUT_FILE))
        seen = {p['question_id'] for p in predictions}
        print(f"[resume] {len(predictions)} already saved\n")

    start = time.time()
    for r in tqdm(test_items):
        if r['question_id'] in seen:
            continue
        image_path = get_test_image_path(r['question_id'])
        if not image_path:
            continue
        tag = "[URGENT]" if r['urgency'] == 'urgent' else "[NON-URGENT]"
        try:
            dpo_pred = generate(tag, r['question'], image_path)
        except Exception as e:
            print(f"[error] {r['question_id']}: {e}")
            dpo_pred = ""
        predictions.append({
            'question_id':         r['question_id'],
            'question':            r['question'],
            'urgency':             r['urgency'],
            'answers':             r['answers'],
            'baseline_prediction': r['prediction'],
            'dpo_prediction':      dpo_pred,
        })

        if len(predictions) % 100 == 0:
            with open(OUT_FILE, 'w') as f:
                json.dump(predictions, f, indent=2, ensure_ascii=False)

    with open(OUT_FILE, 'w') as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - start
    print(f"\n[done] {len(predictions)} predictions in {elapsed/60:.1f} min")
    print(f"[done] saved to {OUT_FILE}")


if __name__ == "__main__":
    main()
