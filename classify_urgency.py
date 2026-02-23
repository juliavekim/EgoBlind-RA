import os
import csv
import base64
import time
import json
import datetime
import cv2  # pip3 install opencv-python
from openai import OpenAI  # pip3 install openai

client = OpenAI(api_key="KEY_REDACTED")

PROJECT_DIR = "."
VIDEO_DIR = os.path.join(PROJECT_DIR, "train_videos")
CSV_PATH = os.path.join(PROJECT_DIR, "train.csv")
OUTPUT_PATH = os.path.join(PROJECT_DIR, "train_labeled.csv")
CHECKPOINT_PATH = os.path.join(PROJECT_DIR, "checkpoint.json")

MODEL = "gpt-5.2-2025-12-11"


def extract_frames(video_path, num_frames=5):
    """Extract evenly-spaced frames from a video, return as base64 JPEG strings."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        return []

    indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    frames_b64 = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            _, buffer = cv2.imencode(".jpg", frame)
            frames_b64.append(base64.standard_b64encode(buffer).decode("utf-8"))

    cap.release()
    return frames_b64


def classify_urgency(video_name, question, start_time):
    """Send frames + question to GPT-5.2 and get urgent/not_urgent."""
    video_path = os.path.join(VIDEO_DIR, f"{video_name}.mp4")
    frames = extract_frames(video_path, num_frames=5)

    content = []

    for frame_b64 in frames:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{frame_b64}",
                "detail": "low"
            }
        })

    content.append({
        "type": "text",
        "text": (
            f"These are frames from an egocentric video recorded by a blind person "
            f"(start time: {start_time}s). They are asking the following question:\n\n"
            f"Question: {question}\n\n"
            f"Does this person need an immediate answer for their safety or navigation? "
            f"Respond with ONLY: urgent or not_urgent"
        )
    })

    response = client.chat.completions.create(
        model=MODEL,
        reasoning_effort="xhigh",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a classifier for the EgoBlind dataset, which contains egocentric "
                    "video from blind or visually impaired people navigating the real world. "
                    "You must classify each question as 'urgent' or 'not_urgent'.\n\n"
                    "A question is URGENT if it relates to:\n"
                    "- Immediate physical safety (oncoming cars, traffic, crossing streets)\n"
                    "- Obstacles or hazards (stairs, curbs, holes, objects in the path)\n"
                    "- Navigation decisions that need a fast answer (is it safe to walk, which way to go)\n"
                    "- Any danger or time-sensitive situation\n\n"
                    "A question is NOT_URGENT if it relates to:\n"
                    "- General scene description (what is around me, describe the area)\n"
                    "- Reading text (signs, labels, menus)\n"
                    "- Identifying objects with no safety implication (what color is this, what brand is this)\n"
                    "- Non-time-sensitive information\n\n"
                    "Respond with ONLY one word: urgent or not_urgent"
                )
            },
            {"role": "user", "content": content}
        ],
        max_completion_tokens=2048
    )

    raw = response.choices[0].message.content
    if raw is None:
        print(f"  WARNING: model returned None content. Full response: {response.choices[0]}")
        return "NONE_RESPONSE"

    label = raw.strip().lower()

    if "urgent" in label and "not" in label:
        return "not_urgent"
    elif "urgent" in label:
        return "urgent"
    else:
        print(f"  WARNING: unexpected label '{label}', defaulting to not_urgent")
        return "not_urgent"


def load_checkpoint():
    """Load checkpoint to resume from where we left off."""
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "r") as f:
            return json.load(f)
    return {"last_completed_index": -1, "labels": {}}


def save_checkpoint(checkpoint):
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(checkpoint, f)


def main():
    with open(CSV_PATH, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Loaded {len(rows)} questions.")

    checkpoint = load_checkpoint()
    start_index = checkpoint["last_completed_index"] + 1
    labels = checkpoint["labels"]

    if start_index > 0:
        print(f"Resuming from row {start_index} ({start_index} already done).")

    start_time_clock = time.time()

    for i in range(start_index, len(rows)):
        row = rows[i]
        video_name = row["video_name"]
        question = row["question"]
        start_time = row["start-time/s"]

        try:
            label = classify_urgency(video_name, question, start_time)
        except Exception as e:
            print(f"  ERROR on row {i}: {e}")
            label = "ERROR"
            time.sleep(5)

        labels[str(i)] = label
        print(f"  Row {i}: [{label}] video={video_name} q=\"{question[:60]}\"")

        # Checkpoint every 10 rows
        if (i + 1) % 10 == 0:
            checkpoint["last_completed_index"] = i
            checkpoint["labels"] = labels
            save_checkpoint(checkpoint)

        # Progress update every 25 rows
        if (i + 1) % 25 == 0:
            elapsed = time.time() - start_time_clock
            rows_done = i + 1 - start_index
            rate = rows_done / elapsed
            remaining = (len(rows) - (i + 1)) / rate
            eta = datetime.datetime.now() + datetime.timedelta(seconds=remaining)
            urgent_count = sum(1 for v in labels.values() if v == "urgent")
            total_labeled = len(labels)
            print(
                f"  === [{i + 1}/{len(rows)}] "
                f"{rate:.1f} rows/sec | "
                f"ETA: {eta.strftime('%H:%M:%S')} | "
                f"~{remaining / 60:.0f} min remaining | "
                f"urgent: {urgent_count}/{total_labeled} ({100 * urgent_count / total_labeled:.1f}%) ==="
            )

        time.sleep(0.5)

    # Final checkpoint
    checkpoint["last_completed_index"] = len(rows) - 1
    checkpoint["labels"] = labels
    save_checkpoint(checkpoint)

    # Apply labels and write output
    for i, row in enumerate(rows):
        row["urgency"] = labels.get(str(i), "not_urgent")

    fieldnames = list(rows[0].keys())
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done! Output saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()