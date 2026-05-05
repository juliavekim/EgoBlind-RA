"""
convert_csv_to_egoblind.py

Converts train_labeled.csv and test_labeled.csv into the JSON format
expected by prepare_egoblind_data.py.

Your CSV columns:
  video_name, question_id, question, answer0, answer1, answer2, answer3,
  type, start-time/s, urgency

Usage (run in Colab or on AWS):
  python convert_csv_to_egoblind.py \
      --train_csv data/train_labeled.csv \
      --test_csv  data/test_labeled.csv \
      --output_dir data/
"""

import argparse
import json
import os
import pandas as pd


ANSWER_COLS = ["answer0", "answer1", "answer2", "answer3"]


def csv_to_records(df: pd.DataFrame, split: str) -> tuple[list[dict], dict]:
    """
    Convert a labeled CSV DataFrame into:
      - records: list of dicts in EgoBlind annotation format
      - urgency_labels: {question_id: "urgent" | "non-urgent"}
    """
    # Drop the 2 ERROR rows if present
    df = df[df["urgency"].isin(["urgent", "not_urgent"])].copy()
    print(f"  [{split}] {len(df)} valid rows after dropping ERROR rows")

    records = []
    urgency_labels = {}

    for _, row in df.iterrows():
        qid = str(row["question_id"])

        # Collect non-null answers from the four answer columns
        answers = [
            str(row[col]).strip()
            for col in ANSWER_COLS
            if pd.notna(row[col]) and str(row[col]).strip() != ""
        ]
        if not answers:
            answers = []  # test set — no ground truth

        # Normalize urgency: "not_urgent" → "non-urgent", "urgent" → "urgent"
        raw_urgency = str(row["urgency"]).strip()
        urgency = "urgent" if raw_urgency == "urgent" else "non-urgent"

        record = {
            "question_id": qid,
            "question": str(row["question"]).strip(),
            "answers": answers,
            "video_id": str(int(row["video_name"])) if pd.notna(row["video_name"]) else qid,
            "question_type": str(row["type"]) if pd.notna(row.get("type")) else "",
            "urgency": urgency,
        }
        records.append(record)
        urgency_labels[qid] = urgency

    return records, urgency_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default="data/train_labeled.csv")
    parser.add_argument("--test_csv",  default="data/test_labeled.csv")
    parser.add_argument("--output_dir", default="data/")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Train ---
    print("Processing train split...")
    train_df = pd.read_csv(args.train_csv)
    train_records, train_urgency = csv_to_records(train_df, "train")

    train_ann_path = os.path.join(args.output_dir, "train.json")
    with open(train_ann_path, "w") as f:
        json.dump(train_records, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(train_records)} train records → {train_ann_path}")

    urgency_path = os.path.join(args.output_dir, "urgency_labels.json")
    with open(urgency_path, "w") as f:
        json.dump(train_urgency, f, indent=2)
    print(f"  Saved urgency labels → {urgency_path}")

    # --- Test ---
    print("Processing test split...")
    test_df = pd.read_csv(args.test_csv)
    test_records, _ = csv_to_records(test_df, "test")

    test_ann_path = os.path.join(args.output_dir, "test.json")
    with open(test_ann_path, "w") as f:
        json.dump(test_records, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(test_records)} test records → {test_ann_path}")

    # --- Summary ---
    urgent_train = sum(1 for v in train_urgency.values() if v == "urgent")
    print(f"\nDone.")
    print(f"  Train: {len(train_records)} total ({urgent_train} urgent, {len(train_records)-urgent_train} non-urgent)")
    print(f"  Test:  {len(test_records)} total")
    print(f"\nNext step — run prepare_egoblind_data.py:")
    print(f"  python scripts/prepare_egoblind_data.py \\")
    print(f"      --egoblind_dir data/ \\")
    print(f"      --urgency_labels data/urgency_labels.json \\")
    print(f"      --output_dir data/")


if __name__ == "__main__":
    main()
