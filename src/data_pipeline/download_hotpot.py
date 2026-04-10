# Downloads HotpotQA from HuggingFace and saves it as JSON for the pipeline.

import argparse
import json
import os


def download(out_path, max_samples, split):
    from datasets import load_dataset

    print(f"Loading HotpotQA (distractor / {split}) from HuggingFace...")
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)

    records = []
    for item in list(ds)[:max_samples]:
        # HuggingFace stores context as separate title and sentences lists.
        # Convert to list of [title, sentences] pairs — this is what hotpot_loader.py expects.
        titles    = item["context"]["title"]
        sentences = item["context"]["sentences"]

        context = []
        for i in range(len(titles)):
            context.append([titles[i], sentences[i]])

        records.append({
            "_id":      item["id"],
            "question": item["question"],
            "answer":   item["answer"],
            "context":  context,
            "type":     item.get("type", ""),
            "level":    item.get("level", ""),
        })

    # Save all records to disk as JSON
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)

    # Count how many samples per question type
    types = {}
    for r in records:
        types[r["type"]] = types.get(r["type"], 0) + 1

    print(f"\nSaved {len(records)} HotpotQA samples → {out_path}")
    print(f"  split : {split}")
    print(f"  types : {types}")
    print(f"\nRun experiments:")
    print(f"  python experiments/run_experiment.py --dataset hotpot --quick")
    print(f"  python experiments/run_experiment.py --dataset hotpot")


def main():
    p = argparse.ArgumentParser(description="Download HotpotQA for the RAG pipeline")
    p.add_argument("--out",         default="data/raw/hotpot_train.json",
                   help="Output path (default: data/raw/hotpot_train.json)")
    p.add_argument("--max-samples", type=int, default=500,
                   help="Number of samples to save (default: 500)")
    p.add_argument("--split",       default="validation",
                   choices=["train", "validation"],
                   help="HotpotQA split to use (default: validation)")
    args = p.parse_args()
    download(args.out, args.max_samples, args.split)


if __name__ == "__main__":
    main()
