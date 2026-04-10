# Loads FEVER claims from HuggingFace and prepares them for the pipeline.

import argparse
import json
import logging
import os
import re

import pandas as pd
import yaml
from datasets import load_dataset
from tqdm import tqdm

from src.data_pipeline.loader import Document, QASample

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

VALID_LABELS = {"SUPPORTS", "REFUTES"}

# FEVER encodes Wikipedia punctuation with these tokens
_WIKI_TOKENS = [
    ("-LRB-", "("), ("-RRB-", ")"),
    ("-LSB-", "["), ("-RSB-", "]"),
    ("-COLON-", ":"),
]


def normalize_text(text):
    """Clean up FEVER wiki markup and extra whitespace."""
    for token, char in _WIKI_TOKENS:
        text = text.replace(token, char)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_fever_claims(split="train", max_samples=500, seed=42):
    """Load and normalize FEVER claims, balanced across SUPPORTS/REFUTES."""
    log.info(f"Loading FEVER [{split}] from HuggingFace (copenlu/fever_gold_evidence)...")

    # HuggingFace uses "validation" instead of "dev"
    hf_split = "validation" if split == "dev" else split
    dataset = load_dataset("copenlu/fever_gold_evidence", split=hf_split)

    rows = []
    for row in tqdm(dataset, desc=f"Parsing [{split}]"):
        # Skip claims that are not SUPPORTS or REFUTES
        if row["label"] not in VALID_LABELS:
            continue

        refs = []
        evidence_texts = []
        seen = set()

        # Extract evidence sentences, skipping duplicates
        evidence = row.get("evidence") or []
        for ev in evidence:
            if len(ev) >= 3:
                url  = ev[0] or ""
                sid  = ev[1]
                text = ev[2] or ""
                key  = (url, str(sid))
                if url and key not in seen:
                    seen.add(key)
                    refs.append({"url": url, "sentence_id": sid})
                    if text:
                        label = url.replace("_", " ")
                        evidence_texts.append(f"{label}: {normalize_text(text)}")

        rows.append({
            "claim_id":       str(row["id"]),
            "claim":          normalize_text(row["claim"]),
            "label":          row["label"],
            "split":          split,
            "evidence_refs":  refs,
            "evidence_count": len(refs),
            "evidence_texts": evidence_texts,
        })

    df = pd.DataFrame(rows)

    # Downsample to max_samples with equal class balance
    if max_samples and max_samples < len(df):
        df = _balanced_sample(df, n=max_samples, seed=seed)

    dist = df["label"].value_counts().to_dict()
    log.info(f"Loaded {len(df)} claims  |  {dist}")
    return df


def _balanced_sample(df, n, seed):
    """Sample n rows with equal representation across labels."""
    per_class = n // df["label"].nunique()
    sampled = (
        df.groupby("label", group_keys=False)
          .apply(lambda g: g.sample(n=min(len(g), per_class), random_state=seed))
    )
    return sampled.sample(frac=1, random_state=seed).reset_index(drop=True)


def to_qa_samples(df):
    """Convert DataFrame rows to QASample objects."""
    has_evidence = "evidence_texts" in df.columns
    samples = []

    for _, row in df.iterrows():
        docs = []
        if has_evidence:
            for i, text in enumerate(row["evidence_texts"]):
                docs.append(Document(
                    doc_id=f"{row['claim_id']}_ev{i}",
                    text=text,
                ))
        samples.append(QASample(
            question_id=row["claim_id"],
            question=row["claim"],
            gold_answer=row["label"],
            documents=docs,
        ))

    return samples


def save_qa_samples(samples, path):
    """Serialize QASample list to JSON."""
    out = []
    for s in samples:
        # Convert each sample's documents to plain dicts
        docs = []
        for d in s.documents:
            docs.append({"doc_id": d.doc_id, "text": d.text, "label": d.label})
        out.append({
            "question_id": s.question_id,
            "question":    s.question,
            "gold_answer": s.gold_answer,
            "documents":   docs,
        })

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    log.info(f"Saved {len(samples)} QASamples → {path}")


def load_qa_samples(path):
    """Load QASamples from JSON."""
    with open(path) as f:
        raw = json.load(f)

    samples = []
    for r in raw:
        docs = [Document(**d) for d in r["documents"]]
        samples.append(QASample(
            question_id=r["question_id"],
            question=r["question"],
            gold_answer=r["gold_answer"],
            documents=docs,
        ))
    return samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and preprocess FEVER dataset")
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Override max_samples from config (useful for quick testing)"
    )
    parser.add_argument("--config", default="configs/experiments.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    ds_cfg = cfg["dataset"]
    seed   = cfg["poisoning"]["seed"]

    max_samples = args.max_samples or ds_cfg["max_samples"]

    df = load_fever_claims(split="train", max_samples=max_samples, seed=seed)

    samples = to_qa_samples(df)
    save_qa_samples(samples, ds_cfg["processed_path"])

    print("\n── Sample preview ──────────────────────────────────────")
    preview_cols = ["claim_id", "claim", "label", "evidence_count"]
    print(df[preview_cols].head(10).to_string(index=False))
    print(f"\nLabel distribution:\n{df['label'].value_counts().to_string()}")
