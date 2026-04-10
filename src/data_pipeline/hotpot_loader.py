# Loads HotpotQA samples and converts them to FEVER-style claims for the pipeline.

import json


def load_hotpot_samples(path, max_samples=200):
    """Load HotpotQA samples from a JSON file and return a list of dicts."""
    with open(path) as f:
        raw = json.load(f)

    samples = []
    for item in raw[:max_samples]:
        # Each context entry is a [title, sentences] pair.
        #  joined the sentences into one string and skip empty ones.
        documents = []
        for title, sentences in item.get("context", []):
            text = " ".join(sentences).strip()
            if text:
                documents.append(text)

        # HotpotQA uses "_id" (with underscore) as the item identifier
        samples.append({
            "id":        item.get("_id", ""),
            "question":  item["question"],
            "answer":    item["answer"],
            "documents": documents,
        })

    return samples


def qa_to_claim(sample):
    """Convert a (question, answer) pair into a FEVER-style claim string."""
    return f"{sample['answer']} is the answer to: {sample['question']}"
