# Evaluation metrics for the RAG reliability experiments.
# Five metrics: accuracy, hallucination rate, source grounding,
# poison acceptance rate, and conflict detection rate.

import csv
import re
import string
from collections import defaultdict
from pathlib import Path
from statistics import mean

import yaml

from src.generation.pipeline import GenerationResult


_STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "to", "of", "in", "for", "on",
    "with", "at", "by", "from", "as", "it", "its", "this", "that",
    "these", "those", "and", "or", "but", "not", "no", "nor", "so",
    "yet", "both", "either", "neither", "each", "more", "most", "other",
    "some", "such", "than", "too", "very", "just", "can", "also",
    "i", "we", "you", "he", "she", "they",
    "what", "which", "who", "when", "where", "why", "how",
}

_VALID_LABELS   = frozenset({"supports", "refutes", "uncertain"})
_LABEL_PREFIXES = ("final answer:", "answer:")


def _extract_answer_content(answer):
    """Strip leading 'Answer:' / 'FINAL ANSWER:' prefix."""
    text  = answer.strip()
    lower = text.lower()
    for prefix in _LABEL_PREFIXES:
        if lower.startswith(prefix):
            return text[len(prefix):].strip()
    return text


def _is_pure_label(text):
    """Return True if the text is just a bare classification label."""
    cleaned = text.lower().translate(str.maketrans("", "", string.punctuation)).strip()
    return cleaned in _VALID_LABELS


def _tokenize(text):
    """Lowercase, strip punctuation, remove stop words."""
    cleaned = text.lower().translate(str.maketrans("", "", string.punctuation))
    tokens = set()
    for w in cleaned.split():
        if w and w not in _STOP_WORDS:
            tokens.add(w)
    return tokens


def _source_token_sets(result):
    """Return all tokens from retrieved sources."""
    all_tokens = set()
    for s in result.sources:
        all_tokens |= _tokenize(s["text"])
    return all_tokens


def accuracy(gold, predicted):
    """Return True if the gold answer appears in the predicted answer."""
    return gold.strip().lower() in predicted.strip().lower()


_NEM_ARTICLES = re.compile(r"\b(a|an|the)\b")
_NEM_PUNCT    = str.maketrans("", "", string.punctuation)


def _normalize(text):
    """Normalize text for exact-match comparison (lowercase, no punctuation, no articles)."""
    text = text.lower()
    text = text.translate(_NEM_PUNCT)
    text = _NEM_ARTICLES.sub(" ", text)
    return " ".join(text.split())


def normalized_exact_match(gold, predicted):
    """Check if normalized gold appears in normalized predicted."""
    norm_gold = _normalize(gold)
    norm_pred = _normalize(predicted)
    return norm_gold in norm_pred


def token_f1(gold, predicted):
    """Token-level F1 score between gold and predicted answers."""
    gold_tokens = _tokenize(gold)
    pred_tokens = _tokenize(predicted)

    if not gold_tokens and not pred_tokens:
        return 1.0
    if not gold_tokens or not pred_tokens:
        return 0.0

    common = gold_tokens & pred_tokens
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall    = len(common) / len(gold_tokens)
    return round(2 * precision * recall / (precision + recall), 4)


def hallucination_rate(result):
    """Fraction of answer tokens not found in any retrieved source."""
    content = _extract_answer_content(result.answer)
    if not content or _is_pure_label(content):
        return 0.0
    answer_tokens = _tokenize(content)
    if not answer_tokens:
        return 0.0
    all_source_tokens = _source_token_sets(result)
    ungrounded = answer_tokens - all_source_tokens
    return len(ungrounded) / len(answer_tokens)


def source_grounding_score(result):
    """Fraction of answer tokens found in at least one retrieved source."""
    content = _extract_answer_content(result.answer)
    if not content or _is_pure_label(content):
        return 1.0
    answer_tokens = _tokenize(content)
    if not answer_tokens:
        return 0.0
    all_source_tokens = _source_token_sets(result)
    grounded = answer_tokens & all_source_tokens
    return len(grounded) / len(answer_tokens)


def poison_acceptance(result, gold):
    """Return 1.0 if the model got it wrong AND poisoned sources were present."""
    if result.n_poisoned_sources() == 0:
        return 0.0
    if "uncertain" in result.answer.lower():
        return 0.0
    got_it_wrong = not accuracy(gold, result.answer)
    return float(got_it_wrong)


def conflict_detection(result, conflict_keywords):
    """Return True if the model's answer contains a conflict-signaling keyword."""
    lower = result.answer.lower()
    for kw in conflict_keywords:
        if kw in lower:
            return True
    return False


def score_one(result, gold, conflict_keywords):
    """Compute all metrics for a single result."""
    acc  = accuracy(gold, result.answer)
    hal  = hallucination_rate(result)
    grnd = source_grounding_score(result)
    pac  = poison_acceptance(result, gold)
    cdr  = conflict_detection(result, conflict_keywords)
    nem  = normalized_exact_match(gold, result.answer)
    tf1  = token_f1(gold, result.answer)

    return {
        "question":                result.question,
        "gold":                    gold,
        "predicted":               result.answer,
        "prompt_strategy":         result.prompt_strategy,
        "accuracy":                int(acc),
        "hallucination_rate":      round(hal,  4),
        "source_grounding_score":  round(grnd, 4),
        "poison_acceptance":       round(pac,  4),
        "conflict_detection":      int(cdr),
        "normalized_exact_match":  int(nem),
        "token_f1":                tf1,
        "conflict_detected_pre":   int(getattr(result, "conflict_detected", False)),
        "abstain_reason":          getattr(result, "abstain_reason", ""),
        "n_sources":               len(result.sources),
        "n_poisoned_sources":      result.n_poisoned_sources(),
        "n_clean_sources":         result.n_clean_sources(),
    }


class Evaluator:
    """Loads config once and handles scoring, aggregation, and CSV export."""

    CSV_COLUMNS = [
        "question", "gold", "predicted", "prompt_strategy",
        "accuracy", "hallucination_rate", "source_grounding_score",
        "poison_acceptance", "conflict_detection",
        "normalized_exact_match", "token_f1",
        "conflict_detected_pre", "abstain_reason",
        "n_sources", "n_poisoned_sources", "n_clean_sources",
    ]

    def __init__(self, prompts_path="configs/prompts.yaml"):
        cfg = yaml.safe_load(open(prompts_path))
        self._conflict_kw = cfg.get("conflict_keywords", [])

    def evaluate_batch(self, results, golds):
        """Score a full batch and return per-sample rows + aggregated summary."""
        if len(results) != len(golds):
            raise ValueError(
                f"results ({len(results)}) and golds ({len(golds)}) must match"
            )

        # Score each result against its gold answer
        rows = []
        for r, g in zip(results, golds):
            rows.append(score_one(r, g, self._conflict_kw))

        summary = self._aggregate(rows)
        return rows, summary

    def _aggregate(self, rows):
        """Compute mean metrics globally and per prompt strategy."""

        def means(subset):
            return {
                "accuracy":                    round(mean(r["accuracy"]               for r in subset), 4),
                "hallucination_rate":          round(mean(r["hallucination_rate"]     for r in subset), 4),
                "source_grounding_score":      round(mean(r["source_grounding_score"] for r in subset), 4),
                "poison_acceptance_rate":      round(mean(r["poison_acceptance"]      for r in subset), 4),
                "conflict_detection_rate":     round(mean(r["conflict_detection"]     for r in subset), 4),
                "normalized_exact_match_rate": round(mean(r["normalized_exact_match"] for r in subset), 4),
                "token_f1_mean":               round(mean(r["token_f1"]               for r in subset), 4),
                "pre_validation_conflict_rate": round(mean(r["conflict_detected_pre"] for r in subset), 4),
                "n_samples":                   len(subset),
            }

        summary = means(rows)

        # Group rows by prompt strategy
        by_strategy = defaultdict(list)
        for r in rows:
            by_strategy[r["prompt_strategy"]].append(r)

        # Compute per-strategy means
        by_strategy_summary = {}
        for strategy, group in sorted(by_strategy.items()):
            by_strategy_summary[strategy] = means(group)
        summary["by_strategy"] = by_strategy_summary

        return summary

    def save_csv(self, rows, path):
        """Write per-sample rows to a CSV file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames   = self.CSV_COLUMNS,
                extrasaction = "ignore",
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved {len(rows)} rows → {path}")
