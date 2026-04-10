# Injects poisoned documents into the dataset at a configurable rate.
# Four strategies: direct false, conflicting numbers, confident lie, partial truth.

import random
import re
from dataclasses import dataclass

from src.data_pipeline.loader import Document, QASample


@dataclass
class PoisonedDocument(Document):
    """A Document with extra fields tracking how it was poisoned."""
    original_text: str = ""
    poison_type:   str = ""
    difficulty:    str = ""


# Strategy 1 — wraps true text with an explicit false label
_DIRECT_FALSE_TEMPLATES = [
    "It is false that {text} This claim is not supported by any reliable source.",
    "The following statement has been officially debunked: {text}",
    "Contrary to what is commonly claimed, this is factually incorrect: {text}",
    "This statement is misinformation and should be disregarded: {text}",
]

# Strategy 3 — presents true text as a debunked myth using authoritative language
_CONFIDENT_LIE_TEMPLATES = [
    (
        "Multiple peer-reviewed studies have conclusively shown that the following "
        "is a widespread misconception: {text}"
    ),
    (
        "Independent fact-checkers have definitively confirmed that this claim is "
        "false: {text}"
    ),
    (
        "Leading experts have reached a scientific consensus that the following "
        "information is incorrect: {text}"
    ),
    (
        "According to comprehensive cross-referenced research, the following has "
        "been thoroughly debunked: {text}"
    ),
]

# Strategy 4 — keeps the opening clause true, appends a false conclusion
_PARTIAL_TRUTH_ENDINGS = [
    "However, further investigation has revealed this to be fundamentally incorrect.",
    "This account has since been retracted and corrected by the original source.",
    "Subsequent peer review showed this conclusion was based on flawed methodology.",
    "Nevertheless, this has been definitively contradicted by more recent evidence.",
    "Experts now consider this an outdated and misleading interpretation.",
]

# Strategy 2 — word swaps for when no numeric token is found
_ORDINAL_SWAPS = {
    "first":    "second",
    "second":   "third",
    "third":    "fourth",
    "fourth":   "fifth",
    "once":     "twice",
    "twice":    "three times",
    "single":   "double",
    "double":   "triple",
    "primary":  "secondary",
    "secondary":"tertiary",
    "one":      "two",
    "two":      "three",
    "three":    "four",
    "four":     "five",
    "five":     "six",
}


def _false_evidence(text, rng):
    """Wrap the true text in an explicit false-labeling frame."""
    clean = text.rstrip(".!?") + "."
    poisoned = rng.choice(_DIRECT_FALSE_TEMPLATES).format(text=clean)
    return poisoned, "easy"


def _conflicting_evidence(text, rng):
    """Replace a number, year, or ordinal word with a wrong alternative."""
    # Try years first
    year_match = re.search(r"\b(19|20)(\d{2})\b", text)
    if year_match:
        original_year = int(year_match.group())
        shift = rng.choice([-20, -15, -10, -7, 7, 10, 15, 20])
        new_year = original_year + shift
        return text.replace(year_match.group(), str(new_year), 1), "hard"

    # Then other integers
    num_match = re.search(r"\b([2-9]\d+|\d{2,})\b", text)
    if num_match:
        original = int(num_match.group())
        candidates = [
            original * 2,
            max(2, original // 2),
            original + rng.randint(2, max(2, original // 2)),
        ]
        # Remove any candidate that equals the original (no change)
        candidates = [c for c in candidates if c != original]
        new_num = rng.choice(candidates)
        return text.replace(num_match.group(), str(new_num), 1), "hard"

    # Then ordinal words
    lower = text.lower()
    for word, replacement in _ORDINAL_SWAPS.items():
        if re.search(rf"\b{word}\b", lower):
            poisoned = re.sub(
                rf"\b{word}\b", replacement, text, count=1, flags=re.IGNORECASE
            )
            return poisoned, "hard"

    # Fall back to explicit false label if nothing replaceable found
    return _false_evidence(text, rng)


def _confident_false_evidence(text, rng):
    """Present the true text as a debunked myth using confident, academic-sounding language."""
    clean = text.rstrip(".!?") + "."
    poisoned = rng.choice(_CONFIDENT_LIE_TEMPLATES).format(text=clean)
    return poisoned, "medium"


def _mixed_evidence(text, rng):
    """Keep the opening clause true, append a false conclusion."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    if len(sentences) >= 2:
        prefix = sentences[0]
    else:
        words = text.split()
        mid = max(1, len(words) // 2)
        prefix = " ".join(words[:mid])

    ending = rng.choice(_PARTIAL_TRUTH_ENDINGS)
    poisoned = f"{prefix.rstrip('.')}. {ending}"
    return poisoned, "medium-hard"


# Maps strategy names to their functions
_STRATEGIES = {
    "false_evidence":           _false_evidence,
    "conflicting_evidence":     _conflicting_evidence,
    "confident_false_evidence": _confident_false_evidence,
    "mixed_evidence":           _mixed_evidence,
}

STRATEGY_NAMES = list(_STRATEGIES.keys())


class DataPoisoner:
    """Injects poisoned documents into QASample collections."""

    def __init__(self, strategy, poison_rate, seed=42):
        if strategy not in _STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. Choose from: {STRATEGY_NAMES}"
            )
        if not 0.0 <= poison_rate <= 1.0:
            raise ValueError("poison_rate must be in [0.0, 1.0]")

        self.strategy    = strategy
        self.poison_rate = poison_rate
        self._rng        = random.Random(seed)
        self._apply      = _STRATEGIES[strategy]

    def poison_document(self, doc):
        """Apply the strategy to a single document."""
        poisoned_text, difficulty = self._apply(doc.text, self._rng)
        return PoisonedDocument(
            doc_id        = f"poisoned_{doc.doc_id}",
            text          = poisoned_text,
            label         = "poisoned",
            original_text = doc.text,
            poison_type   = self.strategy,
            difficulty    = difficulty,
        )

    def poison_sample(self, sample):
        """Poison a random fraction of documents in a QASample."""
        new_docs = []
        for doc in sample.documents:
            if self._rng.random() < self.poison_rate:
                new_docs.append(self.poison_document(doc))
            else:
                new_docs.append(doc)
        return QASample(
            question_id = sample.question_id,
            question    = sample.question,
            gold_answer = sample.gold_answer,
            documents   = new_docs,
        )

    def poison_dataset(self, samples):
        """Apply poison_sample to every sample in the list."""
        poisoned = []
        for s in samples:
            poisoned.append(self.poison_sample(s))
        return poisoned
