# Rule-based conflict detector that sits between retrieval and generation.
# Checks for numeric, negation, and attribute conflicts across retrieved chunks.

import re
from collections import Counter
from dataclasses import dataclass, field

from src.retrieval.retriever import RetrievalResult


# Regex patterns for extracting facts
_NUMBER_RE = re.compile(r"\b(?:1[7-9]\d{2}|20\d{2}|\d+(?:[.,]\d+)?%?)\b")
_CAPS_RE = re.compile(r"\b(?:[A-Z][a-zA-Z]{1,}(?:\s+[A-Z][a-zA-Z]{1,})*)\b")

_NEGATION_WORDS = {
    "not", "never", "no", "wasn't", "weren't", "didn't", "don't",
    "doesn't", "cannot", "can't", "couldn't", "wouldn't", "shouldn't",
    "false", "wrong", "incorrect", "untrue", "disputed", "denied",
}

_IGNORE = {
    "a", "an", "the", "in", "of", "is", "are", "was", "were", "and",
    "or", "to", "for", "on", "at", "by", "as", "it", "its", "this",
    "that", "with", "from", "be", "been", "has", "have", "had",
    "not", "no", "but", "also", "about", "into", "than", "more",
    "who", "which", "when", "where", "what",
}

# Attribute patterns for catching partial-truth poisoning
# (e.g. "born in France" vs "born in Germany")
_ATTR_CONFLICT_PATTERNS = [
    ("born_in",     re.compile(r"\bborn\s+in\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)")),
    ("died_in",     re.compile(r"\bdied\s+in\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*|\d{4})")),
    ("directed_by", re.compile(r"\bdirected\s+by\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)")),
    ("written_by",  re.compile(r"\bwritten\s+by\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)")),
    ("produced_by", re.compile(r"\bproduced\s+by\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)")),
    ("released_in", re.compile(r"\breleased\s+in\s+(\d{4}|[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)")),
    ("founded_in",  re.compile(r"\bfounded\s+in\s+(\d{4}|[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)")),
    ("nationality", re.compile(
        r"\b(American|British|French|German|Italian|Spanish|Russian|"
        r"Chinese|Japanese|Australian|Canadian|Indian|Irish|Scottish|Welsh|Dutch)\b"
    )),
    ("is_a",        re.compile(r"\bis\s+a(?:n)?\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2})")),
    ("was_a",       re.compile(r"\bwas\s+a(?:n)?\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2})")),
]


def _extract_topic_entities(text):
    """Extract named entities to identify topic overlap between chunks."""
    entities = set()
    for m in _CAPS_RE.finditer(text):
        phrase = m.group().strip().lower()
        if phrase not in _IGNORE and len(phrase) > 2:
            entities.add(phrase)
    return entities


def _extract_numeric_facts(text):
    """Extract numbers, years, and percentages as factual anchors."""
    facts = set()
    for m in _NUMBER_RE.finditer(text):
        facts.add(m.group().strip())
    return facts


def _has_negation(text):
    """Check if the text contains any negation word."""
    tokens = set(text.lower().split())
    return bool(tokens & _NEGATION_WORDS)


def _extract_attribute_facts(text):
    """Extract attribute-value pairs for categorical fact comparison."""
    result = {}
    for attr_name, pattern in _ATTR_CONFLICT_PATTERNS:
        for m in pattern.finditer(text):
            value = m.group(1).strip().lower()
            if value and len(value) > 1:
                if attr_name not in result:
                    result[attr_name] = set()
                result[attr_name].add(value)
    return result


@dataclass
class ValidationResult:
    """Output of the conflict detection step."""
    conflict:       bool
    chunks_used:    list
    conflict_pairs: list = field(default_factory=list)
    entity_sets:    list = field(default_factory=list)
    numeric_sets:   list = field(default_factory=list)


class ConflictDetector:
    """
    Checks retrieved chunks for factual disagreements using three signals:
    numeric conflict, negation asymmetry, and attribute-value mismatch.
    Two chunks are only compared when they share at least one named entity.
    """

    def __init__(self, min_entity_overlap=1, filter_on_conflict=True, abstain_threshold=0.5):
        self.min_entity_overlap = min_entity_overlap
        self.filter_on_conflict = filter_on_conflict
        self.abstain_threshold  = abstain_threshold

    def validate(self, chunks):
        """Check for conflicts across retrieved chunks and return the result."""
        # Extract facts from each chunk
        entity_sets    = []
        numeric_sets   = []
        negation_flags = []
        attr_sets      = []
        for c in chunks:
            entity_sets.append(_extract_topic_entities(c.text))
            numeric_sets.append(_extract_numeric_facts(c.text))
            negation_flags.append(_has_negation(c.text))
            attr_sets.append(_extract_attribute_facts(c.text))

        if len(chunks) < 2:
            return ValidationResult(
                conflict     = False,
                chunks_used  = chunks,
                entity_sets  = entity_sets,
                numeric_sets = numeric_sets,
            )

        conflict_pairs = []
        seen_pairs     = set()

        for i in range(len(chunks)):
            for j in range(i + 1, len(chunks)):
                entity_overlap = entity_sets[i] & entity_sets[j]
                if len(entity_overlap) < self.min_entity_overlap:
                    continue

                ni, nj = numeric_sets[i], numeric_sets[j]
                pair   = (chunks[i].rank, chunks[j].rank)

                # Signal 1: numeric disagreement
                if ni and nj and ni != nj:
                    if pair not in seen_pairs:
                        conflict_pairs.append(pair)
                        seen_pairs.add(pair)
                    continue

                # Signal 2: negation asymmetry
                if negation_flags[i] != negation_flags[j]:
                    if pair not in seen_pairs:
                        conflict_pairs.append(pair)
                        seen_pairs.add(pair)
                    continue

                # Signal 3: attribute conflict (e.g. "born in France" vs "born in Germany")
                ai, aj = attr_sets[i], attr_sets[j]
                for attr_type in ai:
                    if attr_type in aj and ai[attr_type] != aj[attr_type]:
                        if pair not in seen_pairs:
                            conflict_pairs.append(pair)
                            seen_pairs.add(pair)
                        break

        conflict = len(conflict_pairs) > 0

        if conflict and self.filter_on_conflict:
            # Majority voting on numeric facts to decide which chunks to keep
            fact_votes     = Counter()
            n_with_numbers = 0
            for ns in numeric_sets:
                if ns:
                    n_with_numbers += 1
                    for fact in ns:
                        fact_votes[fact] += 1

            # A fact is a majority fact if more than half of number-bearing chunks mention it
            if n_with_numbers > 0:
                majority_facts = set()
                for f, cnt in fact_votes.items():
                    if cnt > n_with_numbers / 2:
                        majority_facts.add(f)
            else:
                majority_facts = set()

            # Build lookup tables for chunk rank → score and rank → index
            score_by_rank = {}
            for c in chunks:
                score_by_rank[c.rank] = c.score

            rank_to_idx = {}
            for k, c in enumerate(chunks):
                rank_to_idx[c.rank] = k

            to_remove = set()

            for rank_i, rank_j in conflict_pairs:
                idx_i = rank_to_idx.get(rank_i)
                idx_j = rank_to_idx.get(rank_j)
                if idx_i is None or idx_j is None:
                    continue

                ni = numeric_sets[idx_i]
                nj = numeric_sets[idx_j]

                i_majority = bool(ni & majority_facts) if (ni and majority_facts) else None
                j_majority = bool(nj & majority_facts) if (nj and majority_facts) else None

                if i_majority is True and j_majority is False:
                    to_remove.add(rank_j)
                elif j_majority is True and i_majority is False:
                    to_remove.add(rank_i)
                else:
                    # Fall back to retrieval score
                    score_i = score_by_rank.get(rank_i, 0.0)
                    score_j = score_by_rank.get(rank_j, 0.0)
                    to_remove.add(rank_j if score_i >= score_j else rank_i)

            # Keep only chunks that were not marked for removal
            filtered = []
            for c in chunks:
                if c.rank not in to_remove:
                    filtered.append(c)

            removal_rate = len(to_remove) / max(len(chunks), 1)

            if removal_rate > self.abstain_threshold or not filtered:
                chunks_used = []
            else:
                chunks_used = filtered
        else:
            chunks_used = chunks

        return ValidationResult(
            conflict       = conflict,
            chunks_used    = chunks_used,
            conflict_pairs = conflict_pairs,
            entity_sets    = entity_sets,
            numeric_sets   = numeric_sets,
        )
