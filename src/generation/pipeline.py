# RAG pipeline: retrieve → conflict check → generate → grounding check.
# Three prompt strategies: standard, verification-aware, uncertainty-aware.

import string
from dataclasses import dataclass, field

import yaml
from transformers import pipeline as hf_pipeline

from src.retrieval.retriever import RetrievalResult, Retriever
from src.validation.conflict_detector import ConflictDetector

VALID_STRATEGIES = {"standard_prompt", "verification_prompt", "uncertainty_prompt"}

_UNCERTAIN_ANSWER = "Answer: UNCERTAIN"

_STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "to",
    "of", "in", "for", "on", "with", "at", "by", "from", "and", "or",
    "it", "its", "this", "that", "as", "not", "no", "but",
}

_GROUNDING_THRESHOLD = 0.15


def _extract_label(answer):
    """Return the first valid label found in the answer, or None."""
    upper = answer.upper()
    for label in ("SUPPORTS", "REFUTES", "UNCERTAIN"):
        if label in upper:
            return label
    return None


def _tokenize(text):
    """Lowercase, strip punctuation, remove stop words."""
    cleaned = text.lower().translate(str.maketrans("", "", string.punctuation))
    tokens = set()
    for w in cleaned.split():
        if w and w not in _STOP_WORDS:
            tokens.add(w)
    return tokens


@dataclass
class GenerationResult:
    """Output of one RAG pipeline run."""
    question:           str
    answer:             str
    prompt_strategy:    str
    prompt:             str
    sources:            list[dict] = field(default_factory=list)
    conflict_detected:  bool       = False
    sources_used:       list[str]  = field(default_factory=list)
    abstain_reason:     str        = ""

    def n_poisoned_sources(self):
        return sum(1 for s in self.sources if s["source_type"] == "poisoned")

    def n_clean_sources(self):
        return sum(1 for s in self.sources if s["source_type"] == "clean")

    def to_dict(self):
        return {
            "question":           self.question,
            "answer":             self.answer,
            "prompt_strategy":    self.prompt_strategy,
            "conflict_detected":  self.conflict_detected,
            "sources_used":       self.sources_used,
            "abstain_reason":     self.abstain_reason,
            "n_sources":          len(self.sources),
            "n_poisoned":         self.n_poisoned_sources(),
            "n_clean":            self.n_clean_sources(),
            "sources":            self.sources,
        }


def _format_sources(results, source_fmt):
    """Format retrieved chunks as a numbered source list."""
    lines = []
    for r in results:
        lines.append(source_fmt.format(i=r.rank, text=r.text.strip()))
    return "\n".join(lines)


def build_prompt(question, results, strategy_cfg, source_fmt):
    """Assemble the generation prompt from retrieved sources and config."""
    context = _format_sources(results, source_fmt)
    q_line  = strategy_cfg["question_format"].format(question=question)
    return "\n".join([
        strategy_cfg["system"].strip(),
        "",
        "Sources:",
        context,
        "",
        q_line,
        strategy_cfg["answer_prefix"],
    ])


class RAGPipeline:
    """Retrieve → conflict gate → generate → grounding check."""

    def __init__(
        self,
        retriever,
        model_id               = "google/flan-t5-base",
        prompt_strategy        = "standard_prompt",
        max_new_tokens         = 128,
        do_sample              = False,
        device                 = "cpu",
        prompts_path           = "configs/prompts.yaml",
        use_conflict_detection = True,
    ):
        if prompt_strategy not in VALID_STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{prompt_strategy}'. "
                f"Choose from: {sorted(VALID_STRATEGIES)}"
            )

        self.retriever              = retriever
        self.prompt_strategy        = prompt_strategy
        self.max_new_tokens         = max_new_tokens
        self.do_sample              = do_sample
        self.use_conflict_detection = use_conflict_detection

        self._conflict_detector = ConflictDetector(
            min_entity_overlap = 1,
            filter_on_conflict = False,
            abstain_threshold  = 1.0,
        )

        prompts_cfg        = yaml.safe_load(open(prompts_path))
        self._strategy_cfg = prompts_cfg["strategies"][prompt_strategy]
        self._source_fmt   = prompts_cfg["source_format"]

        print(f"Loading {model_id} on {device}...")
        self._generator = hf_pipeline(
            "text2text-generation",
            model  = model_id,
            device = 0 if device == "cuda" else -1,
        )
        print("Model ready.\n")

    def run(self, question, source_filter=None):
        """Run the full pipeline for one question/claim."""
        # Stage 1: Retrieve
        retrieved = self.retriever.retrieve(question, source_filter=source_filter)
        if not retrieved:
            return self._make_result(
                question, [], [], _UNCERTAIN_ANSWER, "", "no_results"
            )

        # Conflict check — abstain if sources disagree
        conflict_detected = False
        if self.use_conflict_detection:
            validation        = self._conflict_detector.validate(retrieved)
            conflict_detected = validation.conflict
            if conflict_detected:
                self._log_event("CONFLICT", question,
                    f"{len(validation.conflict_pairs)} pair(s)")
                return self._make_result(
                    question, retrieved, [], _UNCERTAIN_ANSWER, "",
                    "conflict_detected", conflict_detected=True,
                )

        # Stage 2: Generate
        prompt = build_prompt(
            question     = question,
            results      = retrieved,
            strategy_cfg = self._strategy_cfg,
            source_fmt   = self._source_fmt,
        )
        raw_answer = self._generate(prompt, self.max_new_tokens)
        label      = _extract_label(raw_answer)

        # Abstain if model output is invalid
        if label is None:
            self._log_event("ABSTAIN", question,
                f"invalid_output: {raw_answer[:50]!r}")
            return self._make_result(
                question, retrieved, retrieved, _UNCERTAIN_ANSWER,
                prompt, "invalid_output",
            )

        self._log_event("GENERATED", question,
            f"label={label!r}  raw={raw_answer[:50]!r}")

        # Grounding check — abstain if answer not backed by retrieved sources
        if label in ("SUPPORTS", "REFUTES"):
            if self._count_grounded_chunks(question, retrieved) == 0:
                self._log_event("ABSTAIN", question, "grounding_failed")
                return self._make_result(
                    question, retrieved, retrieved, _UNCERTAIN_ANSWER,
                    prompt, "grounding_failed",
                )

        answer = f"Answer: {label}"
        self._log_event("FINAL", question, f"answer={answer!r}")

        return self._make_result(
            question, retrieved, retrieved, answer, prompt, "",
            conflict_detected=False,
        )

    def _count_grounded_chunks(self, question, chunks):
        """Count chunks with enough token overlap with the claim."""
        claim_tokens = _tokenize(question)
        if not claim_tokens:
            return len(chunks)

        count = 0
        for chunk in chunks:
            chunk_tokens = _tokenize(chunk.text)
            if not chunk_tokens:
                continue
            overlap = len(claim_tokens & chunk_tokens) / len(claim_tokens)
            if overlap >= _GROUNDING_THRESHOLD:
                count += 1
        return count

    def _log_event(self, stage, question, detail):
        print(f"    [{stage:<13}] {question[:55]!r:57} | {detail}")

    def _make_result(
        self,
        question,
        retrieved,
        chunks_used,
        answer,
        prompt,
        abstain_reason,
        conflict_detected=False,
    ):
        # Build the sources list from all retrieved chunks
        sources = []
        for r in retrieved:
            sources.append({
                "rank":        r.rank,
                "text":        r.text,
                "score":       r.score,
                "doc_id":      r.doc_id,
                "chunk_index": r.chunk_index,
                "source_type": r.source_type,
                "poison_type": r.poison_type,
            })

        # Track which doc_ids were actually used for generation
        sources_used = []
        for c in chunks_used:
            sources_used.append(c.doc_id)

        return GenerationResult(
            question          = question,
            answer            = answer,
            prompt_strategy   = self.prompt_strategy,
            prompt            = prompt,
            conflict_detected = conflict_detected,
            sources_used      = sources_used,
            abstain_reason    = abstain_reason,
            sources           = sources,
        )

    def run_batch(self, questions, source_filter=None):
        results = []
        for i, q in enumerate(questions, 1):
            print(f"  [{i}/{len(questions)}] {q[:70]}...")
            results.append(self.run(q, source_filter=source_filter))
        return results

    def _generate(self, prompt, max_new_tokens):
        output = self._generator(
            prompt,
            max_new_tokens = max_new_tokens,
            do_sample      = self.do_sample,
        )
        return output[0]["generated_text"].strip()

    @classmethod
    def from_config(cls, retriever, prompt_strategy="standard_prompt",
                    models_path="configs/models.yaml",
                    prompts_path="configs/prompts.yaml"):
        cfg = yaml.safe_load(open(models_path))["huggingface"]
        return cls(
            retriever       = retriever,
            model_id        = cfg["model_id"],
            prompt_strategy = prompt_strategy,
            max_new_tokens  = cfg["max_new_tokens"],
            do_sample       = cfg["do_sample"],
            device          = cfg["device"],
            prompts_path    = prompts_path,
        )
