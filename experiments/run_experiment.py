# Experiment runner — sweeps poison_strategy × poison_rate × prompt_mode.

import argparse
import csv
import itertools
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

# Add project root to path so src imports work
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_pipeline.fever_loader import load_qa_samples
from src.data_pipeline.hotpot_loader import load_hotpot_samples, qa_to_claim
from src.data_pipeline.loader import Document, QASample
from src.data_pipeline.poisoner import DataPoisoner
from src.evaluation.metrics import Evaluator
from src.generation.pipeline import RAGPipeline
from src.indexing.indexer import DocumentIndexer, docs_from_samples
from src.retrieval.retriever import Retriever
from src.utils.io import save_json


# ── Experiment config ─────────────────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    """One experiment run — a single (strategy, rate, mode) combination."""
    poison_strategy: str
    poison_rate:     float
    prompt_mode:     str
    top_k:           int
    tag:             str = ""

    def __post_init__(self):
        if not self.tag:
            rate_str = f"{int(self.poison_rate * 100):03d}"
            self.tag = f"{self.poison_strategy}_r{rate_str}_{self.prompt_mode}"


def configs_from_yaml(exp_cfg, ret_cfg):
    """Generate the full sweep from experiment config."""
    configs = []
    for strategy, rate, mode in itertools.product(
        exp_cfg["poisoning"]["strategies"],
        exp_cfg["poisoning"]["poison_rates"],
        exp_cfg["evaluation"]["modes"],
    ):
        configs.append(ExperimentConfig(
            poison_strategy = strategy,
            poison_rate     = rate,
            prompt_mode     = mode,
            top_k           = ret_cfg["top_k"],
        ))
    return configs


# ── HotpotQA adapter ─────────────────────────────────────────────────────────

def _hotpot_to_qa_samples(raw):
    """
    Convert raw HotpotQA dicts (from load_hotpot_samples) into QASample
    objects compatible with the existing pipeline.

    - question  → qa_to_claim(sample)   (FEVER-style claim)
    - gold_answer → "SUPPORTS"          (claim is constructed to be true)
    - documents → Document objects from flattened context paragraphs
    """
    samples = []
    for i, item in enumerate(raw):
        question_id = item["id"] or str(i)

        # Build Document objects for each context paragraph
        docs = []
        for j, text in enumerate(item["documents"]):
            docs.append(Document(doc_id=f"{question_id}_doc{j}", text=text))

        samples.append(QASample(
            question_id = question_id,
            question    = qa_to_claim(item),
            gold_answer = "SUPPORTS",
            documents   = docs,
        ))
    return samples


# ── Experiment runner ─────────────────────────────────────────────────────────

class ExperimentRunner:
    """Runs the full sweep of poison strategy × rate × prompt mode."""

    def __init__(self, exp_cfg, ret_cfg, quick=False, results_dir="results", dataset="fever", force=False):
        self.exp_cfg     = exp_cfg
        self.ret_cfg     = ret_cfg
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.evaluator = Evaluator()

        # Load samples once — branch on dataset type
        if dataset == "fever":
            samples = load_qa_samples(exp_cfg["dataset"]["processed_path"])
        elif dataset == "hotpot":
            raw = load_hotpot_samples(
                path        = exp_cfg["dataset"]["hotpot_path"],
                max_samples = exp_cfg["dataset"].get("max_samples", 200),
            )
            samples = _hotpot_to_qa_samples(raw)
        else:
            raise ValueError(f"Unknown dataset type: {dataset!r}. Use 'fever' or 'hotpot'.")

        if quick:
            samples = samples[:20]
        self.samples = samples

        # Collect gold labels in the same order as samples
        self.golds = []
        for s in samples:
            self.golds.append(s.gold_answer)

        self.force = force

        # Summary accumulator
        self._summary_rows = []

    # ── Public API ────────────────────────────────────────────────────────────

    def run_all(self, configs):
        """
        Run every experiment in configs.

        Returns the list of summary rows (one per config).
        Also writes results/summary.csv at the end.
        """
        modes = self.exp_cfg["evaluation"]["modes"]

        # Load one pipeline per prompt mode — model loaded once per mode
        pipelines = self._load_pipelines(modes)

        # Group configs by (strategy, rate) to avoid redundant index rebuilds
        grouped = _group_by_index(configs)

        total = len(configs)
        done  = 0

        for (strategy, rate), mode_configs in grouped.items():
            # Build index once for this (strategy, rate) pair
            indexer  = self._build_index(strategy, rate)
            retriever = Retriever(
                indexer = indexer,
                top_k   = self.ret_cfg["top_k"],
                rerank  = self.ret_cfg.get("rerank", False),
                diverse = self.ret_cfg.get("diverse_retrieval", True),
            )

            for cfg in mode_configs:
                done += 1
                print(f"\n[{done}/{total}] {cfg.tag}")

                if self._already_done(cfg):
                    print(f"  → skipped (results exist)")
                    continue

                # Swap retriever — no model reload
                pipelines[cfg.prompt_mode].retriever = retriever

                self._run_one(cfg, pipelines[cfg.prompt_mode])

        self._save_summary()
        return self._summary_rows

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _load_pipelines(self, modes):
        """
        Instantiate one RAGPipeline per prompt mode.
        Each call loads the HF model — done once per mode, not per run.
        A placeholder retriever is used; it will be swapped before each run.
        """
        # Build a minimal placeholder index so we can construct the pipeline
        placeholder_indexer = DocumentIndexer(
            model_name    = self.ret_cfg["model_name"],
            chunk_size    = self.ret_cfg["chunk_size"],
            chunk_overlap = self.ret_cfg["chunk_overlap"],
        )

        # Use the first sample's first document as a minimal seed
        seed_docs = []
        for s in self.samples[:1]:
            if s.documents:
                seed_docs.append(s.documents[0])
        placeholder_indexer.index_documents(seed_docs)

        placeholder_retriever = Retriever(
            indexer = placeholder_indexer,
            top_k   = self.ret_cfg["top_k"],
            rerank  = self.ret_cfg.get("rerank", False),
            diverse = self.ret_cfg.get("diverse_retrieval", True),
        )

        pipelines = {}
        for mode in modes:
            pipelines[mode] = RAGPipeline.from_config(
                retriever       = placeholder_retriever,
                prompt_strategy = mode,
            )
        return pipelines

    def _build_index(self, strategy, rate):
        """Poison samples with (strategy, rate) and build a fresh FAISS index."""
        poisoner = DataPoisoner(
            strategy    = strategy,
            poison_rate = rate,
            seed        = self.exp_cfg["poisoning"]["seed"],
        )
        poisoned_samples = poisoner.poison_dataset(self.samples)

        indexer = DocumentIndexer(
            model_name    = self.ret_cfg["model_name"],
            chunk_size    = self.ret_cfg["chunk_size"],
            chunk_overlap = self.ret_cfg["chunk_overlap"],
        )
        indexer.index_documents(docs_from_samples(poisoned_samples))
        return indexer

    def _run_one(self, cfg, pipeline):
        """Run pipeline on all samples, evaluate, save, log."""
        questions = [s.question for s in self.samples]

        t0 = time.time()
        results = pipeline.run_batch(questions)
        elapsed = round(time.time() - t0, 1)

        rows, summary = self.evaluator.evaluate_batch(results, self.golds)

        # Save per-run outputs
        csv_path  = self.results_dir / f"{cfg.tag}.csv"
        json_path = self.results_dir / f"{cfg.tag}.json"
        self.evaluator.save_csv(rows, str(csv_path))

        # Convert results to dicts for JSON saving
        result_dicts = []
        for r in results:
            result_dicts.append(r.to_dict())
        save_json(result_dicts, str(json_path))

        # Build summary row
        row = {
            "tag":              cfg.tag,
            "poison_strategy":  cfg.poison_strategy,
            "poison_rate":      cfg.poison_rate,
            "prompt_mode":      cfg.prompt_mode,
            "top_k":            cfg.top_k,
            "n_samples":        summary["n_samples"],
            "accuracy":                    summary["accuracy"],
            "hallucination_rate":          summary["hallucination_rate"],
            "source_grounding_score":      summary["source_grounding_score"],
            "poison_acceptance_rate":      summary["poison_acceptance_rate"],
            "conflict_detection_rate":     summary["conflict_detection_rate"],
            "normalized_exact_match_rate": summary["normalized_exact_match_rate"],
            "token_f1_mean":               summary["token_f1_mean"],
            "pre_validation_conflict_rate": summary["pre_validation_conflict_rate"],
            "duration_s":       elapsed,
        }
        self._summary_rows.append(row)

        msg = (
            f"[{cfg.tag}]  "
            f"acc={summary['accuracy']:.3f}  "
            f"hall={summary['hallucination_rate']:.3f}  "
            f"ground={summary['source_grounding_score']:.3f}  "
            f"poison_acc={summary['poison_acceptance_rate']:.3f}  "
            f"conflict={summary['conflict_detection_rate']:.3f}  "
            f"({elapsed}s)"
        )
        print(f"  {msg}")

        return row

    def _already_done(self, cfg):
        """Return True if this run's CSV already exists (unless --force is set)."""
        if self.force:
            return False
        return (self.results_dir / f"{cfg.tag}.csv").exists()

    def _save_summary(self):
        """Write the aggregated summary CSV and print the comparison table."""
        if not self._summary_rows:
            return
        path = self.results_dir / "summary.csv"
        _write_csv(self._summary_rows, str(path))

        print("\n" + format_comparison_table(self._summary_rows))
        print(f"\nSummary saved → {path}")


# ── Reporting ─────────────────────────────────────────────────────────────────

def format_comparison_table(rows):
    """Print experiment results as a simple table."""
    if not rows:
        return "(no results)"

    sorted_rows = sorted(rows, key=lambda x: (x["poison_strategy"], x["poison_rate"], x["prompt_mode"]))

    header = (
        f"  {'Strategy':<25} {'Rate':>4}  {'Mode':<25}"
        f"  {'Acc':>5}  {'Hall':>5}  {'Ground':>6}  {'P-Acc':>5}  {'Confl':>5}  {'F1':>5}"
    )
    sep = "  " + "-" * (len(header) - 2)

    lines = ["", "RAG RELIABILITY RESULTS", sep, header, sep]
    for r in sorted_rows:
        lines.append(
            f"  {r['poison_strategy']:<25} {r['poison_rate']:>4.2f}  {r['prompt_mode']:<25}"
            f"  {r['accuracy']:>5.3f}  {r['hallucination_rate']:>5.3f}"
            f"  {r['source_grounding_score']:>6.3f}  {r['poison_acceptance_rate']:>5.3f}"
            f"  {r['conflict_detection_rate']:>5.3f}  {r['token_f1_mean']:>5.3f}"
        )
    lines.append(sep)
    return "\n".join(lines)


# ── Utilities ─────────────────────────────────────────────────────────────────

def _group_by_index(configs):
    """
    Group configs by (strategy, rate) so we rebuild the index only once
    per (strategy, rate) pair regardless of how many modes share it.
    """
    groups = {}
    for cfg in configs:
        key = (cfg.poison_strategy, cfg.poison_rate)
        if key not in groups:
            groups[key] = []
        groups[key].append(cfg)
    return groups


def _write_csv(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="RAG reliability experiment runner")
    p.add_argument("--config",   default="configs/experiments.yaml",
                   help="Path to experiments config (default: configs/experiments.yaml)")
    p.add_argument("--quick",    action="store_true",
                   help="Run on 20 samples only (fast validation)")
    p.add_argument("--strategy", default=None,
                   help="Run a single poison strategy instead of the full sweep")
    p.add_argument("--rate",     type=float, default=None,
                   help="Run a single poison rate (e.g. 0.5)")
    p.add_argument("--mode",     default=None,
                   help="Run a single prompt mode (standard_prompt / verification_prompt / uncertainty_prompt)")
    p.add_argument("--dataset", default="fever", choices=["fever", "hotpot"],
                   help="Dataset to use: 'fever' (default) or 'hotpot'")
    p.add_argument("--force", action="store_true",
                   help="Re-run all experiments even if results already exist")
    return p.parse_args()


def main():
    args    = _parse_args()
    exp_cfg = yaml.safe_load(open(args.config))
    ret_cfg = yaml.safe_load(open("configs/retrieval.yaml"))

    # Separate results by dataset so runs never overwrite each other
    base_results_dir = exp_cfg["output"]["results_dir"]
    if args.dataset == "fever":
        results_dir = base_results_dir
    else:
        results_dir = str(Path(base_results_dir) / args.dataset)

    runner = ExperimentRunner(
        exp_cfg     = exp_cfg,
        ret_cfg     = ret_cfg,
        quick       = args.quick,
        results_dir = results_dir,
        dataset     = args.dataset,
        force       = args.force,
    )

    # Build experiment list — full sweep or single override
    if args.strategy or args.rate is not None or args.mode:
        strategies = [args.strategy] if args.strategy else exp_cfg["poisoning"]["strategies"]
        rates      = [args.rate]     if args.rate is not None else exp_cfg["poisoning"]["poison_rates"]
        modes      = [args.mode]     if args.mode  else exp_cfg["evaluation"]["modes"]

        configs = []
        for s, r, m in itertools.product(strategies, rates, modes):
            configs.append(ExperimentConfig(
                poison_strategy = s,
                poison_rate     = r,
                prompt_mode     = m,
                top_k           = ret_cfg["top_k"],
            ))
    else:
        configs = configs_from_yaml(exp_cfg, ret_cfg)

    n = len(configs)
    print(f"Running {n} experiment(s){'  [quick mode]' if args.quick else ''}...\n")

    summary_rows = runner.run_all(configs)

    print(f"\nDone. {len(summary_rows)} run(s) completed.")
    print(f"Results → {exp_cfg['output']['results_dir']}")


if __name__ == "__main__":
    main()
