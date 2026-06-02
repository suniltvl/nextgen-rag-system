"""End-to-end pipeline orchestrator.

Loads a config, builds every component, walks examples one at a time, and
writes per-example results to JSONL plus a summary JSON.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from tqdm import tqdm

from rag_kag.chunkers import build_chunker
from rag_kag.config import ExperimentCfg, Secrets
from rag_kag.data_loaders import RAGBenchLoader
from rag_kag.embedders import build_embedder
from rag_kag.evaluators import TraceEvaluator, retrieval_diagnostics
from rag_kag.evaluators.trace import TraceInputs
from rag_kag.generators import build_generator
from rag_kag.retrievers import build_retriever
from rag_kag.types import Example, ExampleResult, TraceMetrics
from rag_kag.vectorstores import build_vectorstore


class Pipeline:
    def __init__(
        self,
        cfg: ExperimentCfg,
        secrets: Secrets | None = None,
        verbose: bool = False,
    ):
        self.cfg = cfg
        self.secrets = secrets or Secrets()
        self.verbose = verbose
        self._console = Console() if verbose else None

        self.chunker = build_chunker(cfg.chunker)
        self.embedder = build_embedder(cfg.embedder)
        self.vectorstore = build_vectorstore(cfg.vectorstore, self.secrets)
        self.retriever = build_retriever(
            cfg.retriever, embedder=self.embedder, vectorstore=self.vectorstore
        )
        self.generator = build_generator(cfg.generator)
        self.trace = TraceEvaluator()

        if self.verbose:
            self._log_components()

    # --- public API ------------------------------------------------------

    def run(self) -> Path:
        """Run the experiment, write JSONL + summary, return the run dir."""
        run_dir = self._make_run_dir()
        loader = RAGBenchLoader(
            subset=self.cfg.data.subset,
            split=self.cfg.data.split,
            cache_dir=self.cfg.data.cache_dir,
        )
        examples = loader.iter_examples(limit=self.cfg.data.limit)

        results: list[ExampleResult] = []
        per_example_path = run_dir / "examples.jsonl"
        # In verbose mode, suppress tqdm so per-example logs aren't interleaved
        # with the progress bar.
        iterator = examples if self.verbose else tqdm(examples, desc=self.cfg.name)
        with per_example_path.open("w") as f:
            for idx, ex in enumerate(iterator):
                if self.verbose:
                    self._log_example_start(idx, ex)
                result = self._run_one(ex)
                f.write(json.dumps(_to_jsonable(asdict(result))) + "\n")
                results.append(result)
                if self.verbose:
                    self._log_example_end(result)

        summary = self._summarize(results)
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        # Save the resolved config alongside results for reproducibility.
        (run_dir / "config.json").write_text(self.cfg.model_dump_json(indent=2))
        return run_dir

    def run_one_example(self, example: Example) -> ExampleResult:
        """Public hook used by the UI / tests to run a single example."""
        return self._run_one(example)

    # --- internals -------------------------------------------------------

    def _run_one(self, example: Example) -> ExampleResult:
        start = time.perf_counter()

        t = time.perf_counter()
        chunks = self.chunker.chunk(example)
        if self.verbose:
            self._log_step(
                "chunk",
                f"{len(chunks)} chunks from {len(example.documents)} docs",
                time.perf_counter() - t,
            )

        t = time.perf_counter()
        namespace = example.id
        self.retriever.index(namespace, chunks)
        if self.verbose:
            self._log_step(
                "index",
                f"namespace={namespace}  vectors={len(chunks)}  dim={self.embedder.dim}",
                time.perf_counter() - t,
            )

        top_k = getattr(self.cfg.retriever, "top_k", 5)
        t = time.perf_counter()
        retrieved = self.retriever.retrieve(namespace, example.question, top_k)
        if self.verbose:
            self._log_retrieved(retrieved, time.perf_counter() - t)

        t = time.perf_counter()
        gen = self.generator.generate(example.question, retrieved)
        if self.verbose:
            self._log_step(
                "generate",
                f"model={gen.model}  answer={_truncate(gen.answer, 140)}",
                time.perf_counter() - t,
            )

        metrics = self.trace.score(
            TraceInputs(example=example, retrieved=retrieved, answer=gen.answer)
        )
        diagnostics = retrieval_diagnostics(example, retrieved, k=top_k)
        latency = time.perf_counter() - start

        return ExampleResult(
            example_id=example.id,
            domain=example.domain,
            subset=example.subset,
            question=example.question,
            answer=gen.answer,
            retrieved_chunk_ids=[r.chunk.chunk_id for r in retrieved],
            metrics=metrics,
            diagnostics=diagnostics,
            reference_scores=example.reference_scores,
            latency_s=latency,
            model=gen.model,
        )

    def _make_run_dir(self) -> Path:
        # Avoid Date.now() — use a monotonically-increasing suffix instead.
        # (We can't access wall-clock here in a pure way, but pathlib
        # collisions on subsecond reruns are fine; we append a counter.)
        base = self.cfg.output_dir / f"{self.cfg.name}__{self.cfg.data.subset}"
        candidate = base
        counter = 1
        while candidate.exists():
            candidate = Path(f"{base}__{counter}")
            counter += 1
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate

    def _summarize(self, results: list[ExampleResult]) -> dict[str, Any]:
        if not results:
            return {"n": 0}
        avg_metrics = _mean_metrics(r.metrics for r in results)
        # Compare against the dataset's reference scores when present so
        # we can sanity-check our evaluator (proposal §5.1 step 5).
        ref_keys = ("relevance_score", "utilization_score", "completeness_score", "adherence_score")
        reference_avg: dict[str, float] = {}
        for k in ref_keys:
            vals = [r.reference_scores[k] for r in results if k in r.reference_scores]
            if vals:
                reference_avg[k] = sum(vals) / len(vals)

        return {
            "name": self.cfg.name,
            "subset": self.cfg.data.subset,
            "domain": results[0].domain,
            "n": len(results),
            "model": results[0].model,
            "avg_trace": {
                "context_relevance": avg_metrics.context_relevance,
                "context_utilization": avg_metrics.context_utilization,
                "completeness": avg_metrics.completeness,
                "adherence": avg_metrics.adherence,
            },
            "avg_reference": reference_avg,
            "avg_latency_s": sum(r.latency_s for r in results) / len(results),
        }


    # --- verbose logging -------------------------------------------------

    def _log_components(self) -> None:
        assert self._console is not None
        c = self._console
        c.rule(f"[bold]{self.cfg.name}[/]  ·  subset={self.cfg.data.subset}  ·  limit={self.cfg.data.limit}")
        c.print(
            f"[dim]chunker[/]    {type(self.chunker).__name__}\n"
            f"[dim]embedder[/]   {self.embedder.name}  (dim={self.embedder.dim})\n"
            f"[dim]vectorstore[/] {type(self.vectorstore).__name__}\n"
            f"[dim]retriever[/]  {type(self.retriever).__name__}  (top_k={getattr(self.cfg.retriever, 'top_k', 5)})\n"
            f"[dim]generator[/]  {self.cfg.generator.model}"
        )

    def _log_example_start(self, idx: int, example: Example) -> None:
        assert self._console is not None
        n_rel = len(example.all_relevant_sentence_keys)
        n_util = len(example.all_utilized_sentence_keys)
        self._console.rule(
            f"[bold cyan]example {idx + 1}[/]  id={example.id}  "
            f"({n_rel} relevant / {n_util} utilized sentence keys)",
            align="left",
        )
        self._console.print(f"[bold]Q:[/] {example.question}")

    def _log_step(self, label: str, detail: str, dt: float) -> None:
        assert self._console is not None
        self._console.print(f"  [yellow]{label:>8}[/]  [dim]{dt:5.2f}s[/]  {detail}")

    def _log_retrieved(self, retrieved: list, dt: float) -> None:
        assert self._console is not None
        self._console.print(f"  [yellow]{'retrieve':>8}[/]  [dim]{dt:5.2f}s[/]  top_k={len(retrieved)}")
        for r in retrieved:
            keys = ",".join(r.chunk.sentence_keys) or "-"
            preview = _truncate(r.chunk.text.replace("\n", " "), 100)
            self._console.print(
                f"           [dim]#{r.rank} score={r.score:.3f} keys=[{keys}][/]  {preview}"
            )

    def _log_example_end(self, result: ExampleResult) -> None:
        assert self._console is not None
        ref = result.reference_scores
        m = result.metrics
        table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
        table.add_column("metric")
        table.add_column("ours", justify="right")
        table.add_column("reference", justify="right")
        for ours_val, ref_key, label in (
            (m.context_relevance, "relevance_score", "context_relevance"),
            (m.context_utilization, "utilization_score", "context_utilization"),
            (m.completeness, "completeness_score", "completeness"),
            (m.adherence, "adherence_score", "adherence"),
        ):
            ref_val = ref.get(ref_key)
            table.add_row(
                label,
                f"{ours_val:.3f}",
                f"{ref_val:.3f}" if ref_val is not None else "-",
            )
        self._console.print(
            Panel(table, title=f"latency={result.latency_s:.2f}s", title_align="left")
        )


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1] + "…"


def _mean_metrics(metrics: Iterable[TraceMetrics]) -> TraceMetrics:
    cr = cu = co = ad = 0.0
    n = 0
    for m in metrics:
        cr += m.context_relevance
        cu += m.context_utilization
        co += m.completeness
        ad += m.adherence
        n += 1
    if n == 0:
        return TraceMetrics(0.0, 0.0, 0.0, 0.0)
    return TraceMetrics(cr / n, cu / n, co / n, ad / n)


def _to_jsonable(obj: Any) -> Any:
    """Recursively coerce dataclass dicts into JSON-friendly values."""
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    return obj
