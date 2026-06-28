"""Validate TRACe implementation against RAGBench reference scores."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rag_kag.data_loaders import RAGBenchLoader
from rag_kag.evaluators.trace import TraceEvaluator, TraceInputs
from rag_kag.types import Chunk, Example, RetrievedChunk


@dataclass(slots=True)
class TraceValidationRow:
    example_id: str
    metric: str
    ours: float
    reference: float
    delta: float
    match: bool


@dataclass(slots=True)
class TraceValidationReport:
    subset: str
    split: str
    n: int
    matches: dict[str, int]
    mismatches: list[TraceValidationRow]
    avg_ours: dict[str, float]
    avg_reference: dict[str, float]
    avg_delta: dict[str, float]


def _full_retrieval(example: Example) -> list[RetrievedChunk]:
    """Treat every document sentence as retrieved (validation harness)."""
    retrieved: list[RetrievedChunk] = []
    for doc_idx, doc_sents in enumerate(example.documents_sentences):
        text = " ".join(s.text for s in doc_sents)
        keys = [s.key for s in doc_sents]
        retrieved.append(
            RetrievedChunk(
                chunk=Chunk(
                    chunk_id=f"val::d{doc_idx}",
                    text=text,
                    doc_index=doc_idx,
                    sentence_keys=keys,
                ),
                score=1.0,
                rank=doc_idx,
            )
        )
    return retrieved


def validate_subset(
    subset: str,
    *,
    split: str = "train",
    limit: int | None = 50,
    tolerance: float = 1e-3,
) -> TraceValidationReport:
    """Score dataset responses with gold utilization labels; compare to reference."""
    loader = RAGBenchLoader(subset=subset, split=split)
    evaluator = TraceEvaluator(utilized_keys="dataset")

    pairs = [
        ("context_relevance", "relevance_score"),
        ("context_utilization", "utilization_score"),
        ("completeness", "completeness_score"),
        ("adherence", "adherence_score"),
    ]

    matches = {ours_key: 0 for ours_key, _ in pairs}
    mismatches: list[TraceValidationRow] = []
    sums_ours: dict[str, float] = {k: 0.0 for k, _ in pairs}
    sums_ref: dict[str, float] = {k: 0.0 for k, _ in pairs}
    n = 0

    for ex in loader.iter_examples(limit=limit):
        answer = ex.response or ""
        retrieved = _full_retrieval(ex)
        metrics = evaluator.score(
            TraceInputs(example=ex, retrieved=retrieved, answer=answer)
        )
        n += 1
        metric_values = {
            "context_relevance": metrics.context_relevance,
            "context_utilization": metrics.context_utilization,
            "completeness": metrics.completeness,
            "adherence": metrics.adherence,
        }
        for ours_key, ref_key in pairs:
            ours_val = metric_values[ours_key]
            ref_val = ex.reference_scores.get(ref_key)
            if ref_val is None:
                continue
            sums_ours[ours_key] += ours_val
            sums_ref[ours_key] += ref_val
            delta = ours_val - ref_val
            ok = abs(delta) <= tolerance
            if ok:
                matches[ours_key] += 1
            else:
                mismatches.append(
                    TraceValidationRow(
                        example_id=ex.id,
                        metric=ours_key,
                        ours=ours_val,
                        reference=ref_val,
                        delta=delta,
                        match=False,
                    )
                )

    avg_ours = {k: sums_ours[k] / n if n else 0.0 for k in sums_ours}
    avg_ref = {k: sums_ref[k] / n if n else 0.0 for k in sums_ref}
    avg_delta = {k: avg_ours[k] - avg_ref[k] for k in avg_ours}

    return TraceValidationReport(
        subset=subset,
        split=split,
        n=n,
        matches=matches,
        mismatches=mismatches,
        avg_ours=avg_ours,
        avg_reference=avg_ref,
        avg_delta=avg_delta,
    )


def write_validation_markdown(report: TraceValidationReport, path: Path) -> None:
    lines = [
        f"# TRACe validator — {report.subset} ({report.split})",
        "",
        f"Examples: {report.n}",
        "",
        "| metric | matches | avg ours | avg reference | avg delta |",
        "|--------|---------|----------|---------------|-----------|",
    ]
    for metric in (
        "context_relevance",
        "context_utilization",
        "completeness",
        "adherence",
    ):
        lines.append(
            f"| {metric} | {report.matches.get(metric, 0)}/{report.n} | "
            f"{report.avg_ours.get(metric, 0):.4f} | "
            f"{report.avg_reference.get(metric, 0):.4f} | "
            f"{report.avg_delta.get(metric, 0):+.4f} |"
        )
    if report.mismatches:
        lines.extend(["", "## Sample mismatches (first 20)", ""])
        for row in report.mismatches[:20]:
            lines.append(
                f"- `{row.example_id}` {row.metric}: ours={row.ours:.4f} "
                f"ref={row.reference:.4f} (Δ={row.delta:+.4f})"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
