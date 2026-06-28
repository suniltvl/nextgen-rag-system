"""RAGBench exploratory analysis — doc-length distributions and subset stats.

Used in capstone §1.1 (EDA) and to pick per-domain ``chunk_size`` for V2/V3.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from rag_kag.data_loaders.ragbench import SUBSET_TO_DOMAIN, RAGBenchLoader
from rag_kag.types import Example

# One or two subsets per RAGBench domain (capstone phase 1.1).
PHASE1_REPRESENTATIVE_SUBSETS: tuple[str, ...] = (
    "covidqa",
    "pubmedqa",
    "hotpotqa",
    "msmarco",
    "cuad",
    "emanual",
    "techqa",
    "finqa",
    "tatqa",
)

REFERENCE_CHUNK_SIZE = 256


@dataclass(slots=True)
class SubsetEdaStats:
    subset: str
    domain: str
    split: str
    n_examples: int
    n_documents: int
    docs_per_example_mean: float
    doc_words_min: int
    doc_words_max: int
    doc_words_mean: float
    doc_words_median: float
    doc_words_p90: float
    doc_words_p99: float
    pct_docs_gt_64: float
    pct_docs_gt_128: float
    pct_docs_gt_256: float
    pct_docs_gt_512: float
    median_sentence_words: float
    recommended_chunk_size: int
    recommended_overlap: int
    question_words_mean: float
    n_relevant_keys_mean: float
    n_utilized_keys_mean: float
    reference_relevance_mean: float | None
    reference_utilization_mean: float | None
    reference_completeness_mean: float | None
    reference_adherence_mean: float | None
    local_parquet: bool
    error: str | None = None


def iter_doc_word_counts(examples: Iterator[Example]) -> tuple[list[int], list[int], list[int]]:
    """Return (doc_word_counts, sentence_word_counts, question_word_counts)."""
    doc_counts: list[int] = []
    sent_counts: list[int] = []
    question_counts: list[int] = []
    for ex in examples:
        question_counts.append(_word_count(ex.question))
        for doc in ex.documents:
            doc_counts.append(_word_count(doc))
        for doc_sents in ex.documents_sentences:
            for sent in doc_sents:
                sent_counts.append(_word_count(sent.text))
    return doc_counts, sent_counts, question_counts


def recommended_chunk_params(
    doc_word_counts: Sequence[int],
    *,
    sentence_word_counts: Sequence[int] | None = None,
) -> tuple[int, int]:
    """Pick chunk_size from doc-length p30; overlap ≈ 25% rounded to 8."""
    if not doc_word_counts:
        return 256, 32
    arr = np.array(doc_word_counts, dtype=np.float64)
    chunk_size = max(32, int(np.percentile(arr, 30)))
    if sentence_word_counts:
        median_sent = float(np.median(np.array(sentence_word_counts, dtype=np.float64)))
        if median_sent > 0:
            k = max(1, round(chunk_size / median_sent))
            chunk_size = max(32, int(k * median_sent))
    overlap = max(8, min(chunk_size // 4, 64))
    return chunk_size, overlap


def analyze_subset(
    subset: str,
    *,
    split: str = "train",
    limit: int | None = None,
    local_root: Path | None = None,
) -> SubsetEdaStats:
    """Compute EDA stats for one RAGBench subset."""
    domain = SUBSET_TO_DOMAIN.get(subset, "unknown")
    loader = RAGBenchLoader(subset=subset, split=split, local_root=local_root)
    local_parquet = bool(loader._find_local_parquets())

    try:
        examples = list(loader.iter_examples(limit=limit))
    except Exception as exc:
        return SubsetEdaStats(
            subset=subset,
            domain=domain,
            split=split,
            n_examples=0,
            n_documents=0,
            docs_per_example_mean=0.0,
            doc_words_min=0,
            doc_words_max=0,
            doc_words_mean=0.0,
            doc_words_median=0.0,
            doc_words_p90=0.0,
            doc_words_p99=0.0,
            pct_docs_gt_64=0.0,
            pct_docs_gt_128=0.0,
            pct_docs_gt_256=0.0,
            pct_docs_gt_512=0.0,
            median_sentence_words=0.0,
            recommended_chunk_size=256,
            recommended_overlap=32,
            question_words_mean=0.0,
            n_relevant_keys_mean=0.0,
            n_utilized_keys_mean=0.0,
            reference_relevance_mean=None,
            reference_utilization_mean=None,
            reference_completeness_mean=None,
            reference_adherence_mean=None,
            local_parquet=local_parquet,
            error=str(exc),
        )

    doc_counts, sent_counts, question_counts = iter_doc_word_counts(iter(examples))
    n_examples = len(examples)
    n_documents = len(doc_counts)

    if doc_counts:
        arr = np.array(doc_counts, dtype=np.float64)
        doc_min = int(arr.min())
        doc_max = int(arr.max())
        doc_mean = float(arr.mean())
        doc_median = float(np.median(arr))
        doc_p90 = float(np.percentile(arr, 90))
        doc_p99 = float(np.percentile(arr, 99))
        pct_gt = {threshold: float(np.mean(arr > threshold) * 100) for threshold in (64, 128, 256, 512)}
    else:
        doc_min = doc_max = 0
        doc_mean = doc_median = doc_p90 = doc_p99 = 0.0
        pct_gt = {64: 0.0, 128: 0.0, 256: 0.0, 512: 0.0}

    median_sent = float(np.median(np.array(sent_counts, dtype=np.float64))) if sent_counts else 0.0
    chunk_size, overlap = recommended_chunk_params(doc_counts, sentence_word_counts=sent_counts)

    docs_per_ex = n_documents / n_examples if n_examples else 0.0
    q_mean = float(np.mean(question_counts)) if question_counts else 0.0
    rel_mean = float(np.mean([len(ex.all_relevant_sentence_keys) for ex in examples]))
    util_mean = float(np.mean([len(ex.all_utilized_sentence_keys) for ex in examples]))

    def _ref_mean(key: str) -> float | None:
        vals = [ex.reference_scores[key] for ex in examples if key in ex.reference_scores]
        return float(np.mean(vals)) if vals else None

    return SubsetEdaStats(
        subset=subset,
        domain=domain,
        split=split,
        n_examples=n_examples,
        n_documents=n_documents,
        docs_per_example_mean=docs_per_ex,
        doc_words_min=doc_min,
        doc_words_max=doc_max,
        doc_words_mean=doc_mean,
        doc_words_median=doc_median,
        doc_words_p90=doc_p90,
        doc_words_p99=doc_p99,
        pct_docs_gt_64=pct_gt[64],
        pct_docs_gt_128=pct_gt[128],
        pct_docs_gt_256=pct_gt[256],
        pct_docs_gt_512=pct_gt[512],
        median_sentence_words=median_sent,
        recommended_chunk_size=chunk_size,
        recommended_overlap=overlap,
        question_words_mean=q_mean,
        n_relevant_keys_mean=rel_mean,
        n_utilized_keys_mean=util_mean,
        reference_relevance_mean=_ref_mean("relevance_score"),
        reference_utilization_mean=_ref_mean("utilization_score"),
        reference_completeness_mean=_ref_mean("completeness_score"),
        reference_adherence_mean=_ref_mean("adherence_score"),
        local_parquet=local_parquet,
    )


def analyze_subsets(
    subsets: Sequence[str],
    *,
    split: str = "train",
    limit: int | None = None,
    local_root: Path | None = None,
) -> list[SubsetEdaStats]:
    return [
        analyze_subset(s, split=split, limit=limit, local_root=local_root) for s in subsets
    ]


def stats_to_markdown(
    stats: Sequence[SubsetEdaStats],
    *,
    split: str,
    limit: int | None,
) -> str:
    """Render a mentor-ready EDA report."""
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# RAGBench EDA — Phase 1.1",
        "",
        f"Generated: {now}",
        f"Split: `{split}`",
        f"Limit: {limit if limit is not None else 'full split'}",
        "",
        "## Subset coverage",
        "",
        "Representative subsets (one or two per RAGBench domain):",
        "",
    ]
    for s in PHASE1_REPRESENTATIVE_SUBSETS:
        domain = SUBSET_TO_DOMAIN.get(s, "?")
        lines.append(f"- `{s}` → {domain}")

    lines.extend(
        [
            "",
            "## Document length distribution",
            "",
            "Word counts use whitespace tokenization (same as the sliding-window chunker).",
            f"`pct_gt_{REFERENCE_CHUNK_SIZE}` = % of documents longer than the V1 baseline "
            f"chunk_size ({REFERENCE_CHUNK_SIZE} words). When this is 0%, V1 retrieval is "
            "degenerate (every doc fits in one chunk).",
            "",
            "| subset | domain | n | docs | doc median | doc max | >64% | >128% | >256% | >512% | "
            "rec chunk | overlap | local |",
            "|--------|--------|---|------|------------|---------|------|-------|-------|-------|"
            "-----------|---------|-------|",
        ]
    )

    for st in stats:
        if st.error:
            lines.append(
                f"| {st.subset} | {st.domain} | — | — | — | — | — | — | — | — | — | — | "
                f"error: {st.error[:40]}… |"
            )
            continue
        lines.append(
            f"| {st.subset} | {st.domain} | {st.n_examples} | {st.n_documents} | "
            f"{st.doc_words_median:.0f} | {st.doc_words_max} | "
            f"{st.pct_docs_gt_64:.1f} | {st.pct_docs_gt_128:.1f} | "
            f"{st.pct_docs_gt_256:.1f} | {st.pct_docs_gt_512:.1f} | "
            f"{st.recommended_chunk_size} | {st.recommended_overlap} | "
            f"{'yes' if st.local_parquet else 'hf'} |"
        )

    lines.extend(
        [
            "",
            "## Per-subset detail",
            "",
        ]
    )
    for st in stats:
        lines.append(f"### {st.subset} ({st.domain})")
        if st.error:
            lines.append(f"- **Error:** {st.error}")
            lines.append("")
            continue
        lines.extend(
            [
                f"- Examples: {st.n_examples:,}; documents: {st.n_documents:,} "
                f"(mean {st.docs_per_example_mean:.1f} docs/example)",
                f"- Doc words: min={st.doc_words_min}, mean={st.doc_words_mean:.1f}, "
                f"median={st.doc_words_median:.1f}, p90={st.doc_words_p90:.0f}, "
                f"p99={st.doc_words_p99:.0f}, max={st.doc_words_max}",
                f"- Question words (mean): {st.question_words_mean:.1f}",
                f"- Sentence keys per example: relevant={st.n_relevant_keys_mean:.1f}, "
                f"utilized={st.n_utilized_keys_mean:.1f}",
                f"- Median sentence length: {st.median_sentence_words:.1f} words",
                f"- Recommended `chunk_size` / `overlap`: {st.recommended_chunk_size} / "
                f"{st.recommended_overlap} (p30 doc length + sentence alignment)",
            ]
        )
        ref_parts: list[str] = []
        if st.reference_relevance_mean is not None:
            ref_parts.append(f"relevance={st.reference_relevance_mean:.3f}")
        if st.reference_utilization_mean is not None:
            ref_parts.append(f"utilization={st.reference_utilization_mean:.3f}")
        if st.reference_completeness_mean is not None:
            ref_parts.append(f"completeness={st.reference_completeness_mean:.3f}")
        if st.reference_adherence_mean is not None:
            ref_parts.append(f"adherence={st.reference_adherence_mean:.3f}")
        if ref_parts:
            lines.append(f"- Dataset reference TRACe (mean): {', '.join(ref_parts)}")
        lines.append("")

    lines.extend(
        [
            "## Schema notes",
            "",
            "Key RAGBench fields used by the pipeline:",
            "",
            "- `question` — user query",
            "- `documents` / `documents_sentences` — retrieved corpus with sentence keys (`0a`, `1b`, …)",
            "- `all_relevant_sentence_keys` / `all_utilized_sentence_keys` — TRACe ground truth",
            "- `relevance_score`, `utilization_score`, `completeness_score`, `adherence_score` — "
            "reference scores for evaluator validation",
            "",
            "## Chunk size methodology",
            "",
            "1. Target ≥50% of documents producing >1 chunk at chosen `chunk_size` (use p30 of doc lengths).",
            "2. Align to sentence boundaries via median sentence length.",
            "3. Keep under embedder context (~512 tokens ≈ ~350 words for bge-small).",
            "",
        ]
    )
    return "\n".join(lines)


def write_eda_report(
    output_path: Path,
    *,
    subsets: Sequence[str] = PHASE1_REPRESENTATIVE_SUBSETS,
    split: str = "train",
    limit: int | None = None,
    local_root: Path | None = None,
) -> list[SubsetEdaStats]:
    stats = analyze_subsets(subsets, split=split, limit=limit, local_root=local_root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        stats_to_markdown(stats, split=split, limit=limit), encoding="utf-8"
    )
    return stats


def subset_stats_json(stats: Sequence[SubsetEdaStats]) -> list[dict[str, object]]:
    return [asdict(s) for s in stats]


def _word_count(text: str) -> int:
    return len(text.split())
