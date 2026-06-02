"""Core data types shared across pipeline stages.

These types are deliberately minimal and serializable so experiment runs
can be saved as JSONL without custom encoders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Sentence:
    """A single sentence within a source document, indexed by its RAGBench key.

    The `key` is the dotted index from RAGBench's ``documents_sentences`` field
    (e.g. ``"0a"``, ``"3b"``) so that sentence-level support annotations can be
    matched back to the dataset's reference scores.
    """

    key: str
    text: str
    doc_index: int


@dataclass(slots=True)
class Chunk:
    """A retrievable unit. Carries provenance back to the source sentences.

    ``sentence_keys`` enables TRACe metric computation: a retrieved chunk
    counts toward context-relevance only if it overlaps with the example's
    ``all_relevant_sentence_keys`` set.
    """

    chunk_id: str
    text: str
    doc_index: int
    sentence_keys: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Example:
    """A single RAGBench example normalized into our internal schema.

    Mirrors the fields documented in the proposal §4.2 Important RAGBench
    Fields table. We keep the dataset's reference scores so evaluators can
    validate their implementation against them.
    """

    id: str
    domain: str
    subset: str
    question: str
    documents: list[str]
    documents_sentences: list[list[Sentence]]
    response: str | None = None  # dataset-provided generated response
    all_relevant_sentence_keys: list[str] = field(default_factory=list)
    all_utilized_sentence_keys: list[str] = field(default_factory=list)
    # Reference scores from the dataset (used to validate our evaluator)
    reference_scores: dict[str, float] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass(slots=True)
class RetrievedChunk:
    """A chunk paired with the retriever's score and rank."""

    chunk: Chunk
    score: float
    rank: int


@dataclass(slots=True)
class GenerationResult:
    """LLM output plus the retrieved evidence used to produce it."""

    answer: str
    retrieved: list[RetrievedChunk]
    prompt: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)
    latency_s: float = 0.0


@dataclass(slots=True)
class TraceMetrics:
    """RAGBench TRACe metrics for a single example.

    Definitions follow the RAGBench paper (arXiv:2407.11005):
      * context_relevance — fraction of retrieved context that is relevant.
      * context_utilization — fraction of retrieved context used in the
        generated answer.
      * completeness — fraction of relevant info that appears in the answer.
      * adherence — whether every claim in the answer is grounded in context.
    """

    context_relevance: float
    context_utilization: float
    completeness: float
    adherence: float


@dataclass(slots=True)
class RetrievalDiagnostics:
    """Recall@k / Precision@k / MRR@k for a single example."""

    recall_at_k: float
    precision_at_k: float
    mrr_at_k: float
    k: int


@dataclass(slots=True)
class ExampleResult:
    """Per-example artifact written to JSONL after a run."""

    example_id: str
    domain: str
    subset: str
    question: str
    answer: str
    retrieved_chunk_ids: list[str]
    metrics: TraceMetrics
    diagnostics: RetrievalDiagnostics | None = None
    reference_scores: dict[str, float] = field(default_factory=dict)
    latency_s: float = 0.0
    model: str = ""
