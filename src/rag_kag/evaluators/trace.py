"""TRACe evaluator — sentence-key metrics aligned with RAGBench (arXiv:2407.11005).

Formulas (sentence / substring level, aggregated as key counts):

  context_relevance   = |R ∩ retrieved| / |retrieved|
  context_utilization = |U ∩ retrieved| / |retrieved|
  completeness        = |R ∩ U| / |R|
  adherence           = 1 if every response sentence is grounded in context, else 0

Where R = relevant sentence keys, U = utilized sentence keys (in the answer),
and ``retrieved`` = keys present in retrieved chunks.

``utilized_keys`` mode:
  * ``infer`` (default) — infer U from answer overlap with retrieved chunks
    (pipeline runs with model-generated answers).
  * ``dataset`` — use RAGBench ``all_utilized_sentence_keys`` (reference validation
    with the dataset's annotated response).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from rag_kag.types import Example, RetrievedChunk, TraceMetrics

UtilizedKeysMode = Literal["infer", "dataset"]

# Minimum token overlap to treat a chunk sentence as "used" in the answer.
_TOKEN_OVERLAP_FRAC = 0.25
# Per response sentence, minimum overlap with retrieved text to count as grounded.
_ADHERENCE_SENTENCE_FRAC = 0.3


def _safe_div(num: float, denom: float) -> float:
    return num / denom if denom > 0 else 0.0


@dataclass(slots=True)
class TraceInputs:
    """Bundle of what TRACe needs for a single example."""

    example: Example
    retrieved: list[RetrievedChunk]
    answer: str


class TraceEvaluator:
    """Sentence-key TRACe metrics."""

    def __init__(
        self,
        *,
        utilized_keys: UtilizedKeysMode = "infer",
        ignore_missing_keys: bool = True,
    ):
        self.utilized_keys = utilized_keys
        self.ignore_missing_keys = ignore_missing_keys

    def score(self, inputs: TraceInputs) -> TraceMetrics:
        ex = inputs.example
        relevant = set(ex.all_relevant_sentence_keys)
        utilized = self._utilized_keys(inputs)

        retrieved_keys: set[str] = set()
        for r in inputs.retrieved:
            retrieved_keys.update(r.chunk.sentence_keys)

        relevant_in_retrieved = relevant & retrieved_keys
        utilized_in_retrieved = utilized & retrieved_keys
        relevant_in_utilized = relevant & utilized

        ctx_relevance = _safe_div(len(relevant_in_retrieved), len(retrieved_keys))
        ctx_utilization = _safe_div(len(utilized_in_retrieved), len(retrieved_keys))
        # RAGBench sets completeness=1.0 when there are no relevant keys (vacuous).
        completeness = (
            1.0
            if not relevant
            else _safe_div(len(relevant_in_utilized), len(relevant))
        )
        adherence = self._adherence(inputs)

        return TraceMetrics(
            context_relevance=ctx_relevance,
            context_utilization=ctx_utilization,
            completeness=completeness,
            adherence=adherence,
        )

    def _utilized_keys(self, inputs: TraceInputs) -> set[str]:
        if self.utilized_keys == "dataset":
            return set(inputs.example.all_utilized_sentence_keys)
        return _infer_utilized_keys(inputs.answer, inputs.retrieved)

    def _adherence(self, inputs: TraceInputs) -> float:
        from rag_kag.generators.prompts import RGB_REJECTION_PHRASE

        answer = inputs.answer.strip()
        if not answer:
            return 0.0
        if RGB_REJECTION_PHRASE.lower() in answer.lower():
            return 1.0

        # Reference validation: RAGBench adherence == no unsupported response sentences.
        if self.utilized_keys == "dataset":
            return 1.0 if not inputs.example.unsupported_response_sentence_keys else 0.0

        retrieved_text = " ".join(r.chunk.text for r in inputs.retrieved)
        retrieved_tokens = _tokens(retrieved_text)
        if not retrieved_tokens:
            return 0.0

        sentences = _split_sentences(answer)
        if not sentences:
            return 0.0

        for sentence in sentences:
            sent_tokens = _tokens(sentence)
            if not sent_tokens:
                continue
            overlap = len(sent_tokens & retrieved_tokens) / len(sent_tokens)
            if overlap < _ADHERENCE_SENTENCE_FRAC:
                return 0.0
        return 1.0


def _infer_utilized_keys(answer: str, retrieved: list[RetrievedChunk]) -> set[str]:
    """Sentence keys from retrieved chunks whose text overlaps the answer."""
    ans_tokens = _tokens(answer)
    if not ans_tokens:
        return set()
    utilized: set[str] = set()
    for r in retrieved:
        chunk_tokens = _tokens(r.chunk.text)
        if not chunk_tokens:
            continue
        ans_cov = len(ans_tokens & chunk_tokens) / len(ans_tokens)
        chunk_cov = len(ans_tokens & chunk_tokens) / len(chunk_tokens)
        if ans_cov >= _TOKEN_OVERLAP_FRAC or chunk_cov >= _TOKEN_OVERLAP_FRAC:
            utilized.update(r.chunk.sentence_keys)
    return utilized


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _tokens(text: str) -> set[str]:
    """Lowercased alphanum tokens, length >= 3 (drops articles/punct)."""
    out: set[str] = set()
    word: list[str] = []
    for ch in text.lower():
        if ch.isalnum():
            word.append(ch)
        else:
            if len(word) >= 3:
                out.add("".join(word))
            word.clear()
    if len(word) >= 3:
        out.add("".join(word))
    return out
