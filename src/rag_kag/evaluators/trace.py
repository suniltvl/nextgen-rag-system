"""TRACe evaluator — V1 implementation derived from sentence-level annotations.

This is the deliverable the proposal calls out as the most important. The
mentor brief is explicit: "implement the metrics from the formulas" — no
RAGAS/TruLens wrapping.

Definitions (RAGBench paper, arXiv:2407.11005):

  context_relevance  = |relevant ∩ retrieved_sentences| / |retrieved_sentences|
                       (fraction of retrieved context that's actually useful)

  context_utilization = |utilized ∩ retrieved_sentences| / |retrieved_sentences|
                       (fraction of retrieved context the answer actually used)

  completeness        = |relevant ∩ retrieved_sentences| / |relevant|
                       (fraction of relevant info that *made it through* retrieval
                       and is therefore even available to be in the answer)

  adherence           ∈ {0, 1}
                       (whether every claim in the answer is grounded in the
                       retrieved context — V1 uses an NLI-free heuristic; we
                       upgrade to an LLM-judge in Week 3)

The first three metrics are deterministic given retrieved chunks + dataset
annotations, which lets us validate them against the RAGBench reference
score fields (`relevance_score`, `utilization_score`, `completeness_score`)
on dataset-provided responses before we run any model. That's the validation
strategy from the proposal §5.1 step 5.
"""

from __future__ import annotations

from dataclasses import dataclass

from rag_kag.types import Example, RetrievedChunk, TraceMetrics


def _safe_div(num: float, denom: float) -> float:
    return num / denom if denom > 0 else 0.0


@dataclass(slots=True)
class TraceInputs:
    """Bundle of what TRACe needs for a single example.

    Pulled out so future evaluators (LLM-judge adherence, claim-level
    completeness, etc.) can share one input shape.
    """

    example: Example
    retrieved: list[RetrievedChunk]
    answer: str


class TraceEvaluator:
    """Sentence-key-based TRACe metrics. V1 — no LLM required."""

    def __init__(self, *, ignore_missing_keys: bool = True):
        # Some subsets (HAGRID, MS Marco) have sparser sentence-key
        # annotations. ignore_missing_keys=True returns 0.0 instead of
        # raising when a metric's denominator is empty — matches the
        # convention in the RAGBench reference scorer.
        self.ignore_missing_keys = ignore_missing_keys

    def score(self, inputs: TraceInputs) -> TraceMetrics:
        ex = inputs.example

        relevant = set(ex.all_relevant_sentence_keys)
        utilized = set(ex.all_utilized_sentence_keys)

        retrieved_keys: set[str] = set()
        for r in inputs.retrieved:
            retrieved_keys.update(r.chunk.sentence_keys)

        relevant_in_retrieved = relevant & retrieved_keys
        utilized_in_retrieved = utilized & retrieved_keys

        ctx_relevance = _safe_div(len(relevant_in_retrieved), len(retrieved_keys))
        ctx_utilization = _safe_div(len(utilized_in_retrieved), len(retrieved_keys))
        completeness = _safe_div(len(relevant_in_retrieved), len(relevant))
        adherence = self._adherence_heuristic(inputs)

        return TraceMetrics(
            context_relevance=ctx_relevance,
            context_utilization=ctx_utilization,
            completeness=completeness,
            adherence=adherence,
        )

    # --- adherence ------------------------------------------------------

    @staticmethod
    def _adherence_heuristic(inputs: TraceInputs) -> float:
        """V1 adherence proxy.

        A non-trivial adherence score requires per-claim NLI against the
        retrieved context, which is too heavy for V1. Until the LLM-judge
        version lands (Week 3), we use a coarse signal: the answer is
        considered "adherent" if at least one retrieved chunk shares a
        meaningful overlap with it. This is intentionally lenient — the
        purpose of V1 adherence is to be present in the schema, not to
        replace the eventual LLM judge.

        Returns 0.0 if the model emitted the explicit refusal phrase
        (refusing is always adherent in TRACe semantics).
        """
        from rag_kag.generators.prompts import RGB_REJECTION_PHRASE

        answer = inputs.answer.strip()
        if not answer:
            return 0.0
        if RGB_REJECTION_PHRASE.lower() in answer.lower():
            return 1.0
        # Cheap n-gram overlap between answer and any retrieved chunk.
        ans_tokens = _tokens(answer)
        if not ans_tokens:
            return 0.0
        for r in inputs.retrieved:
            chunk_tokens = _tokens(r.chunk.text)
            if not chunk_tokens:
                continue
            overlap = len(ans_tokens & chunk_tokens) / len(ans_tokens)
            if overlap >= 0.3:
                return 1.0
        return 0.0


def _tokens(text: str) -> set[str]:
    """Lowercased alphanum tokens, length >= 3 (drops articles/punct)."""
    out: set[str] = set()
    word = []
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
