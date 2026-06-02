"""TRACe metric arithmetic — derived directly from the formula in trace.py."""

from __future__ import annotations

from rag_kag.evaluators.trace import TraceEvaluator, TraceInputs
from rag_kag.types import Chunk, Example, RetrievedChunk, Sentence


def _ex(relevant: list[str], utilized: list[str]) -> Example:
    return Example(
        id="ex",
        domain="test",
        subset="test",
        question="q?",
        documents=["doc"],
        documents_sentences=[[Sentence(key=k, text=k, doc_index=0) for k in {*relevant, *utilized}]],
        all_relevant_sentence_keys=relevant,
        all_utilized_sentence_keys=utilized,
    )


def _retrieved(sentence_keys_per_chunk: list[list[str]]) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk=Chunk(chunk_id=f"c{i}", text=" ".join(sks), doc_index=0, sentence_keys=sks),
            score=1.0 - 0.1 * i,
            rank=i,
        )
        for i, sks in enumerate(sentence_keys_per_chunk)
    ]


def test_perfect_retrieval_perfect_metrics() -> None:
    ex = _ex(relevant=["0a", "0b"], utilized=["0a", "0b"])
    retrieved = _retrieved([["0a"], ["0b"]])
    metrics = TraceEvaluator().score(TraceInputs(example=ex, retrieved=retrieved, answer="0a 0b"))
    # 2 retrieved keys; 2/2 relevant; 2/2 utilized; recall 2/2.
    assert metrics.context_relevance == 1.0
    assert metrics.context_utilization == 1.0
    assert metrics.completeness == 1.0


def test_partial_retrieval() -> None:
    ex = _ex(relevant=["0a", "0b", "0c"], utilized=["0a"])
    retrieved = _retrieved([["0a", "0z"]])  # 1 relevant key, 1 noise
    metrics = TraceEvaluator().score(TraceInputs(example=ex, retrieved=retrieved, answer="0a"))
    # retrieved keys = {0a, 0z}.
    assert metrics.context_relevance == 0.5  # |{0a}|/2
    assert metrics.context_utilization == 0.5  # |{0a}|/2
    assert metrics.completeness == 1 / 3  # |{0a}|/3


def test_empty_retrieval_returns_zeros() -> None:
    ex = _ex(relevant=["0a"], utilized=["0a"])
    metrics = TraceEvaluator().score(TraceInputs(example=ex, retrieved=[], answer=""))
    assert metrics.context_relevance == 0.0
    assert metrics.context_utilization == 0.0
    assert metrics.completeness == 0.0
    assert metrics.adherence == 0.0


def test_adherence_rewards_overlap_with_retrieved() -> None:
    ex = _ex(relevant=["0a"], utilized=["0a"])
    retrieved = _retrieved([["0a"]])
    retrieved[0].chunk.text = "the mitochondria is the powerhouse of the cell"
    answer = "mitochondria is the powerhouse"
    metrics = TraceEvaluator().score(TraceInputs(example=ex, retrieved=retrieved, answer=answer))
    assert metrics.adherence == 1.0


def test_adherence_credits_explicit_rejection() -> None:
    from rag_kag.generators.prompts import RGB_REJECTION_PHRASE

    ex = _ex(relevant=[], utilized=[])
    retrieved = _retrieved([["0z"]])
    metrics = TraceEvaluator().score(
        TraceInputs(example=ex, retrieved=retrieved, answer=RGB_REJECTION_PHRASE)
    )
    assert metrics.adherence == 1.0
