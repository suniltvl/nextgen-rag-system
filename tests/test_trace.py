"""TRACe metric arithmetic — derived directly from the formula in trace.py."""

from __future__ import annotations

from rag_kag.evaluators.trace import TraceEvaluator, TraceInputs
from rag_kag.types import Chunk, Example, RetrievedChunk, Sentence


def _ex(relevant: list[str], utilized: list[str]) -> Example:
    texts = {k: f"sentence text for key {k} with enough words" for k in {*relevant, *utilized}}
    return Example(
        id="ex",
        domain="test",
        subset="test",
        question="q?",
        documents=["doc"],
        documents_sentences=[
            [Sentence(key=k, text=texts[k], doc_index=0) for k in sorted({*relevant, *utilized})]
        ],
        all_relevant_sentence_keys=relevant,
        all_utilized_sentence_keys=utilized,
    )


def _retrieved(
    sentence_keys_per_chunk: list[list[str]],
    texts: dict[str, str] | None = None,
) -> list[RetrievedChunk]:
    lookup = texts or {}
    return [
        RetrievedChunk(
            chunk=Chunk(
                chunk_id=f"c{i}",
                text=" ".join(lookup.get(k, f"sentence text for key {k} with enough words") for k in sks),
                doc_index=0,
                sentence_keys=sks,
            ),
            score=1.0 - 0.1 * i,
            rank=i,
        )
        for i, sks in enumerate(sentence_keys_per_chunk)
    ]


def test_perfect_retrieval_perfect_metrics() -> None:
    ex = _ex(relevant=["0a", "0b"], utilized=["0a", "0b"])
    retrieved = _retrieved([["0a"], ["0b"]])
    answer = "sentence text for key 0a with enough words sentence text for key 0b"
    metrics = TraceEvaluator().score(TraceInputs(example=ex, retrieved=retrieved, answer=answer))
    # 2 retrieved keys; 2/2 relevant; 2/2 utilized; recall 2/2.
    assert metrics.context_relevance == 1.0
    assert metrics.context_utilization == 1.0
    assert metrics.completeness == 1.0


def test_completeness_uses_utilized_not_retrieval() -> None:
    """Completeness = |R ∩ U| / |R|, not retrieval recall."""
    ex = _ex(relevant=["0a", "0b", "0c"], utilized=["0a", "0b"])
    # All relevant retrieved, but only 2/3 utilized in answer.
    retrieved = _retrieved([["0a", "0b", "0c"]])
    metrics = TraceEvaluator(utilized_keys="dataset").score(
        TraceInputs(example=ex, retrieved=retrieved, answer="ignored")
    )
    assert metrics.completeness == 2 / 3
    assert metrics.context_relevance == 1.0


def test_partial_retrieval() -> None:
    ex = _ex(relevant=["0a", "0b", "0c"], utilized=["0a"])
    retrieved = _retrieved(
        [["0a"], ["0z"]],
        texts={
            "0a": "sentence text for key 0a with enough words",
            "0z": "zebra kangaroo vocabulary completely unrelated",
        },
    )
    answer = "sentence text for key 0a with enough words"
    metrics = TraceEvaluator().score(TraceInputs(example=ex, retrieved=retrieved, answer=answer))
    assert metrics.context_relevance == 0.5  # |{0a}|/2
    assert metrics.context_utilization == 0.5  # |{0a}|/2
    assert metrics.completeness == 1 / 3  # |{0a}|/3 relevant utilized (infer)


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
