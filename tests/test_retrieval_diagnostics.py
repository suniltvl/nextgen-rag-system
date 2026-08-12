"""Recall@k / Precision@k / MRR@k tests."""

from __future__ import annotations

from rag_kag.evaluators.retrieval import retrieval_diagnostics
from rag_kag.types import Chunk, Example, RetrievedChunk, Sentence


def _ex(relevant: list[str]) -> Example:
    return Example(
        id="ex",
        domain="test",
        subset="test",
        question="q?",
        documents=["doc"],
        documents_sentences=[[Sentence(key=k, text=k, doc_index=0) for k in relevant]],
        all_relevant_sentence_keys=relevant,
    )


def _r(idx: int, sks: list[str]) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(chunk_id=f"c{idx}", text=" ".join(sks), doc_index=0, sentence_keys=sks),
        score=1.0 - 0.1 * idx,
        rank=idx,
    )


def test_mrr_uses_first_relevant_rank() -> None:
    ex = _ex(["0a"])
    # Relevant chunk at rank 2 (third position) → MRR = 1/3.
    retrieved = [_r(0, ["0z"]), _r(1, ["0y"]), _r(2, ["0a"])]
    diag = retrieval_diagnostics(ex, retrieved, k=3)
    assert diag.recall_at_k == 1.0
    assert diag.precision_at_k == 1 / 3
    assert diag.mrr_at_k == 1 / 3


def test_no_relevant_returns_zeros() -> None:
    ex = _ex([])
    diag = retrieval_diagnostics(ex, [_r(0, ["0a"])], k=1)
    assert diag.recall_at_k == 0.0
    assert diag.precision_at_k == 0.0
    assert diag.mrr_at_k == 0.0
