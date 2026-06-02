"""Retrieval-side diagnostics: Recall@k / Precision@k / MRR@k.

These are NOT TRACe metrics — they're explanatory signals that help
interpret why a TRACe metric moved when a component changes. They use the
same gold sentence keys (`all_relevant_sentence_keys`) as TRACe.
"""

from __future__ import annotations

from rag_kag.types import Example, RetrievalDiagnostics, RetrievedChunk


def retrieval_diagnostics(
    example: Example,
    retrieved: list[RetrievedChunk],
    *,
    k: int | None = None,
) -> RetrievalDiagnostics:
    """A retrieved chunk is "relevant" if it contains ANY gold sentence key."""
    if k is None:
        k = len(retrieved)
    top = retrieved[:k]
    relevant_keys = set(example.all_relevant_sentence_keys)

    if not relevant_keys or not top:
        return RetrievalDiagnostics(recall_at_k=0.0, precision_at_k=0.0, mrr_at_k=0.0, k=k)

    relevant_chunks = [
        r for r in top if any(sk in relevant_keys for sk in r.chunk.sentence_keys)
    ]

    # Recall: did we retrieve at least one relevant key per gold sentence?
    retrieved_keys = {sk for r in top for sk in r.chunk.sentence_keys}
    covered = relevant_keys & retrieved_keys
    recall = len(covered) / len(relevant_keys)

    precision = len(relevant_chunks) / len(top)

    mrr = 0.0
    for r in top:
        if any(sk in relevant_keys for sk in r.chunk.sentence_keys):
            mrr = 1.0 / (r.rank + 1)
            break

    return RetrievalDiagnostics(recall_at_k=recall, precision_at_k=precision, mrr_at_k=mrr, k=k)
