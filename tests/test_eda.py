"""Tests for RAGBench EDA helpers."""

from __future__ import annotations

from rag_kag.data_loaders.eda import recommended_chunk_params
from rag_kag.types import Example, Sentence


def test_recommended_chunk_params_from_lengths() -> None:
    # 70% of docs above 64 words → p30 should land around 64
    doc_lens = [50] * 30 + [100] * 70
    chunk_size, overlap = recommended_chunk_params(doc_lens)
    assert 32 <= chunk_size <= 100
    assert 8 <= overlap <= 64


def test_recommended_chunk_params_sentence_alignment() -> None:
    doc_lens = [200, 200, 200]
    sent_lens = [16] * 30
    chunk_size, _ = recommended_chunk_params(doc_lens, sentence_word_counts=sent_lens)
    assert chunk_size % 16 == 0 or chunk_size >= 32


def test_eda_word_count_via_example() -> None:
    from rag_kag.data_loaders.eda import iter_doc_word_counts

    ex = Example(
        id="1",
        domain="biomedical",
        subset="covidqa",
        question="what is covid",
        documents=["one two three four five"],
        documents_sentences=[
            [Sentence(key="0a", text="one two", doc_index=0), Sentence(key="0b", text="three four five", doc_index=0)]
        ],
    )
    docs, sents, qs = iter_doc_word_counts(iter([ex]))
    assert docs == [5]
    assert sents == [2, 3]
    assert qs == [3]
