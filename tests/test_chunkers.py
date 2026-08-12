"""Sliding-window chunker — boundary cases + provenance preservation."""

from __future__ import annotations

from rag_kag.chunkers.sliding_window import SlidingWindowChunker
from rag_kag.types import Example, Sentence


def _make_example(text: str, sentences: list[tuple[str, str]]) -> Example:
    """Build an Example from a doc string + list of (key, sentence) tuples."""
    return Example(
        id="ex1",
        domain="test",
        subset="test",
        question="q?",
        documents=[text],
        documents_sentences=[[Sentence(key=k, text=t, doc_index=0) for k, t in sentences]],
    )


def test_sliding_window_basic_split() -> None:
    text = " ".join(f"w{i}" for i in range(100))
    ex = _make_example(text, sentences=[])
    chunker = SlidingWindowChunker(chunk_size=20, overlap=5)
    chunks = chunker.chunk(ex)
    # 100 words, step = 20-5 = 15. Windows start at 0,15,30,45,60,75,90.
    # The window starting at 90 has only 10 words but still gets emitted.
    assert len(chunks) == 7
    assert chunks[0].text.startswith("w0 w1")
    assert chunks[-1].text.endswith("w99")
    # Provenance fields present.
    assert all(c.metadata["doc_index"] == 0 for c in chunks)
    assert all(c.chunk_id.startswith("ex1::d0::c") for c in chunks)


def test_sliding_window_preserves_sentence_keys() -> None:
    sent_a = "the cat sat on the mat"
    sent_b = "the dog sat on the rug"
    text = f"{sent_a}. {sent_b}."
    ex = _make_example(text, sentences=[("0a", sent_a), ("0b", sent_b)])
    chunker = SlidingWindowChunker(chunk_size=64, overlap=0)
    chunks = chunker.chunk(ex)
    # Both sentences fit in one window.
    assert len(chunks) == 1
    assert set(chunks[0].sentence_keys) == {"0a", "0b"}


def test_sliding_window_rejects_bad_args() -> None:
    import pytest

    with pytest.raises(ValueError):
        SlidingWindowChunker(chunk_size=0)
    with pytest.raises(ValueError):
        SlidingWindowChunker(chunk_size=10, overlap=10)


def test_sliding_window_empty_doc() -> None:
    ex = _make_example("", sentences=[])
    chunker = SlidingWindowChunker(chunk_size=20, overlap=5)
    assert chunker.chunk(ex) == []
