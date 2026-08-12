"""Sliding-window chunker — the V1 baseline.

Operates over whitespace-tokenized words for portability (no tokenizer dep).
The embedder's tokenizer enforces the real model-side limit downstream.
"""

from __future__ import annotations

from rag_kag.chunkers.base import Chunker
from rag_kag.types import Chunk, Example, Sentence


class SlidingWindowChunker(Chunker):
    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be in [0, chunk_size)")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, example: Example) -> list[Chunk]:
        chunks: list[Chunk] = []
        for doc_idx, doc_text in enumerate(example.documents):
            doc_sents = (
                example.documents_sentences[doc_idx]
                if doc_idx < len(example.documents_sentences)
                else []
            )
            chunks.extend(
                self._chunk_document(
                    doc_text,
                    doc_idx=doc_idx,
                    sentences=doc_sents,
                    example_id=example.id,
                )
            )
        return chunks

    def _chunk_document(
        self,
        text: str,
        *,
        doc_idx: int,
        sentences: list[Sentence],
        example_id: str,
    ) -> list[Chunk]:
        words = text.split()
        if not words:
            return []
        step = self.chunk_size - self.overlap
        out: list[Chunk] = []
        for start in range(0, len(words), step):
            window = words[start : start + self.chunk_size]
            if not window:
                break
            chunk_text = " ".join(window)
            sentence_keys = self._covered_sentence_keys(chunk_text, sentences)
            chunk_id = f"{example_id}::d{doc_idx}::c{start:06d}"
            out.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    doc_index=doc_idx,
                    sentence_keys=sentence_keys,
                    metadata={
                        "example_id": example_id,
                        "doc_index": doc_idx,
                        "word_start": start,
                        "word_end": start + len(window),
                    },
                )
            )
            if start + self.chunk_size >= len(words):
                break
        return out

    @staticmethod
    def _covered_sentence_keys(chunk_text: str, sentences: list[Sentence]) -> list[str]:
        """A chunk "covers" a sentence if the sentence text appears within it.

        Cheap substring match. Good enough for sliding-window chunks where
        sentence boundaries usually fall inside the window. Sentence-aware
        chunkers will replace this with exact provenance.
        """
        keys: list[str] = []
        for s in sentences:
            if s.text and s.text in chunk_text:
                keys.append(s.key)
        return keys
