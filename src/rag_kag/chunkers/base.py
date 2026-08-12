"""Chunker interface. Every chunking strategy implements `chunk(example)`.

Chunkers consume an `Example` (which already has sentence-level structure
from RAGBench) and emit a list of `Chunk`s. Provenance — which source
sentences a chunk covers — is preserved on every chunk so the TRACe
context-relevance metric can reuse the dataset's `all_relevant_sentence_keys`
without re-annotation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag_kag.types import Chunk, Example


class Chunker(ABC):
    """Abstract chunker."""

    @abstractmethod
    def chunk(self, example: Example) -> list[Chunk]:
        """Produce a list of chunks for the given example."""

    def chunk_text(
        self,
        text: str,
        *,
        chunk_id_prefix: str,
        doc_index: int = 0,
    ) -> list[Chunk]:
        """Convenience for tests / ad-hoc use without an Example."""
        ex = Example(
            id=chunk_id_prefix,
            domain="adhoc",
            subset="adhoc",
            question="",
            documents=[text],
            documents_sentences=[[]],
        )
        return self.chunk(ex)
