"""VectorStore interface — minimal contract used by retrievers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from rag_kag.types import Chunk


@dataclass(slots=True)
class VectorHit:
    chunk_id: str
    score: float


class VectorStore(ABC):
    """A per-example collection of embedded chunks plus query."""

    @abstractmethod
    def reset(self, namespace: str) -> None:
        """Drop and recreate the namespace (per-example index)."""

    @abstractmethod
    def add(
        self,
        namespace: str,
        chunks: Iterable[Chunk],
        embeddings: NDArray[np.float32],
    ) -> None:
        ...

    @abstractmethod
    def query(
        self,
        namespace: str,
        query_embedding: NDArray[np.float32],
        top_k: int,
    ) -> list[VectorHit]:
        ...

    @abstractmethod
    def get_chunks(self, namespace: str, chunk_ids: list[str]) -> list[Chunk]:
        ...
