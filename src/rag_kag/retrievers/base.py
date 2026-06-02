"""Retriever interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag_kag.types import Chunk, RetrievedChunk


class Retriever(ABC):
    """Index a set of chunks for one example, then answer queries against them."""

    @abstractmethod
    def index(self, namespace: str, chunks: list[Chunk]) -> None:
        ...

    @abstractmethod
    def retrieve(self, namespace: str, query: str, top_k: int) -> list[RetrievedChunk]:
        ...
