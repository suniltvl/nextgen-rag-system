"""Generator interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag_kag.types import GenerationResult, RetrievedChunk


class Generator(ABC):
    @abstractmethod
    def generate(self, question: str, retrieved: list[RetrievedChunk]) -> GenerationResult:
        ...
