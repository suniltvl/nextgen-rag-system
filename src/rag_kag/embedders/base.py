"""Embedder interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


class Embedder(ABC):
    """Maps text into a fixed-dim float32 vector space."""

    @property
    @abstractmethod
    def dim(self) -> int:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifier used for collection naming so a model swap re-indexes."""

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Embed a batch of texts. Returns shape (len(texts), dim)."""

    def embed_query(self, text: str) -> NDArray[np.float32]:
        """Embed a single query. Defaults to `embed([text])[0]` but subclasses
        may override (some models use distinct query/passage encoders)."""
        return self.embed([text])[0]
