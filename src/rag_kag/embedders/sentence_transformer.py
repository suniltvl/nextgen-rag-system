"""sentence-transformers backed embedder. Lazy-loads the model on first call."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from rag_kag.embedders.base import Embedder


class SentenceTransformerEmbedder(Embedder):
    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        batch_size: int = 64,
        normalize: bool = True,
        device: str | None = None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self.device = device
        self._model = None
        self._dim: int | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        # Imported lazily so the package import stays cheap (and tests that
        # don't touch embeddings don't pay the torch import cost).
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.model_name, device=self.device)
        self._dim = int(self._model.get_sentence_embedding_dimension())

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._load()
        assert self._dim is not None
        return self._dim

    @property
    def name(self) -> str:
        # Slashes break Chroma collection names; replace with double-underscore.
        return self.model_name.replace("/", "__")

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        self._load()
        assert self._model is not None
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.astype(np.float32, copy=False)
