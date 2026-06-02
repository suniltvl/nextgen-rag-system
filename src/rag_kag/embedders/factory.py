"""Build an embedder from its config."""

from __future__ import annotations

from rag_kag.config import EmbedderCfg, SentenceTransformerEmbedderCfg
from rag_kag.embedders.base import Embedder
from rag_kag.embedders.sentence_transformer import SentenceTransformerEmbedder


def build_embedder(cfg: EmbedderCfg) -> Embedder:
    if isinstance(cfg, SentenceTransformerEmbedderCfg):
        return SentenceTransformerEmbedder(
            model_name=cfg.model_name,
            batch_size=cfg.batch_size,
            normalize=cfg.normalize,
            device=cfg.device,
        )
    raise NotImplementedError(f"Embedder kind {cfg.kind!r} not supported.")
