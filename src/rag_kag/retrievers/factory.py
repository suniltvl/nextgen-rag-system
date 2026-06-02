"""Build a retriever from its config."""

from __future__ import annotations

from rag_kag.config import DenseRetrieverCfg, RetrieverCfg
from rag_kag.embedders.base import Embedder
from rag_kag.retrievers.base import Retriever
from rag_kag.retrievers.dense import DenseRetriever
from rag_kag.vectorstores.base import VectorStore


def build_retriever(
    cfg: RetrieverCfg,
    *,
    embedder: Embedder,
    vectorstore: VectorStore,
) -> Retriever:
    if isinstance(cfg, DenseRetrieverCfg):
        return DenseRetriever(embedder=embedder, store=vectorstore)
    raise NotImplementedError(
        f"Retriever kind {cfg.kind!r} not yet implemented (BM25/hybrid/HyDE land in V4-V5)."
    )
