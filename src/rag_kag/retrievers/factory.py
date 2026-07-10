"""Build a retriever from its config."""

from __future__ import annotations

from rag_kag.config import DenseRetrieverCfg, RetrieverCfg, BM25RetrieverCfg, HybridRetrieverCfg
from rag_kag.embedders.base import Embedder
from rag_kag.retrievers.base import Retriever
from rag_kag.retrievers.dense import DenseRetriever
from rag_kag.retrievers.bm25 import BM25Retriever
from rag_kag.retrievers.hybrid import HybridRetriever
from rag_kag.vectorstores.base import VectorStore


def build_retriever(
    cfg: RetrieverCfg,
    *,
    embedder: Embedder,
    vectorstore: VectorStore,
) -> Retriever:
    if isinstance(cfg, DenseRetrieverCfg):
        return DenseRetriever(embedder=embedder, store=vectorstore)
    if isinstance(cfg, BM25RetrieverCfg):
        return BM25Retriever()
    if isinstance(cfg, HybridRetrieverCfg):
        return HybridRetriever(dense_retriever=DenseRetriever, bm25_retriever=BM25Retriever)
    raise NotImplementedError(
        f"Retriever kind {cfg.kind!r} not yet implemented (BM25/hybrid/HyDE land in V4-V5)."
    )
