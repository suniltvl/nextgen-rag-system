"""Build a vector store from its config."""

from __future__ import annotations

from rag_kag.config import ChromaVectorStoreCfg, Secrets, VectorStoreCfg
from rag_kag.vectorstores.base import VectorStore
from rag_kag.vectorstores.chroma import ChromaVectorStore


def build_vectorstore(cfg: VectorStoreCfg, secrets: Secrets | None = None) -> VectorStore:
    """Build a vector store. In-memory by default (per-example collections);
    set `persist_dir` in the YAML config to enable on-disk persistence."""
    del secrets  # currently unused; reserved for future cloud-backed stores
    if isinstance(cfg, ChromaVectorStoreCfg):
        persist = str(cfg.persist_dir) if cfg.persist_dir else None
        return ChromaVectorStore(collection_prefix=cfg.collection_prefix, persist_dir=persist)
    raise NotImplementedError(f"Vector store kind {cfg.kind!r} not supported.")
