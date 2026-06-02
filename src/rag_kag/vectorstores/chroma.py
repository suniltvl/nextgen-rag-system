"""Chroma-backed vector store.

Each RAGBench example has its own scoped corpus (the documents shipped with
the question), so indexing semantics are per-example. We use one Chroma
collection per ``namespace`` (typically the example id). For V1 we use the
in-memory client — the build cost per example is small and we want the
experiment loop to be hermetic. Swap to ``PersistentClient`` later if we
re-run experiments that share a chunker × embedder pair.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rag_kag.types import Chunk
from rag_kag.vectorstores.base import VectorHit, VectorStore


_COLLECTION_SAFE = re.compile(r"[^A-Za-z0-9_-]+")


def _safe_name(s: str) -> str:
    """Chroma collection names must be 3-63 chars, alnum/_/-."""
    cleaned = _COLLECTION_SAFE.sub("_", s).strip("_-")
    if len(cleaned) < 3:
        cleaned = f"col_{cleaned}"
    return cleaned[:63]


class ChromaVectorStore(VectorStore):
    def __init__(self, collection_prefix: str = "rag_kag", persist_dir: str | None = None):
        # Lazy import keeps `rag_kag` import-light.
        import chromadb

        self.collection_prefix = collection_prefix
        if persist_dir:
            self._client = chromadb.PersistentClient(path=persist_dir)
        else:
            self._client = chromadb.EphemeralClient()

    def _coll_name(self, namespace: str) -> str:
        return _safe_name(f"{self.collection_prefix}_{namespace}")

    def reset(self, namespace: str) -> None:
        name = self._coll_name(namespace)
        try:
            self._client.delete_collection(name)
        except Exception:
            # Collection didn't exist — that's fine.
            pass
        # Cosine: matches normalized embeddings from bge / e5 family.
        self._client.create_collection(name=name, metadata={"hnsw:space": "cosine"})

    def add(
        self,
        namespace: str,
        chunks: Iterable[Chunk],
        embeddings: NDArray[np.float32],
    ) -> None:
        chunks_list = list(chunks)
        if not chunks_list:
            return
        if len(chunks_list) != embeddings.shape[0]:
            raise ValueError(
                f"chunk/embedding count mismatch: {len(chunks_list)} vs {embeddings.shape[0]}"
            )
        coll = self._client.get_collection(self._coll_name(namespace))
        coll.add(
            ids=[c.chunk_id for c in chunks_list],
            embeddings=embeddings.tolist(),
            documents=[c.text for c in chunks_list],
            metadatas=[self._chunk_metadata(c) for c in chunks_list],
        )

    def query(
        self,
        namespace: str,
        query_embedding: NDArray[np.float32],
        top_k: int,
    ) -> list[VectorHit]:
        coll = self._client.get_collection(self._coll_name(namespace))
        # Chroma may have fewer items than top_k for short examples.
        n = min(top_k, coll.count())
        if n == 0:
            return []
        result = coll.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n,
            include=["distances"],
        )
        ids = result["ids"][0]
        # Cosine distance in [0,2]; convert to similarity in [-1,1] via 1 - d.
        distances = result["distances"][0]
        return [VectorHit(chunk_id=cid, score=1.0 - float(d)) for cid, d in zip(ids, distances, strict=True)]

    def get_chunks(self, namespace: str, chunk_ids: list[str]) -> list[Chunk]:
        if not chunk_ids:
            return []
        coll = self._client.get_collection(self._coll_name(namespace))
        result = coll.get(ids=chunk_ids, include=["documents", "metadatas"])
        # Chroma preserves the requested order in `ids`, but to be safe we
        # rebuild a lookup keyed by chunk_id.
        by_id: dict[str, Chunk] = {}
        for cid, doc, meta in zip(result["ids"], result["documents"], result["metadatas"], strict=True):
            by_id[cid] = self._chunk_from_record(cid, doc, meta or {})
        return [by_id[cid] for cid in chunk_ids if cid in by_id]

    # --- helpers ---------------------------------------------------------

    @staticmethod
    def _chunk_metadata(c: Chunk) -> dict[str, Any]:
        meta = dict(c.metadata)
        # Chroma metadata values must be primitives. Sentence keys are joined.
        meta["doc_index"] = c.doc_index
        meta["sentence_keys"] = ",".join(c.sentence_keys)
        return meta

    @staticmethod
    def _chunk_from_record(chunk_id: str, document: str, metadata: dict[str, Any]) -> Chunk:
        sk = metadata.get("sentence_keys", "")
        sentence_keys = [k for k in sk.split(",") if k] if isinstance(sk, str) else []
        return Chunk(
            chunk_id=chunk_id,
            text=document or "",
            doc_index=int(metadata.get("doc_index", 0)),
            sentence_keys=sentence_keys,
            metadata={k: v for k, v in metadata.items() if k not in {"doc_index", "sentence_keys"}},
        )
