"""Dense vector retriever — embedder + vector store, top-k by cosine sim."""

from __future__ import annotations

from rag_kag.embedders.base import Embedder
from rag_kag.retrievers.base import Retriever
from rag_kag.types import Chunk, RetrievedChunk
from rag_kag.vectorstores.base import VectorStore


class DenseRetriever(Retriever):
    def __init__(self, embedder: Embedder, store: VectorStore):
        self.embedder = embedder
        self.store = store
        # chunk_id → Chunk lookup, populated at index() time. Keeps the
        # downstream pipeline able to access the original `Chunk` (with
        # full sentence_keys provenance) even when the vector store only
        # returns ids and distances.
        self._chunk_index: dict[str, dict[str, Chunk]] = {}

    def index(self, namespace: str, chunks: list[Chunk]) -> None:
        self.store.reset(namespace)
        if not chunks:
            self._chunk_index[namespace] = {}
            return
        embeddings = self.embedder.embed([c.text for c in chunks])
        self.store.add(namespace, chunks, embeddings)
        self._chunk_index[namespace] = {c.chunk_id: c for c in chunks}

    def retrieve(self, namespace: str, query: str, top_k: int) -> list[RetrievedChunk]:
        if not self._chunk_index.get(namespace):
            return []
        q_vec = self.embedder.embed_query(query)
        hits = self.store.query(namespace, q_vec, top_k)
        out: list[RetrievedChunk] = []
        for rank, hit in enumerate(hits):
            chunk = self._chunk_index[namespace].get(hit.chunk_id)
            if chunk is None:
                # Cache miss — fall back to the store's record.
                chunks = self.store.get_chunks(namespace, [hit.chunk_id])
                if not chunks:
                    continue
                chunk = chunks[0]
            out.append(RetrievedChunk(chunk=chunk, score=hit.score, rank=rank))
        return out
