from typing import Sequence

from rank_bm25 import BM25Okapi

from rag_kag.types import Chunk, RetrievedChunk


class BM25Retriever:
    """A retriever that uses BM25 for lexical similarity with namespace support."""

    def __init__(self):
        """Initialize the BM25Retriever.

        """
        self._bm25_by_namespace: dict[str, BM25Okapi] = {}
        self._chunks_by_namespace: dict[str, Sequence[Chunk]] = {}

    def index(self, namespace: str, chunks: Sequence[Chunk]) -> None:
        """Index chunks for a specific namespace.

        Args:
            namespace: The unique identifier for the set of chunks (e.g., example ID).
            chunks: A sequence of Chunk objects to index.
        """
        self._chunks_by_namespace[namespace] = chunks
        tokenized_corpus = [chunk.text.split() for chunk in chunks]
        self._bm25_by_namespace[namespace] = BM25Okapi(tokenized_corpus)

    def retrieve(self, namespace: str, query: str, top_k: int) -> list[RetrievedChunk]:
        """Retrieve the top k chunks for a given query and namespace.

        Args:
            query: The query string.
            namespace: The namespace to retrieve chunks from.
            top_k: The number of top chunks to retrieve.

        Returns:
            A list of the top k RetrievedChunk objects with their scores.
        """
        if namespace not in self._bm25_by_namespace:
            raise RuntimeError(f"No BM25 model found for namespace '{namespace}'. Call index() first.")

        bm25_model = self._bm25_by_namespace[namespace]
        chunks = self._chunks_by_namespace[namespace]

        tokenized_query = query.split()
        doc_scores = bm25_model.get_scores(tokenized_query)

        # Combine scores with chunk indices and sort by score in descending order
        scored_chunks = sorted(
            [(score, i) for i, score in enumerate(doc_scores)],
            key=lambda x: x[0],
            reverse=True,
        )

        # Take the top k and convert to RetrievedChunk objects
        retrieved_results = []
        for score, original_index in scored_chunks[:top_k]: # Fixed: Changed self._k to top_k
            original_chunk = chunks[original_index]
            retrieved_results.append(
                RetrievedChunk(
                    chunk=original_chunk,
                    score=score,
                    rank=original_index + 1
                )
            )
        return retrieved_results
