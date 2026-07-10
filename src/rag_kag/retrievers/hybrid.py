import math
from typing import Sequence, List, Dict, Optional

from rag_kag.types import Chunk, RetrievedChunk
from rag_kag.retrievers.dense import DenseRetriever
from rag_kag.retrievers.bm25 import BM25Retriever


class HybridRetriever:
    """
    A hybrid retriever that combines results from DenseRetriever and BM25Retriever.
    """

    def __init__(self, dense_retriever: DenseRetriever, bm25_retriever: BM25Retriever):
        """
        Initializes the HybridRetriever with instances of DenseRetriever and BM25Retriever.

        Args:
            dense_retriever: An initialized instance of DenseRetriever.
            bm25_retriever: An initialized instance of BM25Retriever.
        """
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self._chunks_by_namespace: Dict[str, Sequence[Chunk]] = {}

    def index(self, namespace: str, chunks: Sequence[Chunk]) -> None:
        """
        Indexes chunks using both the dense and BM25 retrievers.

        Args:
            namespace: The unique identifier for the set of chunks (e.g., example ID).
            chunks: A sequence of Chunk objects to index.
        """
        self.bm25_retriever.index(namespace, chunks)
        self.dense_retriever.index(namespace, chunks)
        self._chunks_by_namespace[namespace] = chunks

    def retrieve(self, namespace: str, query: str, top_k: int, dense_weight: float = 0.5) -> List[RetrievedChunk]:
        """
        Retrieves and combines chunks from BM25 and Dense retrievers using a convex linear combination.
        The function first runs BM25, then Dense retrieval. Results are combined,
        scores are combined using dense_weight, and then re-ranked by the combined score.

        Args:
            namespace: The namespace to retrieve chunks from.
            query: The query string.
            top_k: The number of top chunks to retrieve.
            dense_weight: The weight given to the dense retriever score (between 0 and 1).

        Returns:
            A list of the top k RetrievedChunk objects, combined and re-ranked.
        """
        if not (0 <= dense_weight <= 1):
            raise ValueError("dense_weight must be between 0 and 1.")

        if namespace not in self._chunks_by_namespace:
            raise RuntimeError(f"No chunks indexed for namespace '{namespace}'. Call index() first.")

        num_total_chunks = len(self._chunks_by_namespace[namespace])

        # Retrieve all chunks with their scores from both retrievers
        bm25_results = self.bm25_retriever.retrieve(namespace, query, top_k=num_total_chunks)
        dense_results = self.dense_retriever.retrieve(namespace, query, top_k=num_total_chunks)

        # Create a mapping from chunk content to scores
        chunk_data: Dict[str, Dict[str, float]] = {}
        chunk_map: Dict[str, Chunk] = {} # To store the actual chunk object

        for rc in bm25_results:
            chunk_content = rc.chunk.text
            if chunk_content not in chunk_data:
                chunk_data[chunk_content] = {'bm25_score': 0.0, 'dense_score': 0.0}
                chunk_map[chunk_content] = rc.chunk
            chunk_data[chunk_content]['bm25_score'] = rc.score

        for rc in dense_results:
            chunk_content = rc.chunk.text
            if chunk_content not in chunk_data:
                chunk_data[chunk_content] = {'bm25_score': 0.0, 'dense_score': 0.0}
                chunk_map[chunk_content] = rc.chunk
            chunk_data[chunk_content]['dense_score'] = rc.score

        # Calculate combined scores
        hybrid_retrieved_chunks: List[RetrievedChunk] = []
        for chunk_content, scores in chunk_data.items():
            bm25_score = scores.get('bm25_score', 0.0)
            dense_score = scores.get('dense_score', 0.0)

            # Convex linear combination
            combined_score = (dense_weight * dense_score) + ((1 - dense_weight) * bm25_score)

            hybrid_retrieved_chunks.append(
                RetrievedChunk(
                    chunk=chunk_map[chunk_content],
                    score=combined_score,
                    rank=0 # Will be updated after sorting
                )
            )

        # Sort by combined score in descending order
        hybrid_retrieved_chunks.sort(key=lambda x: x.score, reverse=True)

        # Assign ranks and return top_k
        final_results: List[RetrievedChunk] = []
        for i, rc in enumerate(hybrid_retrieved_chunks[:top_k]):
            final_results.append(
                RetrievedChunk(
                    chunk=rc.chunk,
                    score=rc.score,
                    rank=i + 1
                )
            )
        return final_results
