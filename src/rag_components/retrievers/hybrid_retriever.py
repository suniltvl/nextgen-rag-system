from .base import BaseRetriever

class HybridRetriever(BaseRetriever):

    def __init__(
        self,
        vector_store,
        bm25_retriever,
        alpha=0.5
    ):
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.alpha = alpha

    def retrieve(self, query: str, k: int = 5):

        dense_results = self.vector_store.query(
            query=query,
            k=k
        )

        sparse_results = self.bm25_retriever.retrieve(
            query=query,
            k=k
        )

        return dense_results, sparse_results
