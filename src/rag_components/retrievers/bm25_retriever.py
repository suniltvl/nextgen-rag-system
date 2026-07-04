from .base import BaseRetriever

class BM25Retriever(BaseRetriever):

    def __init__(self, bm25_index):
        self.index = bm25_index

    def retrieve(self, query: str, k: int = 5):
        return self.index.search(query, top_k=k)