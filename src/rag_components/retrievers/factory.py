from src.models import RetrievalType
from .vector_retriever import VectorRetriever
from .hybrid_retriever import HybridRetriever
from .bm25_retriever import BM25Retriever


class RetrieverFactory:

    @staticmethod
    def create(
        config,
        vector_store=None,
        bm25_index=None
    ):

        retriever_type = config.type.value if hasattr(config.type, "value") else str(config.type).lower()

        if retriever_type == RetrievalType.SIMILARITY.value:
            return VectorRetriever(vector_store)

        elif retriever_type == RetrievalType.HYBRID.value:
            return HybridRetriever(
                vector_store=vector_store,
                bm25_retriever=BM25Retriever(bm25_index),
                alpha=config.alpha
            )

        elif retriever_type == RetrievalType.BM25.value:
            return BM25Retriever(bm25_index)

        raise ValueError(
            f"Unsupported retriever type: {retriever_type}"
        )
