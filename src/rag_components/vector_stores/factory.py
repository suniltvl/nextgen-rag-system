from src.models import VectorStoreProvider
from src.utils import helper
from src.utils.logger import Logger
# from .faiss_store import FaissStore
from .chroma_db_store import ChromaDBStore
# from .qdrant_store import QdrantStore


class VectorStoreAdapter:

    def __init__(self, store):
        self.store = store

    def add_documents(self, documents, metadatas=None):
        return self.store.add_documents(documents, metadatas=metadatas)

    def query(self, query, k=5):
        return self.store.query(query=query, k=k)

    def get_collection(self):
        if hasattr(self.store, "get_collection"):
            return self.store.get_collection()
        raise AttributeError("Underlying vector store does not expose get_collection()")


class VectorStoreFactory:

    # _stores = {
    #     "faiss": FaissStore,
    #     "chroma": ChromaStore,
    #     "qdrant": QdrantStore,
    # }

    @classmethod
    def create(self, config, embeddings):
        provider = config.provider
        collection_name = config.collection_name
        persist_directory = helper.create_dir(config.persist_directory) if config.persist_directory else None

        if(persist_directory is None):
            Logger.warning("Using in-memory vector store")


        match provider:
            # case VectorStoreType.FAISS:
            #     return FaissStore(embeddings)
            case VectorStoreProvider.CHROMA:
                return VectorStoreAdapter(
                    ChromaDBStore(embeddings, collection_name, persist_directory)
                )
            # case VectorStoreType.QDRANT:
            #     return QdrantStore(embeddings)
            case _:
                raise ValueError(
                    f"Unsupported vector store: {provider}"
                )

        # This line should never be reached due to the default case in match
        raise ValueError(f"Unsupported vector store: {provider}")
