from .base import BaseVectorStore
# from langchain_chroma import Chroma
from langchain_community.vectorstores import Chroma

from src.utils import helper


class ChromaDBStore(BaseVectorStore):

    chroma_client = None
   
    def __init__(self, embeddings,
        collection_name: str,
        persist_directory: str):
        super().__init__()
        print("Initializing ChromaDBStore")

        self.embeddings = embeddings
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        
        if(self.chroma_client is None):
            self.chroma_client = self._get_vector_client()


    def _get_vector_client(self):
        db_path_relative = self.persist_directory
        is_db_path_exist = helper.is_dir_in_project(db_path_relative)

        print(f"db path: {db_path_relative}")
        print(f"Is db path exist: {is_db_path_exist}")

        if not is_db_path_exist:
            helper.create_dir(db_path_relative)
            
        db_path = helper.get_dir_in_project(db_path_relative)

        chroma_client = Chroma(
                    collection_name=self.collection_name,
                    persist_directory=str(db_path),
                    embedding_function=self.embeddings
                )

        chroma_client._client.get_or_create_collection(self.collection_name)
                
        return chroma_client

    def get_collection(self):
        return self.chroma_client._client.get_or_create_collection(
            name=self.collection_name
        )


    def add_documents(self, documents, metadatas=None):
        collection = self.chroma_client._client.get_or_create_collection(
            name=self.collection_name
        )
        
        ids = [f"doc_{i}" for i in range(len(documents))]

        add_kwargs = {
            "ids": ids,
            "documents": documents,
        }
        if metadatas is not None:
            add_kwargs["metadatas"] = metadatas

        collection.add(**add_kwargs)

    def query(self, query, k=5):
        
        collection = self.chroma_client._client.get_or_create_collection(
            name=self.collection_name
        )

        results = collection.query(
            query_texts=[query],
            n_results=k
        )
        return results
