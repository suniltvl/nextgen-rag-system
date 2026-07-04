

from abc import ABC, abstractmethod
from src.utils import Logger

class BaseVectorStore(ABC, Logger):

    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def add_documents(self, documents):
        """
        Add documents to the vector store
        """
        self.logger.info("Adding documents to vector store")
        pass
    
    @abstractmethod
    def query(self, query: str, n_results: int):
        """
        Query the vector store
        """
        self.logger.info("Querying vector store")
        pass
