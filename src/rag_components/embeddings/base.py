# embeddings/base.py

from abc import ABC, abstractmethod
from typing import List

class BaseEmbedder(ABC):

    @abstractmethod
    def embed_documents(self, texts: List[str]):
        pass

    @abstractmethod
    def embed_query(self, text: str):
        pass