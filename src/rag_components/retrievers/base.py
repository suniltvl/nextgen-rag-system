from abc import ABC, abstractmethod

class BaseRetriever(ABC):

    @abstractmethod
    def retrieve(self, query: str, k: int = 5, alpha: float = 0.3):
        pass