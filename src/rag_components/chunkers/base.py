
from abc import ABC, abstractmethod
from src.utils import Logger


class BaseChunker(ABC, Logger):
    

    @abstractmethod
    def chunk(self, text: str) -> list[str]:
        pass
    