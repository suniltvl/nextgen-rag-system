
from abc import ABC, abstractmethod
from typing import Any
from src.utils import Logger

class BaseDataLoader(ABC, Logger):

    @abstractmethod
    def load(self) -> Any:
        pass