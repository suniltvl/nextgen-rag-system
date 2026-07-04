from abc import ABC, abstractmethod
from src.utils import Logger


class BaseGenerator(ABC, Logger):

    @abstractmethod
    def generate(
        self,
        question: str,
        context: str,
        system_prompt: str | None = None
    ) -> str:
        pass