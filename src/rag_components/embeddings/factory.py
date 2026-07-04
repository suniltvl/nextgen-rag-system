# from .huggingface_embedder import HuggingFaceEmbeddings
# from .openai_embedder import OpenAIEmbedder
from src.models import EmbeddingProvider
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings


class EmbedderFactory:

    _instances = {}

    @staticmethod
    def clear_cache():
        """Clear the embedder cache."""
        EmbedderFactory._instances.clear()

    @staticmethod
    def create(config):
        cache_key = f"{config.provider}:{config.model}"
        provider = config.provider
        model_name = config.model


        if cache_key not in EmbedderFactory._instances:
            if provider == EmbeddingProvider.HUGGINGFACE:
                EmbedderFactory._instances[cache_key] = HuggingFaceEmbeddings(
                    model_name=config.model
                )

            elif provider == EmbeddingProvider.OPENAI:
                EmbedderFactory._instances[cache_key] = OpenAIEmbeddings(
                    model=config.model
                )

            else:
                self.logger.error(f"Unsupported provider: {provider}")
                raise ValueError(
                    f"Unsupported provider: {provider}"
                )

        return EmbedderFactory._instances[cache_key]
            
