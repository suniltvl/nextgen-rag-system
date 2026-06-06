# src/rag_components/data_loaders/factory.py

from .huggingface_loader import HuggingFaceDataLoader
from .local_loader import LocalLoader
from .web_loader import WebLoader
from src.models import DataLoaderSource


class DataLoaderFactory:

    @staticmethod
    def create(config):

        source = config.source

        if source == DataLoaderSource.HUGGINGFACE:
            return HuggingFaceDataLoader(
                dataset_name=config.dataset_name,
                subset=config.subset,
                split=config.split,
                cache_dir=config.cache_dir,
                streaming=config.streaming,
            )

        elif source == DataLoaderSource.LOCAL:
            return LocalLoader(
                subset=config.subset,
                split=config.split,
                data_dir=config.data_dir,
                file_extension=config.file_extension
            )

        elif source == DataLoaderSource.WEB:
            return WebLoader(config.url)

        raise ValueError(f"Unsupported source: {source}")