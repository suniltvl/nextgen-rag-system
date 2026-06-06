# src/rag_components/data_loaders/huggingface_loader.py

from datasets import load_dataset

from .base import BaseDataLoader


class HuggingFaceDataLoader(BaseDataLoader):

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        subset: str | None = None,
        cache_dir: str | None = "./data/hf_cache",
        data_dir: str | None = None,
        streaming: bool = True
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.split = split
        self.subset = subset
        self.cache_dir = cache_dir
        self.data_dir = data_dir
        self.streaming = streaming
    def load(self):
        self.info(f"Loading HuggingFace dataset: {self.dataset_name}")

        loader = None
        if self.subset:
            loader = load_dataset(
                self.dataset_name,
                self.subset,
                split=self.split,
                cache_dir=self.cache_dir,
                streaming=self.streaming

            )

        if loader is None:
            loader = load_dataset(
                self.dataset_name,
                split=self.split,
                cache_dir=self.cache_dir,
                data_dir=self.data_dir,
                streaming=self.streaming
            )

        
        self.debug(f"Loader created: {loader}")
        
        return loader