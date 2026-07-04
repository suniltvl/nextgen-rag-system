
from venv import logger

from src.utils.logger import Logger
from .base import BaseDataLoader
from datasets import load_dataset
from src.models.enums import DataSetType
from src.utils.helper import helper


class LocalLoader(BaseDataLoader):
    def __init__(self,
                 subset: str | None = None,
                 split: str = "test",
                 data_dir: str | None = None,                 
                 file_extension: str = "parquet",                 
                 ):
        super().__init__()
        self.subset = subset
        self.split = split
        self.data_dir = helper.get_dir_in_project(data_dir) if data_dir else data_dir
        self.file_extension = file_extension

    def load(self):
        self.info("Loading local data...")

        test_data_files = f"{self.data_dir}/{self.subset}/{DataSetType.TEST.value}*.{self.file_extension}"
        # self.info(f"Test data files: {test_data_files}")

        train_data_files = f"{self.data_dir}/{self.subset}/{DataSetType.TRAIN.value}*.{self.file_extension}"
        # self.info(f"Train data files: {train_data_files}")

        validation_data_files = f"{self.data_dir}/{self.subset}/{DataSetType.VALIDATION.value}*.{self.file_extension}"
        # self.info(f"Validation data files: {validation_data_files}")

        data_files_path = {
            "test": test_data_files,
            "train": train_data_files,
            "validation": validation_data_files
        }

        loader = load_dataset(
            self.file_extension,
            data_files=data_files_path,
            split=self.split
        )
        
        self.debug(f"Loader created: {loader}")
        
        return loader
    