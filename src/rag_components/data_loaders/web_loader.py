
import http

from utils import helper
from .base import BaseDataLoader

class WebLoader(BaseDataLoader):
    def __init__(self, base_url, file_extension, subset, split, data_dir):
        self.base_url = base_url
        self.file_extension = file_extension
        self.subset = subset
        self.split = split
        self.data_dir = data_dir

    
    def load(self):

        lst_split = self.split if self.split and self.split.__class__.__name__ == "list" else [self.split]

        base_data_dir_path = helper.get_dir_in_project(self.data_dir)

        for split_item in lst_split:
            final_url = f"{self.base_url}/{data_dir_path}/{split_item}*.{self.file_extension}"

            conn = http.client.HTTPSConnection(self.base_url)
            conn.request("GET", final_url)
            response = conn.getresponse()
            print(response.read())


        
        pass