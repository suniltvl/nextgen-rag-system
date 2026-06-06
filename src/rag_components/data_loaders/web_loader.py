
from .base import BaseDataLoader

class WebLoader(BaseDataLoader):
    def __init__(self, url: str):
        self.url = url
    
    def load(self):
        
        pass