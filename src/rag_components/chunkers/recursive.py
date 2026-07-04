from langchain_text_splitters import RecursiveCharacterTextSplitter

from .base import BaseChunker


class RecursiveChunker(BaseChunker):

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ):
        super().__init__()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def chunk(self, text: str) -> list[str]:

        chunks = []
        for text_item in text:        
            
            splitter_text = self.splitter.split_text(text_item)
            
            print(f"Initializing RecursiveChunker splitter_text={splitter_text}")   
            chunks.append(splitter_text)
        return chunks
    
    def chunk_list(self, texts: list[str]) -> list[str]:
        all_chunks = []
        for text in texts:
            chunks = self.chunk(text)
            all_chunks.extend(chunks)
        return all_chunks