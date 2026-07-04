
from .base import BaseChunker

class FixedSizeChunker(BaseChunker):

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str) -> list[str]:

        ar_doc_text =[]
        chunks = []

        start = 0

        # self.debug(f"Text {type(text)} length: {len(text)}")

        for text_item in text:
            # self.debug(f"Text item {type(text_item)} length: {len(text_item)}")
            # self.info("========================================")
            while start < len(text_item):

                end = start + self.chunk_size
                chunk = text_item[start:end]
                # self.debug(f"Created chunk from index {start} to {end}, chunk length: {len(chunk)}, chunk content: {chunk[:30]}...")
                chunks.append(chunk)

                start += self.chunk_size - self.chunk_overlap

            return chunks
    
    def chunk_list(self, texts: list[str]) -> list[str]:
        all_chunks = []
        for text in texts:
            chunks = self.chunk(text)
            all_chunks.extend(chunks)
        return all_chunks
    
    # def chunk(self, text: str) -> list[str]:
    #     self.debug(f"Making chunks for text of length {len(text)}")
    #     return self.chunk(text)