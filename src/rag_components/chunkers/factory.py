from venv import logger

from src.models.enums import ChunkingStrategy

from .fixed_size import FixedSizeChunker
from .recursive import RecursiveChunker

class ChunkingFactory:

    @staticmethod
    def create(config):
        strategy = config.strategy
        chunk_size = config.chunk_size
        chunk_overlap = config.chunk_overlap

        logger.debug(f"Creating chunker with strategy: {strategy}, chunk_size: {chunk_size}, chunk_overlap: {chunk_overlap}")

        if strategy == ChunkingStrategy.FIXED_SIZE:
            logger.debug(f"Creating chunker with strategy: {strategy}, chunk_size: {chunk_size}, chunk_overlap: {chunk_overlap}")
            return FixedSizeChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        elif strategy == ChunkingStrategy.RECURSIVE:
            return RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        
        else:
            print(f"Unknown chunking strategy: {strategy}")
            raise ValueError(f"Unknown chunking strategy: {strategy}")