from enum import Enum


class ChunkingStrategy(str, Enum):
    FIXED_SIZE = "fixed_size"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    SENTENCE = "sentence"
    TOKEN = "token"

class DataLoaderSource(str, Enum):
    LOCAL = "local"
    S3 = "s3"
    GCS = "gcs"
    AZURE = "azure"
    HUGGINGFACE = "huggingface"
    WEB = "web"

class EmbeddingProvider(str, Enum):
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"


class VectorStoreProvider(str, Enum):
    CHROMA = "chroma"
    QDRANT = "qdrant"
    MILVUS = "milvus"
    FAISS = "faiss"

class RetrievalType(str, Enum):
    SIMILARITY = "similarity"
    MMR = "mmr"
    HYBRID = "hybrid"
    BM25 = "bm25"

class GeneratorProvider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    GOOGLE = "google"
    LMSTUDIO = "lmstudio"
    OPENROUTER = "openrouter"
    CEREBRAS = "cerebras"

class DataSetType(str, Enum):
    TRAIN = "train"
    TEST = "test"
    VALIDATION = "validation"