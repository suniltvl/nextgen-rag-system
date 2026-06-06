from typing import List

from pydantic import BaseModel, Field

from .enums import ChunkingStrategy, EmbeddingProvider, GeneratorProvider, RetrievalType, VectorStoreProvider, DataLoaderSource


class ExperimentConfig(BaseModel):
    name: str
    version: float


class PipelineConfig(BaseModel):
    query_classification: bool = False
    reranking: bool = False
    summarization: bool = False


class DataLoaderConfig(BaseModel):
    source: DataLoaderSource = DataLoaderSource.LOCAL
    path: str | None = None
    dataset_name: str | None = None
    subset: str | None = None
    split: str | None = None
    cache_dir: str | None = None
    data_dir: str | None = None # For Local loader to specify the data directory
    streaming: bool = True
    file_extension: str = ""


class ChunkingConfig(BaseModel):
    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE

    chunk_size: int = Field(gt=0)
    chunk_overlap: int = Field(ge=0)


class EmbeddingConfig(BaseModel):
    provider: EmbeddingProvider = EmbeddingProvider.HUGGINGFACE

    model: str


class VectorStoreConfig(BaseModel):
    provider: VectorStoreProvider = VectorStoreProvider.CHROMA

    collection_name: str


class RetrievalConfig(BaseModel):
    type: RetrievalType = RetrievalType.SIMILARITY

    top_k: int = Field(gt=0)


class GeneratorConfig(BaseModel):
    provider: GeneratorProvider

    model: str
    temperature: float = Field(ge=0.0, le=2.0)


class EvaluationConfig(BaseModel):
    dataset: str
    metrics: List[str]


class RAGConfig(BaseModel):
    experiment: ExperimentConfig
    pipeline: PipelineConfig
    data_loader: DataLoaderConfig | None = None
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    vector_store: VectorStoreConfig
    retrieval: RetrievalConfig
    generator: GeneratorConfig
    evaluation: EvaluationConfig