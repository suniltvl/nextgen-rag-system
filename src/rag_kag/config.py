"""Pydantic config schema. Every experiment is a validated YAML file.

The discriminated `kind` fields let V2 swap chunker, V3 swap embedder, etc.
by editing one line of YAML — no Python changes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Secrets(BaseSettings):
    """Provider keys read from environment / .env. Never logged or pickled."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_region_name: str = "us-east-1"
    ollama_api_base: str = "http://localhost:11434"
    hf_token: str | None = None
    chroma_persist_dir: Path = Path("./indices/chroma")


# --- Component configs (discriminated by `kind`) ---------------------------


class SlidingWindowChunkerCfg(BaseModel):
    kind: Literal["sliding_window"] = "sliding_window"
    chunk_size: int = 512  # tokens (approx via whitespace; embedder tokenizer also enforces)
    overlap: int = 64


class SentenceAwareChunkerCfg(BaseModel):
    kind: Literal["sentence_aware"] = "sentence_aware"
    max_sentences: int = 5
    overlap_sentences: int = 1


class SemanticChunkerCfg(BaseModel):
    kind: Literal["semantic"] = "semantic"
    breakpoint_threshold: float = 0.75  # cosine-distance breakpoint
    min_sentences: int = 2
    max_sentences: int = 12


ChunkerCfg = SlidingWindowChunkerCfg | SentenceAwareChunkerCfg | SemanticChunkerCfg


class SentenceTransformerEmbedderCfg(BaseModel):
    kind: Literal["sentence_transformer"] = "sentence_transformer"
    model_name: str = "BAAI/bge-small-en-v1.5"
    batch_size: int = 64
    normalize: bool = True
    device: str | None = None  # auto-detect if None


EmbedderCfg = SentenceTransformerEmbedderCfg


class ChromaVectorStoreCfg(BaseModel):
    kind: Literal["chroma"] = "chroma"
    collection_prefix: str = "rag_kag"
    persist_dir: Path | None = None  # falls back to Secrets.chroma_persist_dir


VectorStoreCfg = ChromaVectorStoreCfg


class DenseRetrieverCfg(BaseModel):
    kind: Literal["dense"] = "dense"
    top_k: int = 5


class BM25RetrieverCfg(BaseModel):
    kind: Literal["bm25"] = "bm25"
    top_k: int = 5


class HybridRetrieverCfg(BaseModel):
    kind: Literal["hybrid"] = "hybrid"
    top_k: int = 5
    dense_weight: float = 0.5  # convex combination weight on dense scores


RetrieverCfg = DenseRetrieverCfg | BM25RetrieverCfg | HybridRetrieverCfg


class CrossEncoderRerankerCfg(BaseModel):
    kind: Literal["cross_encoder"] = "cross_encoder"
    model_name: str = "BAAI/bge-reranker-base"
    top_k: int = 5  # how many to keep after reranking


RerankerCfg = CrossEncoderRerankerCfg


class LLMGeneratorCfg(BaseModel):
    kind: Literal["llm"] = "llm"
    model: str = "ollama/llama3.1:8b"  # litellm-style identifier
    temperature: float = 0.0
    max_tokens: int = 512
    prompt_template: str = "grounded_default"


GeneratorCfg = LLMGeneratorCfg


# --- Top-level experiment config -------------------------------------------


class DataCfg(BaseModel):
    """Which RAGBench subset(s) to run on, and how much."""

    subset: str  # e.g. "covidqa", "finqa"
    split: str = "test"
    limit: int | None = None  # None = full split
    cache_dir: Path | None = None


class ExperimentCfg(BaseModel):
    """Top-level experiment description — one YAML file per V1..V8 variant."""

    name: str
    description: str = ""
    data: DataCfg
    chunker: ChunkerCfg = Field(default_factory=SlidingWindowChunkerCfg, discriminator="kind")
    embedder: EmbedderCfg = Field(
        default_factory=SentenceTransformerEmbedderCfg, discriminator="kind"
    )
    vectorstore: VectorStoreCfg = Field(
        default_factory=ChromaVectorStoreCfg, discriminator="kind"
    )
    retriever: RetrieverCfg = Field(default_factory=DenseRetrieverCfg, discriminator="kind")
    reranker: RerankerCfg | None = None
    generator: GeneratorCfg = Field(default_factory=LLMGeneratorCfg, discriminator="kind")
    output_dir: Path = Path("experiments/runs")
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentCfg:
        with open(path) as f:
            raw: dict[str, Any] = yaml.safe_load(f)
        return cls.model_validate(raw)
