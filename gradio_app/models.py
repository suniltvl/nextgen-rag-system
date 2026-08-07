"""Typed data models for the RAG dashboard.

These dataclasses define the shape of data flowing from the data layer
(currently JSON files, later a FastAPI backend) into the UI. Keeping the
shape typed here means `data_loader.py` can be swapped for real API calls
without any change to `ui.py` or `callbacks.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RetrievedDocument:
    rank: int
    similarity_score: float
    document_name: str
    page_number: int
    chunk_id: str
    chunk_length: int
    retrieved_text: str


@dataclass
class PipelineConfig:
    chunk_strategy: str
    chunk_size: int
    chunk_overlap: int
    embedding_model: str
    embedding_dimension: int
    vector_database: str
    retrieval_type: str
    top_k: int
    generator_llm: str
    evaluation_llm: str


@dataclass
class EvaluationMetrics:
    overall_score: float
    faithfulness: float
    context_relevance: float
    context_utilization: float
    answer_completeness: float


@dataclass
class LatencyInfo:
    retrieval_latency_ms: int
    generation_latency_ms: int
    evaluation_latency_ms: int
    total_response_time_ms: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class Domain:
    id: str
    name: str
    question_count: int


@dataclass
class RAGResult:
    id: str
    domain_id: str
    domain_name: str
    question: str
    answer: str
    ground_truth: str
    retrieved_documents: list[RetrievedDocument] = field(default_factory=list)
    pipeline_config: PipelineConfig | None = None
    evaluation_metrics: EvaluationMetrics | None = None
    latency: LatencyInfo | None = None

    @staticmethod
    def from_dict(data: dict) -> "RAGResult":
        return RAGResult(
            id=data["id"],
            domain_id=data["domain_id"],
            domain_name=data["domain_name"],
            question=data["question"],
            answer=data["answer"],
            ground_truth=data["ground_truth"],
            retrieved_documents=[
                RetrievedDocument(**doc) for doc in data.get("retrieved_documents", [])
            ],
            pipeline_config=PipelineConfig(**data["pipeline_config"])
            if data.get("pipeline_config")
            else None,
            evaluation_metrics=EvaluationMetrics(**data["evaluation_metrics"])
            if data.get("evaluation_metrics")
            else None,
            latency=LatencyInfo(**data["latency"]) if data.get("latency") else None,
        )
