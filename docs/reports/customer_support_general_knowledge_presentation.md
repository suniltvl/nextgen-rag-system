# Customer Support & General Knowledge Domain
# Real World RAG System using RAGBench

**PG Diploma Capstone Project**

**Domain**
- Customer Support
- General Knowledge

**Author**
Venkat

---

# Problem Statement

Large Language Models (LLMs) often produce:

- Hallucinated responses
- Outdated information
- Domain-independent answers
- Unsupported factual claims

Retrieval-Augmented Generation (RAG) mitigates these issues by retrieving relevant knowledge before generating responses.

---

# Project Objective

Build an enterprise-grade Retrieval-Augmented Generation system capable of answering Customer Support and General Knowledge questions with explainable evaluation.

Project Goals

- Accurate retrieval
- Grounded generation
- Explainable evaluation
- Modular architecture
- Easy experimentation

---

# RAGBench Dataset

Dataset

RAGBench

Five Domains

- Biomedical
- Finance
- Customer Support ✅
- General Knowledge ✅
- Legal

Dataset Characteristics

- Real-world documents
- Human-created questions
- Ground truth answers
- Explainable evaluation
- Enterprise use cases

Total Evaluation Questions

**48 Questions**

---

# Why Customer Support & General Knowledge?

## Customer Support

Enterprise customer support requires:

- Product manuals
- Installation guides
- Troubleshooting
- FAQs
- Configuration documentation

Challenges

- Long documents
- Similar instructions
- Procedural answers
- Multiple document retrieval

---

## General Knowledge

General Knowledge evaluates

- Open-domain QA
- Multi-hop reasoning
- Factual correctness
- Knowledge grounding

Challenges

- Broad topics
- Diverse document sources
- Retrieval precision
- Hallucination prevention

---

# Overall Architecture

```text
User Question
      │
      ▼
Query Processing
      │
      ▼
Chunk Retrieval
      │
      ▼
Prompt Builder
      │
      ▼
Gemma Generator
      │
      ▼
Generated Answer
      │
      ▼
Gemma Judge
      │
      ▼
Evaluation Metrics
```

---

# End-to-End RAG Pipeline

## Document Processing

- PDF Loader
- Recursive Chunking

↓

## Embedding

BAAI/bge-small-en-v1.5

↓

## Vector Database

ChromaDB

↓

## Retrieval

Similarity Search

↓

## Generation

Google Gemma

↓

## Evaluation

Google Gemma Judge

---

# Technology Stack

| Component | Technology |
|------------|------------|
| Language | Python |
| Framework | LangChain |
| API | FastAPI |
| UI | Gradio |
| Embeddings | BAAI/bge-small-en-v1.5 |
| Vector DB | ChromaDB |
| Database | PostgreSQL |
| Generator | Google Gemma |
| Judge | Google Gemma |

---

# Experiment Configurations

The framework supports multiple configurations.

Configurable Parameters

- Chunk Size
- Chunk Overlap
- Embedding Model
- Vector Database
- Retrieval Strategy
- Generator Model
- Judge Model

Experiments were logged into PostgreSQL for comparison.

---

# Evaluation Framework

RAGBench evaluates responses using explainable metrics.

## Context Relevance

Measures whether retrieved chunks answer the question.

---

## Context Utilization

Measures whether retrieved context is actually used.

---

## Answer Faithfulness

Measures grounding of generated answer.

---

## Answer Completeness

Measures whether all aspects of the answer are covered.

---

# Customer Support Evaluation

Evaluation performed on

- RAGBench Customer Support
- 48 Questions

Captured Metrics

- Context Relevance
- Context Utilization
- Faithfulness
- Completeness

Evaluation performed using Google Gemma Judge.

---

# General Knowledge Evaluation

Evaluation performed on

- RAGBench General Knowledge

Captured Metrics

- Context Relevance
- Context Utilization
- Faithfulness
- Completeness

---

# Evaluation Database

All experiments were stored inside PostgreSQL.

Captured Information

- Session ID
- Chunk Size
- Chunk Overlap
- Embedding Model
- Vector DB
- Retrieval Strategy
- Generator Model
- Judge Model
- Generated Answer
- Ground Truth
- Individual Metric Scores

This enables reproducible experimentation and model comparison.

---

# Comparative Experiments

Different configurations can be compared using:

- Chunk Sizes
- Embedding Models
- Generator Models
- Judge Models
- Retrieval Strategies

Each experiment is reproducible through configuration-driven execution.

---

# Sample Workflow

Question

↓

Relevant Documents Retrieved

↓

Prompt Construction

↓

Gemma Response

↓

Judge Evaluation

↓

Scores Stored in PostgreSQL

↓

Displayed in Gradio Dashboard

---

# Explainable Evaluation

Each answer contains

- Generated Response
- Retrieved Documents
- Context Relevance Score
- Context Utilization Score
- Faithfulness Score
- Completeness Score

This provides transparency instead of a single accuracy number.

---

# Current Strengths

✔ Modular Architecture

✔ Configuration-driven Pipeline

✔ Explainable Evaluation

✔ PostgreSQL Experiment Tracking

✔ Gradio Visualization

✔ FastAPI Backend

✔ Enterprise-ready Design

---

# Current Challenges

Observed during experimentation

- Retrieval quality strongly affects generation
- Chunk boundaries influence answer completeness
- Similar chunks reduce retrieval precision
- General Knowledge requires broader retrieval
- Customer Support requires procedural grounding

---

# Future Improvements

Planned Enhancements

- Hybrid Search (BM25 + Dense)
- Cross Encoder Re-ranking
- HyDE Retrieval
- Parent-Child Chunking
- Multi-query Retrieval
- Metadata Filtering
- Milvus / Qdrant Comparison
- Agentic RAG
- Production Deployment

---

# Conclusion

Successfully developed a production-style Retrieval-Augmented Generation system for Customer Support and General Knowledge domains.

Key achievements

- End-to-End RAG Pipeline
- Explainable Evaluation
- Configuration-based Experiments
- PostgreSQL Tracking
- FastAPI APIs
- Gradio Dashboard
- Modular & Extensible Architecture

The project provides a strong foundation for future enterprise RAG applications.

---

# Thank You

Questions?