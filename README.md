# RAG-KAG Capstone

Domain-specific RAG evaluation and optimization with a controlled KAG enhancement layer.

**Team:** Shiva · Sunil · Sourav · Vinay
**Supervisor:** Dr. Manish Shrivastava (IIIT-H)
**Mentors:** Gopichand · Lokesh

## What this is

An evaluation-driven RAG system across the five RAGBench domains (biomedical, general-knowledge, legal, customer-support, finance), with from-scratch implementations of TRACe and RGB metrics, plus a small KAG (Knowledge-Augmented Generation) prototype on relationship-heavy domains.

The mandatory deliverables are RAGBench end-to-end evaluation and RGB four-ability evaluation. KAG is an innovation extension — never a replacement.

## Quickstart

```bash
# 1. Install (uv recommended)
uv venv && source .venv/bin/activate
uv pip install -e ".[dev,ui]"

# Or with plain pip:
# python -m venv .venv && source .venv/bin/activate
# pip install -e ".[dev,ui]"

# 2. Configure secrets (only the providers you'll use)
cp .env.example .env
# edit .env

# 3. Smoke-test the V1 baseline on a small CovidQA sample
rag-kag run --config configs/v1_baseline.yaml --domain covidqa --limit 20

# 4. Inspect results
ls experiments/runs/
```

## Layout

```
configs/                  YAML experiment configs (V1..V8)
src/rag_kag/
  data_loaders/           RAGBench adapter
  chunkers/               sliding-window, sentence-aware, semantic
  embedders/              sentence-transformer wrappers
  vectorstores/           Chroma (FAISS optional)
  retrievers/             dense, BM25, hybrid, HyDE
  rerankers/              cross-encoder
  generators/             litellm-backed LLM call
  evaluators/             TRACe (RAGBench) + RGB metrics
  kag/                    triple extraction + graph retrieval (Phase 3)
  pipeline.py             orchestrator
  cli.py                  Typer CLI
experiments/              run logs and result matrices
tests/
```

## Experiment matrix (proposal §7)

| Version | Configuration                                                       | Purpose                          |
|---------|---------------------------------------------------------------------|----------------------------------|
| V1      | sliding chunks + bge-small + Chroma + dense + grounded prompt       | End-to-end baseline              |
| V2      | V1 + semantic / sentence-aware chunking                             | Measure chunking impact          |
| V3      | V2-best + improved embedding model                                  | Measure embedding impact         |
| V4      | V3-best + BM25 + hybrid retrieval                                   | Sparse + semantic                |
| V5      | V4 + HyDE / query rewriting                                         | Hard-query retrieval             |
| V6      | V5 + cross-encoder reranking + context repacking                    | Ordering and relevance           |
| V7      | Best RAG + KAG triples (finance/legal)                              | RAG vs KAG-enhanced              |
| V8      | RGB evaluation across 4–5 open-source LLMs                          | Generation behavior              |

## Hard rules (mentor brief)

- TRACe and RGB metrics implemented from formulas — **no RAGAS / TruLens**.
- Cover all five domains; one common pipeline with per-domain config swaps.
- Change one component at a time in experiments.
- Evaluation correctness is judged above demo polish.

## License

MIT (see `pyproject.toml`).
