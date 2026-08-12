# NextGen RAG System

**IIIT Hyderabad — AIML26 Capstone (August 2026)**

Configuration-driven Retrieval-Augmented Generation (RAG) pipeline evaluated on **RAGBench** across Customer Support, General Knowledge, Legal (CUAD), and Finance (FinQA / TAT-QA), plus diagnostic **RGB** stress tests of the generator. The Legal track further evolves the stack from vanilla RAG → **Knowledge-Augmented Generation (KAG / GraphRAG)** → **Agentic KAG**, closing the gap to the RAGBench Claude Haiku reference on CUAD.

| | |
|---|---|
| **Supervisor** | Dr. Manish Shrivastava |
| **Mentors** | Gopichand · Lokesh |
| **Team** | Sunil · Shiva · Sourav · Vinay |
| **Repo** | [suniltvl/nextgen-rag-system](https://github.com/suniltvl/nextgen-rag-system) |

---

## Highlights

- Modular ingestion → chunk → embed → retrieve → generate → TRACe judge pipeline
- Domain ablations over **chunking**, **embeddings**, **retrieval** (dense / hybrid), and **generator–judge** pairs
- Results persisted to **Postgres** (CS/GK, Finance) and **SQLite** (Legal)
- Local **Gradio** inspection dashboard for answers, chunks, metrics, and config
- Legal CUAD track: **small2big** RAG → Neo4j ontology + `category_slot_rrf` fusion → deterministic Agentic KAG (**completeness 0.767** vs reference **0.770**)

### Headline results (pilot)

| Track | Best signal (this project) |
|---|---|
| Customer Support | 256/50 + `bge-large-en-v1.5` + `gemma-3-12b-it` / `gemma-4-26b` judge (high composite / adherence) |
| General Knowledge (`hotpotqa`) | Strong completeness in places; relevance remains low (multi-hop limit of flat dense top‑k) |
| Legal (CUAD) RAG | `bge-small-en-v1.5` + **small2big** — best relevance & completeness RMSE vs reference |
| Legal (CUAD) KAG / V8 | **0.767** completeness, **1.00** adherence, **~1.3 s** latency (deterministic router) |
| Finance | Hybrid + mid/large chunks; best adherence F1 **0.850** (GPT ref) / **0.889** (Claude ref) |
| RGB | Strong noise robustness; weak negative rejection across generators (accuracy vs refusal trade-off) |

> Full tables, RMSE/F1 validation, error analysis, and discussion: [`docs/reports/`](docs/reports/).

---

## Track ownership

| Domain / track | Owner |
|---|---|
| Customer Support & General Knowledge (RAGBench) | Sunil (Venkata Lakshmi Sunil Talluri) |
| Finance — FinQA / TAT-QA (RAGBench) | Sourav |
| Legal — CUAD (RAGBench → KAG → Agentic KAG) | Shiva |
| RGB (Noise Robustness, Negative Rejection, Information Integration) | Project team |

---

## Architecture (online path)

```text
Question bank (Postgres / SQLite)
        │
        ▼
Retriever (Chroma dense / hybrid; Legal: + Neo4j Cypher / category slot)
        │
        ▼
Prompt (domain system prompt + context)
        │
        ▼
Generator LLM ──► Judge LLM (TRACe JSON rubric)
        │                    │
        └──────────┬─────────┘
                   ▼
         Metrics + config row persisted
                   │
                   ▼
         Gradio inspection dashboard
```

**Offline:** HuggingFace RAGBench load → document/question dedupe → chunk → embed → Chroma collections (and Neo4j graph build for Legal KAG).

---

## Repository layout

```text
nextgen-rag-system/
├── helper.py                 # RAGHelper — generate, judge, TRACe parse, DB write
├── main.py                   # Small RAGBench loader smoke entrypoint
├── pyproject.toml            # uv / Python ≥ 3.12 dependencies
├── src/
│   ├── Vectorization.ipynb   # Ingest, chunk, embed → Chroma
│   ├── questionDB.ipynb      # Question bank + RAGBench reference scores
│   ├── Generation.ipynb      # Retrieve → generate → evaluate → persist
│   ├── rag_components/       # Chunkers, embeddings, retrievers, generators, stores
│   ├── utils/                # Shared helpers
│   └── ...
├── gradio_app/               # Local demo / inspection UI
├── docs/reports/             # Capstone report, track write-ups, presentations
├── database_chroma/          # Persisted Chroma collections (local)
└── sqldb / db /              # SQLite experiment DBs (Legal and related)
```

---

## Datasets & benchmarks

| Resource | Role |
|---|---|
| [RAGBench](https://huggingface.co/datasets/rungalileo/ragbench) (`rungalileo/ragbench`, project mirror `suniltvl/ragbench`) | End-to-end RAG evaluation with TRACe labels |
| Domains used | Customer Support (`delucionqa`, `emanual`, `techqa`), GK (`hotpotqa`), Legal (`cuad`), Finance (`finqa`, `tatqa`) — **test** split |
| [RGB](https://arxiv.org/abs/2309.01431) | Generator diagnostics: noise robustness, negative rejection, information integration |

### TRACe metrics

| Metric | What it measures |
|---|---|
| **Relevance** | Retrieved context vs question |
| **Utilization** | How much retrieved context the answer uses |
| **Completeness** | Coverage of relevant content in the answer |
| **Adherence** | Faithfulness — claims grounded in retrieved context |

Continuous scores are also validated against RAGBench GPT/Claude references via **RMSE**; binary adherence via **F1**.

---

## Tech stack

| Layer | Choice |
|---|---|
| Orchestration | LangChain |
| Vector store | ChromaDB |
| Embeddings | `BAAI/bge-large-en-v1.5`, `BAAI/LLM-Embedder`, `bge-small-en-v1.5`, `bge-m3`, `bge-base-en-v1.5`, … |
| Generators / judges | Groq, OpenRouter, OpenAI-compatible, Ollama (track-specific) |
| Legal graph (KAG) | Neo4j (property graph + vector + fulltext indexes) |
| Persistence | PostgreSQL (`psycopg`), SQLite |
| UI | Gradio |
| Env / packaging | `uv`, Python ≥ 3.12 |

---

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/suniltvl/nextgen-rag-system.git
cd nextgen-rag-system
uv sync
```

### 2. Environment

Create a `.env` in the repo root (do not commit secrets). Typical keys:

```env
GROQ_API_KEY=
OPEN_ROUTER_API_KEY=
DATABASE_URL=                 # or LOCAL_POSTGRE_DATABASE_URL
# Optional: GEMMA_API_KEY, NVIDIA_MODEL_API_KEY, Neo4j URI/auth for Legal KAG
```

### 3. Core notebooks (CS / GK style flow)

From the project root, with the venv active:

1. **`src/Vectorization.ipynb`** — load & dedupe RAGBench docs, chunk, embed, persist Chroma  
2. **`src/questionDB.ipynb`** — build question bank + attach RAGBench reference TRACe scores  
3. **`src/Generation.ipynb`** — retrieval × generation × judge ablation; write evaluation rows  

Shared generation/evaluation logic lives in [`helper.py`](helper.py) (`RAGHelper`).

### 4. Gradio dashboard

```bash
uv run python gradio_app/app.py
```

See [`gradio_app/README.md`](gradio_app/README.md) for UI architecture (demo/inspection, not a chatbot).

### 5. Legal KAG / Agentic KAG (CUAD)

Legal experiments extend the RAG baseline with:

1. **V1–V6** — chunking/embedding ablation → best **small2big** + `bge-small-en-v1.5` (completeness ≈ **0.621**)  
2. **V7** — Neo4j CUAD ontology; breakthrough is **`category_slot_rrf`** fusion (completeness **0.767**)  
3. **V8** — deterministic category router (same accuracy, **~1.3 s** vs ReAct **13–27 s**)

CLI patterns (when the `rag-kag` package/configs are available in-tree):

```bash
rag-kag kag init-db
rag-kag kag build -c configs/legal/kag/cuad_kag_v7c.yaml --limit 50
rag-kag run -c configs/legal/kag/cuad_agentic_kag_v7c.yaml --limit 50
```

Details: Capstone Legal section + RAG → KAG handbook under [`docs/reports/`](docs/reports/).

---

## Key findings (short)

1. **Generator / judge choice** dominates the faithfulness vs coverage trade-off more than retrieval knobs.  
2. **Chunking strategy** moves relevance / utilization / completeness more than swapping embedders alone.  
3. **Judge reliability varies** — always validate automated TRACe scores against RAGBench reference labels (RMSE / F1).  
4. On CUAD, **fusion design** (`category_slot_rrf`) mattered more than “having a graph”; **grounded Yes/No prompts** fixed no-evidence cases retrieval cannot solve.  
5. RGB shows **noise robustness can coexist with poor refusal** — hallucination risk is partly generator-intrinsic.

---

## Documentation

| Document | Description |
|---|---|
| [`docs/reports/Capstone_Final_Report_Cleaned.md`](docs/reports/Capstone_Final_Report_Cleaned.md) | Full Capstone report (RAGBench + RGB) |
| [`docs/reports/RAG_KAG_Handbook.md`](docs/reports/RAG_KAG_Handbook.md) | Legal CUAD technical handbook — RAG → KAG → Agentic KAG |
| [`docs/reports/customer_support_general_knowledge_report.md`](docs/reports/customer_support_general_knowledge_report.md) | CS & GK track report |
| Presentations under `docs/reports/` | Capstone and track slide decks |

---

## References

- Friel et al. (2024). *RAGBench*. [arXiv:2407.11005](https://arxiv.org/abs/2407.11005) · [HF dataset](https://huggingface.co/datasets/rungalileo/ragbench)  
- Chen et al. (2023). *RGB Benchmark*. [arXiv:2309.01431](https://arxiv.org/abs/2309.01431)  
- Hendrycks et al. (2021). *CUAD*. [arXiv:2103.06268](https://arxiv.org/abs/2103.06268)  
- Wang et al. (2024). *Searching for Best Practices in RAG* (EMNLP). [arXiv:2407.01219](https://arxiv.org/abs/2407.01219)  
- Sarthi et al. (2024). *RAPTOR* (ICLR). [arXiv:2401.18059](https://arxiv.org/abs/2401.18059)  
- Neeser et al. (2025). *QuOTE*. [arXiv:2502.10976](https://arxiv.org/abs/2502.10976)

---

## License / academic use

Capstone coursework for **IIIT Hyderabad AIML26**. Use and redistribution of third-party datasets (RAGBench, RGB, CUAD, etc.) remain subject to their original licenses and terms.
