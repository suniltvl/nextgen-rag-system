# Modular Google Colab — Saurav RAG + SQLite

Run Saurav's RAG workflow on **Google Colab** with persistent assets on Google Drive.

## Drive layout

Upload / clone the repo under your existing project:

```
MyDrive/capstone-rag-kag/
  .env                              # OPENAI_API_KEY (copy from local)
  data/raw/ragbench/                # already downloaded
    covidqa/
    finqa/
    ...
  models/
    bge-small-en-v1.5/              # local embedder (reused, no HF download)
  repo/                             # this git branch cloned here
    helper.py
    sqldb/
    colab/
  runtime/                          # created by notebooks (persistent)
    database/biomedical_72_18/      # Chroma indexes
    sqldb/covidqa.db                # SQLite results
    experiments/runs/covidqa_tuned__covidqa/
      summary.json
      results.csv
      trace_metrics.png
```

## One-time setup

1. **Create branch locally** (already done): `shiva/colab-modular`
2. **Clone to Drive:**
   ```bash
   cd /path/to/MyDrive/capstone-rag-kag
   git clone <your-repo-url> repo
   cd repo && git checkout shiva/colab-modular
   ```
3. **Copy `.env`** to `MyDrive/capstone-rag-kag/.env`:
   ```
   OPENAI_API_KEY=sk-...
   ```
4. **Open Colab** and upload notebooks from `colab/notebooks/` (or open from Drive after sync).

## Run order

| Notebook | Purpose |
|----------|---------|
| `00_setup_and_config.ipynb` | Mount Drive, install deps, validate paths |
| `01_vectorization.ipynb` | Build/load Chroma index on Drive |
| `02_run_experiment.ipynb` | RAG + TRACe judge → SQLite |
| `03_analyze_results.ipynb` | Tables, CSV export, bar chart |

Run **00 → 01 → 02 → 03** in order.

## Quick smoke test (2 samples)

In `02_run_experiment.ipynb`, set:

```python
OVERRIDES = {
    "samples": 2,
    "max_workers": 1,
    "rebuild_db": True,   # first time only
    "resume": False,
}
```

## Switch experiment preset / dataset

Edit the config cell in any notebook:

```python
PRESET = "covidqa_tuned"   # saurav | baseline
DOMAIN = "finance"         # uses finqa parquet
DATASET = None             # or override: "cuad"
```

Or edit [`colab/config/experiments.yaml`](config/experiments.yaml).

## Persistence behavior

| Asset | Behavior |
|-------|----------|
| SQLite | `runtime/sqldb/{dataset}.db` — **never deleted** unless `rebuild_db=True` |
| Chroma | Reused if folder exists; set `REBUILD_INDEX=True` to rebuild |
| Embedder | Loaded from Drive with `local_files_only=True` |
| Dataset | Read from Drive parquet; HF fallback only if file missing |
| CSV + chart | Written to `runtime/experiments/runs/{preset}__{dataset}/` |

## Resume after disconnect

```python
OVERRIDES = {
    "samples": 50,
    "resume": True,
    "rebuild_db": False,
    "rebuild_index": False,
}
```

Already-evaluated sample ids in `nextgenrag_v1` are skipped.

## Secrets

Priority:
1. `{DRIVE_ROOT}/.env` via `python-dotenv`
2. Colab Secrets → `OPENAI_API_KEY`

## Future extensions

- **Reranker:** set `reranker:` in `experiments.yaml` (hook in `vectorization.py` later)
- **New domain:** add entry under `domains:` in YAML
- **New embed model:** download to `models/` and set `embed_model_path`

## Local parity check

From repo root (offline, local paths):

```bash
uv run python smoke_saurav_covidqa.py --model-type openai --samples 2
uv run python sqldb/view_results.py --compare --limit 5
```

Colab uses the same `helper.py`, `sqldb/schema.sql`, and TRACe judge flow.
