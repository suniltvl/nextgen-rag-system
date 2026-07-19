# SQLite schema (Saurav Postgres parity)

Local database for covidqa batch runs without Postgres.

**Default path:** `sqldb/saurav_covidqa.db`

## Tables

| Table | Purpose |
|-------|---------|
| `nextgenrag_questions` | RAGBench reference scores (GPT/Claude) per deduped question |
| `nextgenrag_sample_questions` | Work queue (`id`, `query`) |
| `nextgenrag_v1` | RAG run results (config + TRACe metrics) |

## Initialize

```bash
uv run python sqldb/init_db.py --rebuild --samples 50
```

## View results

```bash
uv run python sqldb/view_results.py --db sqldb/saurav_covidqa.db --counts
uv run python sqldb/view_results.py --db sqldb/saurav_covidqa.db --session latest
uv run python sqldb/view_results.py --db sqldb/saurav_covidqa.db --compare --limit 10
uv run python sqldb/view_results.py --db sqldb/saurav_covidqa.db --export-csv experiments/runs/saurav_covidqa_50/results.csv
```

## sqlite3 snippets

```bash
sqlite3 sqldb/saurav_covidqa.db ".tables"

sqlite3 sqldb/saurav_covidqa.db "SELECT COUNT(*) FROM nextgenrag_v1;"

sqlite3 sqldb/saurav_covidqa.db "
  SELECT v.id, substr(s.query,1,60), v.relevance, v.utilization, v.completeness, v.adherence
  FROM nextgenrag_v1 v
  JOIN nextgenrag_sample_questions s ON v.id = s.id
  ORDER BY v.created_at DESC LIMIT 10;"
```
