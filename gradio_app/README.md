# Real World RAG System — Dashboard

A Gradio dashboard that demonstrates how a Retrieval-Augmented Generation
(RAG) pipeline works over the RAGBench dataset: pick a domain and question,
run the pipeline, and inspect the generated answer, retrieved chunks,
evaluation metrics, pipeline configuration, and latency breakdown.

This is a **demo/inspection dashboard**, not a chatbot.

## Project structure

```
gradio_app/
├── app.py                  # Entry point (python gradio_app/app.py)
├── ui.py                   # gr.Blocks layout — no business logic
├── callbacks.py            # Event handlers: UI inputs -> data_loader -> component updates
├── components.py           # Reusable HTML builders (badges, metric cards, progress bars, etc.)
├── data_loader.py          # Data access layer — currently reads JSON, swappable for an API client
├── models.py                # Typed dataclasses shared by every layer
├── generate_sample_data.py # One-off script that (re)generates the sample dataset
├── styles.css               # Dashboard styling (light/dark aware)
├── sample_data/
│   ├── domains.json
│   └── questions.json
└── README.md
```

## Running it

From the repository root, with the project's virtual environment active:

```bash
uv run python gradio_app/app.py
```

or, without `uv`:

```bash
python gradio_app/app.py
```

Then open the local URL Gradio prints (defaults to `http://127.0.0.1:7860`).

## Regenerating the sample dataset

```bash
python gradio_app/generate_sample_data.py
```

This produces 5 domains × 10 questions each (Biomedical Research, Customer
Support, Finance, General Knowledge, Legal), with a generated answer, ground
truth, top-4 retrieved chunks, pipeline configuration, evaluation metrics,
and latency/token counts per question.

## Swapping JSON for a real API later

Every data access the UI needs goes through four functions in
`data_loader.py`:

- `load_domains()` / `get_domain_names()`
- `load_questions_for_domain(domain_name)`
- `get_result_for_question(domain_name, question_text)`
- `total_question_count()`

To move to a live backend, replace the bodies of these functions with calls
to `POST /api/rag/query` (or whichever endpoints you stand up) that return
data shaped like `models.RAGResult`. Nothing in `ui.py`, `callbacks.py`, or
`components.py` needs to change, since they only ever depend on these
function signatures and the `models.py` dataclasses.
