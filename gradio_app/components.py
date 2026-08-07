"""Reusable HTML snippet builders for the RAG dashboard.

Gradio Blocks doesn't offer rich progress-bar / badge / card widgets out of
the box, so these functions render small, self-contained HTML fragments
(styled by `styles.css`) that get poured into `gr.HTML` components. Keeping
the markup generation here — separate from `ui.py` (layout) and
`callbacks.py` (event wiring) — is what makes the UI layer "dumb": it only
ever asks "what HTML represents this data", never "where does this data
come from".
"""

from __future__ import annotations

from models import EvaluationMetrics, LatencyInfo, PipelineConfig, RetrievedDocument


def _score_color(value: float) -> str:
    """Green / amber / red thresholds shared by every progress bar in the app."""
    if value >= 0.95:
        return "var(--metric-good)"
    if value >= 0.85:
        return "var(--metric-warn)"
    return "var(--metric-bad)"


def progress_bar_html(value: float) -> str:
    pct = round(value * 100, 1)
    color = _score_color(value)
    return f"""
    <div class="progress-track">
      <div class="progress-fill" style="width:{pct}%; background:{color};"></div>
    </div>
    """


def badge_html(icon: str, label: str, value: str) -> str:
    return f"""
    <div class="badge">
      <span class="badge-icon">{icon}</span>
      <span class="badge-label">{label}</span>
      <span class="badge-value">{value}</span>
    </div>
    """


def header_badges_html(total_questions: int, domain: str, gen_model: str, embed_model: str) -> str:
    return f"""
    <div class="badge-row">
      {badge_html("📚", "Total Questions", str(total_questions))}
      {badge_html("🗂️", "Current Domain", domain)}
      {badge_html("🤖", "Generator Model", gen_model)}
      {badge_html("🧬", "Embedding Model", embed_model)}
    </div>
    """


def metric_card_html(label: str, value: float) -> str:
    pct = round(value * 100, 1)
    color = _score_color(value)
    return f"""
    <div class="metric-card">
      <div class="metric-card-top">
        <span class="metric-label">{label}</span>
        <span class="metric-dot" style="background:{color};"></span>
      </div>
      <div class="metric-value">{pct}%</div>
      {progress_bar_html(value)}
    </div>
    """


def metrics_grid_html(metrics: EvaluationMetrics | None) -> str:
    if metrics is None:
        return "<div class='empty-state'>Run a query to see evaluation metrics.</div>"
    cards = [
        metric_card_html("Overall Score", metrics.overall_score),
        metric_card_html("Faithfulness", metrics.faithfulness),
        metric_card_html("Context Relevance", metrics.context_relevance),
        metric_card_html("Context Utilization", metrics.context_utilization),
        metric_card_html("Answer Completeness", metrics.answer_completeness),
    ]
    return f"<div class='metrics-grid'>{''.join(cards)}</div>"


def doc_summary_html(doc: RetrievedDocument) -> str:
    pct = round(doc.similarity_score * 100, 1)
    return f"""
    <div class="doc-meta">
      <div class="doc-meta-row">
        <span class="doc-chip">Rank #{doc.rank}</span>
        <span class="doc-chip">{doc.document_name}</span>
        <span class="doc-chip">Page {doc.page_number}</span>
        <span class="doc-chip">Chunk {doc.chunk_id}</span>
        <span class="doc-chip">{doc.chunk_length} chars</span>
      </div>
      <div class="doc-score-row">
        <span class="doc-score-label">Similarity Score: {pct}%</span>
        {progress_bar_html(doc.similarity_score)}
      </div>
    </div>
    """


def doc_accordion_label(doc: RetrievedDocument | None, slot: int) -> str:
    if doc is None:
        return f"Document {slot}"
    pct = round(doc.similarity_score * 100, 1)
    return f"#{doc.rank}  •  {doc.document_name}  •  {pct}% match"


def pipeline_config_html(config: PipelineConfig | None) -> str:
    if config is None:
        return "<div class='empty-state'>Select a question to load pipeline configuration.</div>"
    rows = [
        ("Chunk Strategy", config.chunk_strategy),
        ("Chunk Size", str(config.chunk_size)),
        ("Chunk Overlap", str(config.chunk_overlap)),
        ("Embedding Model", config.embedding_model),
        ("Embedding Dimension", str(config.embedding_dimension)),
        ("Vector Database", config.vector_database),
        ("Retrieval Type", config.retrieval_type),
        ("Top-K", str(config.top_k)),
        ("Generator LLM", config.generator_llm),
        ("Evaluation LLM", config.evaluation_llm),
    ]
    row_html = "".join(
        f"<div class='kv-row'><span class='kv-key'>{k}</span><span class='kv-value'>{v}</span></div>"
        for k, v in rows
    )
    return f"<div class='kv-list'>{row_html}</div>"


def pipeline_summary_html(config: PipelineConfig | None) -> str:
    if config is None:
        return "<div class='empty-state'>Run a query to see the pipeline summary.</div>"
    rows = [
        ("Chunk Strategy", config.chunk_strategy),
        ("Embedding Model", config.embedding_model),
        ("Vector DB", config.vector_database),
        ("Retriever", f"{config.retrieval_type} (top-{config.top_k})"),
        ("Generator", config.generator_llm),
        ("Judge LLM", config.evaluation_llm),
    ]
    row_html = "".join(
        f"<div class='kv-row'><span class='kv-key'>{k}</span><span class='kv-value'>{v}</span></div>"
        for k, v in rows
    )
    return f"<div class='kv-list summary'>{row_html}</div>"


def performance_html(latency: LatencyInfo | None) -> str:
    if latency is None:
        return "<div class='empty-state'>Run a query to see latency and token usage.</div>"
    rows = [
        ("Retrieval Latency", f"{latency.retrieval_latency_ms} ms"),
        ("Generation Latency", f"{latency.generation_latency_ms} ms"),
        ("Evaluation Latency", f"{latency.evaluation_latency_ms} ms"),
        ("Total Response Time", f"{latency.total_response_time_ms} ms"),
        ("Prompt Tokens", f"{latency.prompt_tokens:,}"),
        ("Completion Tokens", f"{latency.completion_tokens:,}"),
        ("Total Tokens", f"{latency.total_tokens:,}"),
    ]
    row_html = "".join(
        f"<div class='kv-row'><span class='kv-key'>{k}</span><span class='kv-value'>{v}</span></div>"
        for k, v in rows
    )
    return f"<div class='kv-list'>{row_html}</div>"


def ground_truth_html(ground_truth: str | None) -> str:
    if not ground_truth:
        return "<div class='empty-state'>Run a query to see the ground truth answer.</div>"
    return f"""
    <div class="gt-card">
      <div class="gt-title">✅ Ground Truth Answer</div>
      <div class="gt-body">{ground_truth}</div>
    </div>
    """


def question_card_html(question: str | None) -> str:
    if not question:
        return "<div class='empty-state'>Select a domain and question, then click Go.</div>"
    return f"""
    <div class="question-card">
      <div class="question-eyebrow">Question</div>
      <div class="question-text">{question}</div>
    </div>
    """
