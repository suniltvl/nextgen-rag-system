"""Event handlers wiring UI interactions to the data layer.

Every function here is a pure translation: (UI inputs) -> (data_loader call)
-> (component updates). None of them know or care that `data_loader` is
currently JSON-backed — that's the whole point of the split described in
`data_loader.py`.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import tempfile
from pathlib import Path

import gradio as gr

import components as c
import data_loader as dl
from models import RAGResult

EMPTY_DOC = ("Document", "", "")


def on_domain_change(domain_name: str | None):
    questions = dl.load_questions_for_domain(domain_name) if domain_name else []
    question_update = gr.update(choices=questions, value=questions[0] if questions else None)

    total = dl.total_question_count()
    badges = c.header_badges_html(total, domain_name or "—", "—", "—")

    reset_question = c.question_card_html(None)
    reset_answer = ""
    reset_metrics = c.metrics_grid_html(None)
    reset_gt = c.ground_truth_html(None)
    reset_summary = c.pipeline_summary_html(None)
    reset_perf = c.performance_html(None)
    reset_config = c.pipeline_config_html(None)

    doc_updates: list = []
    for _ in range(4):
        doc_updates.append("<div class='empty-state'>No document loaded.</div>")
        doc_updates.append("")

    return (
        question_update,
        badges,
        reset_question,
        reset_answer,
        reset_metrics,
        *doc_updates,
        reset_gt,
        reset_summary,
        reset_perf,
        reset_config,
        None,
        None,
    )


def _empty_result_outputs(message: str):
    badges = c.header_badges_html(dl.total_question_count(), "—", "—", "—")
    doc_updates: list = []
    for _ in range(4):
        doc_updates.append(f"<div class='empty-state'>{message}</div>")
        doc_updates.append("")
    return (
        f"<div class='empty-state'>{message}</div>",
        "",
        c.metrics_grid_html(None),
        *doc_updates,
        c.ground_truth_html(None),
        c.pipeline_summary_html(None),
        c.performance_html(None),
        c.pipeline_config_html(None),
        badges,
        None,
        None,
    )


def on_go_click(domain_name: str | None, question_text: str | None):
    if not domain_name or not question_text:
        return _empty_result_outputs("Select a domain and question, then click Go.")

    result: RAGResult | None = dl.get_result_for_question(domain_name, question_text)
    if result is None:
        return _empty_result_outputs("No result found for this question.")

    question_html = c.question_card_html(result.question)
    metrics_html = c.metrics_grid_html(result.evaluation_metrics)
    ground_truth = c.ground_truth_html(result.ground_truth)
    pipeline_summary = c.pipeline_summary_html(result.pipeline_config)
    performance = c.performance_html(result.latency)
    pipeline_config = c.pipeline_config_html(result.pipeline_config)

    doc_updates: list = []
    docs = sorted(result.retrieved_documents, key=lambda d: d.rank)[:4]
    for i in range(4):
        if i < len(docs):
            doc_updates.append(c.doc_summary_html(docs[i]))
            doc_updates.append(docs[i].retrieved_text)
        else:
            doc_updates.append("<div class='empty-state'>No document in this slot.</div>")
            doc_updates.append("")

    gen_model = result.pipeline_config.generator_llm if result.pipeline_config else "—"
    embed_model = result.pipeline_config.embedding_model if result.pipeline_config else "—"
    badges = c.header_badges_html(dl.total_question_count(), result.domain_name, gen_model, embed_model)

    result_dict = dataclasses.asdict(result)

    return (
        question_html,
        result.answer,
        metrics_html,
        *doc_updates,
        ground_truth,
        pipeline_summary,
        performance,
        pipeline_config,
        badges,
        result_dict,
        result_dict,
    )


def make_download_file(result_dict: dict | None):
    if not result_dict:
        return gr.update(value=None)
    fd, path = tempfile.mkstemp(suffix=".json", prefix="rag_result_")
    with open(fd, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2)
    return gr.update(value=path)


def make_export_csv(result_dict: dict | None):
    if not result_dict:
        return gr.update(value=None)

    fd, path = tempfile.mkstemp(suffix=".csv", prefix="rag_export_")
    metrics = result_dict.get("evaluation_metrics") or {}
    latency = result_dict.get("latency") or {}
    config = result_dict.get("pipeline_config") or {}

    rows = [
        ("question", result_dict.get("question", "")),
        ("domain", result_dict.get("domain_name", "")),
        ("answer", result_dict.get("answer", "")),
        ("overall_score", metrics.get("overall_score", "")),
        ("faithfulness", metrics.get("faithfulness", "")),
        ("context_relevance", metrics.get("context_relevance", "")),
        ("context_utilization", metrics.get("context_utilization", "")),
        ("answer_completeness", metrics.get("answer_completeness", "")),
        ("generator_llm", config.get("generator_llm", "")),
        ("evaluation_llm", config.get("evaluation_llm", "")),
        ("total_response_time_ms", latency.get("total_response_time_ms", "")),
        ("total_tokens", latency.get("total_tokens", "")),
    ]

    with open(fd, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["field", "value"])
        writer.writerows(rows)

    return gr.update(value=path)
