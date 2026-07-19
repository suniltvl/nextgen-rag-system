"""Batch RAG + TRACe eval with persistent SQLite on Drive."""

from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from helper import RAGHelper
from sqldb.init_db import init_db

from colab.lib.data_loader import load_parquet_rows
from colab.lib.experiment_config import ExperimentRun
from colab.lib.vectorization import build_retriever

EVAL_TABLE = "nextgenrag_v1"
_print_lock = Lock()


def _mean(values: list) -> float | None:
    nums = [v for v in values if v is not None]
    return sum(nums) / len(nums) if nums else None


def load_sample_rows(db_path: Path, *, skip_ids: set[int] | None = None) -> list[dict]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT s.id, s.query AS question, q.gpt_relevance AS relevance_score,
                   q.gpt_utilization AS utilization_score,
                   q.gpt_completeness AS completeness_score,
                   q.gpt_adherence AS adherence_score
            FROM nextgenrag_sample_questions s
            LEFT JOIN nextgenrag_questions q ON s.query = q.query
            ORDER BY s.id
            """
        ).fetchall()
        return [dict(r) for r in rows if skip_ids is None or r["id"] not in skip_ids]
    finally:
        conn.close()


def completed_sample_ids(db_path: Path) -> set[int]:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("SELECT DISTINCT id FROM nextgenrag_v1").fetchall()
        return {r[0] for r in rows}
    finally:
        conn.close()


def process_sample(
    row: dict,
    *,
    retriever,
    exp: ExperimentRun,
    session_id: str,
) -> dict:
    rg = RAGHelper(api_key=os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY"))
    qid = row["id"]
    query = row["question"]
    last_err: Exception | None = None

    for attempt in range(1, exp.llm_retries + 1):
        try:
            query, answer, sent_list = rg.simple_rag(
                query=query,
                expert_domain=exp.domain,
                retriever=retriever,
                gen_model=exp.gen_model,
                model_type=exp.model_type,
            )
            eval_response, relevance, utilization, completeness, adherence = (
                rg.evaluate_and_score_with_retry(
                    query=query,
                    response=answer,
                    sent_list=sent_list,
                    eval_model=exp.eval_model,
                    model_type=exp.model_type,
                    max_retries=exp.judge_retries,
                )
            )
            rg.db_insert(
                chunk_size=exp.chunk_size,
                chunk_overlap=exp.chunk_overlap,
                table_name=EVAL_TABLE,
                database_url=str(exp.sqlite_db),
                qid=qid,
                vector_db="chroma",
                session_id=session_id,
                retrieval_type=exp.retrieval_type,
            )
            return {
                "id": qid,
                "question": query,
                "reference_scores": {
                    "relevance": row.get("relevance_score"),
                    "utilization": row.get("utilization_score"),
                    "completeness": row.get("completeness_score"),
                    "adherence": row.get("adherence_score"),
                },
                "our_scores": {
                    "relevance": relevance,
                    "utilization": utilization,
                    "completeness": completeness,
                    "adherence": adherence,
                },
                "parse_error": rg.parse_error,
                "llm_error": False,
                "answer_preview": answer[:200],
                "judge_preview": eval_response[:200],
            }
        except Exception as err:
            last_err = err
            print(f"Sample {qid} attempt {attempt}/{exp.llm_retries} failed: {err}")
            if attempt < exp.llm_retries:
                time.sleep(2 ** attempt)

    return {
        "id": qid,
        "question": query,
        "reference_scores": {
            "relevance": row.get("relevance_score"),
            "utilization": row.get("utilization_score"),
            "completeness": row.get("completeness_score"),
            "adherence": row.get("adherence_score"),
        },
        "our_scores": {
            "relevance": None,
            "utilization": None,
            "completeness": None,
            "adherence": None,
        },
        "parse_error": True,
        "llm_error": True,
        "error": str(last_err),
        "answer_preview": "",
        "judge_preview": "",
    }


def build_index(exp: ExperimentRun):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    index_rows = load_parquet_rows(exp.parquet_file, limit=exp.index_limit)
    print(f"Indexing {len(index_rows)} rows from {exp.parquet_file}")
    return build_retriever(
        index_rows,
        embedder_path=exp.embedder_dir,
        persist_directory=exp.chroma_dir,
        chunk_size=exp.chunk_size,
        chunk_overlap=exp.chunk_overlap,
        search_type=exp.search_type,
        search_kwargs=exp.search_kwargs,
        rebuild=exp.rebuild_index,
    )


def init_sqlite(exp: ExperimentRun) -> None:
    init_db(
        exp.sqlite_db,
        exp.parquet_file,
        rebuild=exp.rebuild_db,
        sample_count=exp.samples,
        sample_offset=exp.sample_offset,
        domain_name=exp.domain,
        dataset_name=exp.dataset,
    )


def run_experiment(exp: ExperimentRun, *, retriever=None) -> dict:
    if exp.model_type == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for model_type=openai")

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    init_sqlite(exp)
    retriever = retriever or build_index(exp)

    skip_ids = completed_sample_ids(exp.sqlite_db) if exp.resume else set()
    sample_rows = load_sample_rows(exp.sqlite_db, skip_ids=skip_ids)
    session_id = str(uuid.uuid4())
    print(
        f"Running {len(sample_rows)} samples | session={session_id} | "
        f"chunk={exp.chunk_size}/{exp.chunk_overlap} | workers={exp.max_workers} | "
        f"gen={exp.gen_model} | eval={exp.eval_model} | type={exp.model_type}"
    )
    if exp.resume and skip_ids:
        print(f"Resuming: skipping {len(skip_ids)} already-evaluated ids")

    results: list[dict] = []
    completed = 0
    with ThreadPoolExecutor(max_workers=exp.max_workers) as pool:
        futures = {
            pool.submit(
                process_sample,
                row,
                retriever=retriever,
                exp=exp,
                session_id=session_id,
            ): row["id"]
            for row in sample_rows
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1
            with _print_lock:
                print(
                    f"[{completed}/{len(sample_rows)}] id={result['id']} "
                    f"ours={result['our_scores']} err={result['parse_error']}"
                )

    results.sort(key=lambda r: r["id"])
    summary = {
        "session_id": session_id,
        "preset": exp.preset,
        "domain": exp.domain,
        "dataset": exp.dataset,
        "samples": len(results),
        "chunk_size": exp.chunk_size,
        "chunk_overlap": exp.chunk_overlap,
        "gen_model": exp.gen_model,
        "eval_model": exp.eval_model,
        "model_type": exp.model_type,
        "max_workers": exp.max_workers,
        "parse_errors": sum(1 for r in results if r["parse_error"]),
        "llm_errors": sum(1 for r in results if r.get("llm_error")),
        "avg_reference": {
            "relevance": _mean([r["reference_scores"]["relevance"] for r in results]),
            "utilization": _mean([r["reference_scores"]["utilization"] for r in results]),
            "completeness": _mean([r["reference_scores"]["completeness"] for r in results]),
            "adherence": _mean([r["reference_scores"]["adherence"] for r in results]),
        },
        "avg_ours": {
            "relevance": _mean([r["our_scores"]["relevance"] for r in results]),
            "utilization": _mean([r["our_scores"]["utilization"] for r in results]),
            "completeness": _mean([r["our_scores"]["completeness"] for r in results]),
            "adherence": _mean([r["our_scores"]["adherence"] for r in results]),
        },
        "results": results,
    }

    exp.summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = exp.summary_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print(f"\n{'=' * 80}")
    print("BATCH SUMMARY")
    print(f"session_id: {session_id}")
    print(f"avg reference: {summary['avg_reference']}")
    print(f"avg ours:      {summary['avg_ours']}")
    print(f"parse_errors:  {summary['parse_errors']}")
    print(f"summary JSON:  {summary_path}")
    print(f"sqlite DB:     {exp.sqlite_db}")
    return summary
