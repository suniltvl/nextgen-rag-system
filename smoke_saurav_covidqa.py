"""Batch covidqa runner for Saurav's helper.py workflow with SQLite parity.

Uses local parquet, local embedder, isolated Chroma, parallel LLM calls, and SQLite persistence.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import pandas as pd
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from helper import RAGHelper
from sqldb.init_db import init_db

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_PARQUET = REPO_ROOT / "data/raw/ragbench/covidqa/train-00000-of-00001.parquet"
DEFAULT_EMBEDDER = REPO_ROOT / "models/bge-small-en-v1.5"
CHROMA_DIR = REPO_ROOT / "database/saurav_covidqa_gemma4"
DEFAULT_SQLITE = REPO_ROOT / "sqldb/saurav_covidqa.db"
SUMMARY_DIR = REPO_ROOT / "experiments/runs/saurav_covidqa_50"
DEFAULT_GEN_MODEL = "gpt-4o-mini"
DEFAULT_EVAL_MODEL = "gpt-4o-mini"
DEFAULT_MODEL_TYPE = "openai"

SEPARATORS = ["\n\n", "\n", " ", ".", ","]
SEARCH_TYPE = "similarity"
SEARCH_KWARGS = {"k": 3}
DOMAIN = "biomedical"
EVAL_TABLE = "nextgenrag_v1"

CHUNK_PRESETS = {
    "covidqa_tuned": (72, 18),
    "saurav": (1024, 200),
    "baseline": (256, 32),
}

_print_lock = Lock()


def load_covidqa_rows(parquet_path: Path, limit: int) -> list[dict]:
    df = pd.read_parquet(parquet_path)
    if limit <= 0:
        rows = df.to_dict(orient="records")
    else:
        rows = df.head(limit).to_dict(orient="records")
    for row in rows:
        row["documents"] = list(row["documents"])
    return rows


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


def deduplicate_data(data: list[dict]) -> dict[str, dict]:
    data_dict: dict[str, dict] = {}
    for item in data:
        document = " ".join(item["documents"])
        if document in data_dict:
            data_dict[document]["docid"].append(item["id"])
        else:
            data_dict[document] = {"docid": [item["id"]]}
    return data_dict


def chroma_subdir(chunk_size: int, chunk_overlap: int) -> Path:
    return CHROMA_DIR / f"biomedical_{chunk_size}_{chunk_overlap}"


def build_retriever(
    rows: list[dict],
    *,
    embedder_path: Path,
    persist_directory: Path,
    chunk_size: int,
    chunk_overlap: int,
    rebuild: bool,
):
    if rebuild and persist_directory.exists():
        shutil.rmtree(persist_directory)
    persist_directory.parent.mkdir(parents=True, exist_ok=True)

    dedup = deduplicate_data(rows)
    docs = [
        Document(page_content=content, metadata=meta)
        for content, meta in dedup.items()
    ]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARATORS,
    )
    docs_chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name=str(embedder_path),
        model_kwargs={"local_files_only": True},
    )

    if persist_directory.exists() and any(persist_directory.iterdir()):
        print(f"Loading existing vector database at {persist_directory}...")
        vector_db = Chroma(
            persist_directory=str(persist_directory),
            embedding_function=embeddings,
        )
    else:
        print(f"Creating vector database at {persist_directory} ({len(docs_chunks)} chunks)...")
        vector_db = Chroma.from_documents(
            documents=docs_chunks,
            embedding=embeddings,
            persist_directory=str(persist_directory),
        )

    return vector_db.as_retriever(search_type=SEARCH_TYPE, search_kwargs=SEARCH_KWARGS)


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
    gen_model: str,
    eval_model: str,
    model_type: str,
    session_id: str,
    sqlite_db: Path,
    chunk_size: int,
    chunk_overlap: int,
    judge_retries: int,
    llm_retries: int,
) -> dict:
    rg = RAGHelper(api_key=os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY"))
    qid = row["id"]
    query = row["question"]

    last_err: Exception | None = None
    for attempt in range(1, llm_retries + 1):
        try:
            query, answer, sent_list = rg.simple_rag(
                query=query,
                expert_domain=DOMAIN,
                retriever=retriever,
                gen_model=gen_model,
                model_type=model_type,
            )
            eval_response, relevance, utilization, completeness, adherence = (
                rg.evaluate_and_score_with_retry(
                    query=query,
                    response=answer,
                    sent_list=sent_list,
                    eval_model=eval_model,
                    model_type=model_type,
                    max_retries=judge_retries,
                )
            )
            rg.db_insert(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                table_name=EVAL_TABLE,
                database_url=str(sqlite_db),
                qid=qid,
                vector_db="chroma",
                session_id=session_id,
                retrieval_type="dense",
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
            print(f"Sample {qid} attempt {attempt}/{llm_retries} failed: {err}")
            if attempt < llm_retries:
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


def _mean(values: list) -> float | None:
    nums = [v for v in values if v is not None]
    return sum(nums) / len(nums) if nums else None


def run_batch(
    *,
    parquet_path: Path,
    embedder_path: Path,
    sqlite_db: Path,
    persist_directory: Path,
    gen_model: str,
    eval_model: str,
    model_type: str,
    chunk_size: int,
    chunk_overlap: int,
    index_limit: int,
    sample_count: int,
    sample_offset: int,
    max_workers: int,
    judge_retries: int,
    llm_retries: int,
    rebuild_index: bool,
    rebuild_db: bool,
    resume: bool,
) -> dict:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    init_db(
        sqlite_db,
        parquet_path,
        rebuild=rebuild_db,
        sample_count=sample_count,
        sample_offset=sample_offset,
    )

    index_rows = load_covidqa_rows(parquet_path, index_limit)
    print(f"Indexing {len(index_rows)} covidqa rows from {parquet_path}")

    retriever = build_retriever(
        index_rows,
        embedder_path=embedder_path,
        persist_directory=persist_directory,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        rebuild=rebuild_index,
    )

    skip_ids = completed_sample_ids(sqlite_db) if resume else set()
    sample_rows = load_sample_rows(sqlite_db, skip_ids=skip_ids)
    session_id = str(uuid.uuid4())
    print(
        f"Running {len(sample_rows)} samples | session={session_id} | "
        f"chunk={chunk_size}/{chunk_overlap} | workers={max_workers} | "
        f"gen={gen_model} | eval={eval_model} | type={model_type}"
    )
    if resume and skip_ids:
        print(f"Resuming: skipping {len(skip_ids)} already-evaluated ids")

    results: list[dict] = []
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                process_sample,
                row,
                retriever=retriever,
                gen_model=gen_model,
                eval_model=eval_model,
                model_type=model_type,
                session_id=session_id,
                sqlite_db=sqlite_db,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                judge_retries=judge_retries,
                llm_retries=llm_retries,
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
        "samples": len(results),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "gen_model": gen_model,
        "eval_model": eval_model,
        "model_type": model_type,
        "max_workers": max_workers,
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

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = SUMMARY_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print(f"\n{'=' * 80}")
    print("BATCH SUMMARY")
    print(f"session_id: {session_id}")
    print(f"avg reference: {summary['avg_reference']}")
    print(f"avg ours:      {summary['avg_ours']}")
    print(f"parse_errors:  {summary['parse_errors']}")
    print(f"summary JSON:  {summary_path}")
    print(f"sqlite DB:     {sqlite_db}")
    return summary


def resolve_chunk_args(args: argparse.Namespace) -> tuple[int, int]:
    if args.preset == "custom":
        if args.chunk_size is None or args.chunk_overlap is None:
            raise SystemExit("--preset custom requires --chunk-size and --chunk-overlap")
        return args.chunk_size, args.chunk_overlap
    if args.chunk_size is not None or args.chunk_overlap is not None:
        return (
            args.chunk_size or CHUNK_PRESETS[args.preset][0],
            args.chunk_overlap or CHUNK_PRESETS[args.preset][1],
        )
    return CHUNK_PRESETS[args.preset]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--embedder", type=Path, default=DEFAULT_EMBEDDER)
    parser.add_argument("--sqlite-db", type=Path, default=DEFAULT_SQLITE)
    parser.add_argument(
        "--model-type",
        choices=["openai", "ollama", "groq"],
        default=DEFAULT_MODEL_TYPE,
        help="LLM backend for generation and judge",
    )
    parser.add_argument(
        "--gen-model",
        default=None,
        help="Generation model (default: gpt-4o-mini for openai, gemma4:e4b for ollama)",
    )
    parser.add_argument(
        "--eval-model",
        default=None,
        help="Judge model (defaults to --gen-model)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Shorthand: sets both --gen-model and --eval-model",
    )
    parser.add_argument("--preset", choices=[*CHUNK_PRESETS, "custom"], default="covidqa_tuned")
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--chunk-overlap", type=int, default=None)
    parser.add_argument("--index-limit", type=int, default=0, help="0 = all train rows")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--judge-retries", type=int, default=3)
    parser.add_argument("--llm-retries", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--rebuild-db", action="store_true")
    args = parser.parse_args()

    if not args.parquet.exists():
        raise SystemExit(f"Missing parquet: {args.parquet}")
    if not args.embedder.exists():
        raise SystemExit(f"Missing embedder: {args.embedder}")

    if args.model_type == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required for --model-type openai")

    default_model = "gemma4:e4b" if args.model_type == "ollama" else "llama-3.1-8b-instant"
    if args.model_type == "openai":
        default_model = DEFAULT_GEN_MODEL
    gen_model = args.gen_model or args.model or default_model
    eval_model = args.eval_model or args.model or gen_model

    chunk_size, chunk_overlap = resolve_chunk_args(args)
    persist_directory = chroma_subdir(chunk_size, chunk_overlap)

    run_batch(
        parquet_path=args.parquet,
        embedder_path=args.embedder,
        sqlite_db=args.sqlite_db,
        persist_directory=persist_directory,
        gen_model=gen_model,
        eval_model=eval_model,
        model_type=args.model_type,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        index_limit=args.index_limit,
        sample_count=args.samples,
        sample_offset=args.sample_offset,
        max_workers=args.max_workers,
        judge_retries=args.judge_retries,
        llm_retries=args.llm_retries,
        rebuild_index=args.rebuild_index,
        rebuild_db=args.rebuild_db,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
