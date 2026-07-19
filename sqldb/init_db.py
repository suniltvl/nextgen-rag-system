"""Initialize and seed SQLite tables for Saurav Postgres parity."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"
DEFAULT_PARQUET = REPO_ROOT / "data/raw/ragbench/covidqa/train-00000-of-00001.parquet"
DEFAULT_DB = REPO_ROOT / "sqldb/saurav_covidqa.db"

DEFAULT_DOMAIN = "biomedical"
DEFAULT_DATASET = "covidqa"


def apply_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))


def seed_questions(
    conn: sqlite3.Connection,
    parquet_path: Path,
    *,
    force: bool = False,
    domain_name: str = DEFAULT_DOMAIN,
    dataset_name: str = DEFAULT_DATASET,
) -> int:
    existing = conn.execute("SELECT COUNT(*) FROM nextgenrag_questions").fetchone()[0]
    if existing and not force:
        return existing

    df = pd.read_parquet(parquet_path)
    quest_dict: dict[str, dict] = {}

    for row in df.to_dict(orient="records"):
        question = row["question"]
        gen_model = str(row.get("generation_model_name") or "")
        entry = {
            "adherence": int(row["adherence_score"]) if row.get("adherence_score") is not None else None,
            "relevance": row.get("relevance_score"),
            "utilization": row.get("utilization_score"),
            "completeness": row.get("completeness_score"),
        }
        if question not in quest_dict:
            quest_dict[question] = {}
        if gen_model.startswith("gpt"):
            quest_dict[question].update(
                {
                    "gpt_adherence": entry["adherence"],
                    "gpt_relevance": entry["relevance"],
                    "gpt_utilization": entry["utilization"],
                    "gpt_completeness": entry["completeness"],
                }
            )
        elif gen_model.startswith("claude"):
            quest_dict[question].update(
                {
                    "claude_adherence": entry["adherence"],
                    "claude_relevance": entry["relevance"],
                    "claude_utilization": entry["utilization"],
                    "claude_completeness": entry["completeness"],
                }
            )

    conn.execute("DELETE FROM nextgenrag_questions")
    for query, scores in quest_dict.items():
        conn.execute(
            """
            INSERT INTO nextgenrag_questions (
                domain_name, dataset_name, query,
                gpt_adherence, gpt_relevance, gpt_utilization, gpt_completeness,
                claude_adherence, claude_relevance, claude_utilization, claude_completeness
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                domain_name,
                dataset_name,
                query,
                scores.get("gpt_adherence"),
                scores.get("gpt_relevance"),
                scores.get("gpt_utilization"),
                scores.get("gpt_completeness"),
                scores.get("claude_adherence"),
                scores.get("claude_relevance"),
                scores.get("claude_utilization"),
                scores.get("claude_completeness"),
            ),
        )
    return len(quest_dict)


def seed_samples(
    conn: sqlite3.Connection,
    parquet_path: Path,
    *,
    sample_count: int,
    sample_offset: int,
) -> int:
    df = pd.read_parquet(parquet_path)
    slice_df = df.iloc[sample_offset : sample_offset + sample_count]
    conn.execute("DELETE FROM nextgenrag_sample_questions")
    for i, row in enumerate(slice_df.itertuples(index=False), start=1):
        conn.execute(
            "INSERT INTO nextgenrag_sample_questions (id, query) VALUES (?, ?)",
            (i, row.question),
        )
    return len(slice_df)


def init_db(
    db_path: Path,
    parquet_path: Path,
    *,
    rebuild: bool,
    sample_count: int,
    sample_offset: int,
    domain_name: str = DEFAULT_DOMAIN,
    dataset_name: str = DEFAULT_DATASET,
) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if rebuild and db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    try:
        apply_schema(conn)
        n_questions = seed_questions(
            conn,
            parquet_path,
            force=rebuild,
            domain_name=domain_name,
            dataset_name=dataset_name,
        )
        n_samples = seed_samples(
            conn,
            parquet_path,
            sample_count=sample_count,
            sample_offset=sample_offset,
        )
        conn.commit()
        print(f"SQLite ready: {db_path}")
        print(f"  nextgenrag_questions: {n_questions} rows")
        print(f"  nextgenrag_sample_questions: {n_samples} rows")
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--sample-offset", type=int, default=0)
    args = parser.parse_args()
    init_db(
        args.db,
        args.parquet,
        rebuild=args.rebuild,
        sample_count=args.samples,
        sample_offset=args.sample_offset,
    )


if __name__ == "__main__":
    main()
