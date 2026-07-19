"""Inspect SQLite results from Saurav-style covidqa runs."""

from __future__ import annotations

import argparse
import csv
import sqlite3
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = REPO_ROOT / "sqldb/saurav_covidqa.db"


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def show_counts(db_path: Path) -> None:
    conn = _connect(db_path)
    try:
        for table in (
            "nextgenrag_questions",
            "nextgenrag_sample_questions",
            "nextgenrag_v1",
        ):
            row = conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()
            print(f"{table}: {row['n']}")
    finally:
        conn.close()


def _latest_session(conn: sqlite3.Connection) -> str | None:
    row = conn.execute(
        "SELECT session_id FROM nextgenrag_v1 ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    return row["session_id"] if row else None


def show_session(db_path: Path, session_id: str | None) -> None:
    conn = _connect(db_path)
    try:
        sid = session_id if session_id and session_id != "latest" else _latest_session(conn)
        if not sid:
            print("No sessions in nextgenrag_v1")
            return
        row = conn.execute(
            """
            SELECT session_id,
                   COUNT(*) AS n,
                   AVG(relevance) AS avg_relevance,
                   AVG(utilization) AS avg_utilization,
                   AVG(completeness) AS avg_completeness,
                   AVG(adherence) AS avg_adherence,
                   SUM(parse_error) AS parse_errors,
                   MIN(chunk_size) AS chunk_size,
                   MIN(chunk_overlap) AS chunk_overlap
            FROM nextgenrag_v1
            WHERE session_id = ?
            """,
            (sid,),
        ).fetchone()
        print(f"session_id: {row['session_id']}")
        print(f"rows: {row['n']}")
        print(f"chunk: {row['chunk_size']}/{row['chunk_overlap']}")
        print(
            "avg scores:",
            {
                "relevance": round(row["avg_relevance"] or 0, 4),
                "utilization": round(row["avg_utilization"] or 0, 4),
                "completeness": round(row["avg_completeness"] or 0, 4),
                "adherence": round(row["avg_adherence"] or 0, 4),
            },
        )
        print(f"parse_errors: {row['parse_errors']}")
    finally:
        conn.close()


def show_compare(db_path: Path, session_id: str | None, limit: int) -> None:
    conn = _connect(db_path)
    try:
        sid = session_id if session_id and session_id != "latest" else _latest_session(conn)
        if not sid:
            print("No sessions in nextgenrag_v1")
            return
        rows = conn.execute(
            """
            SELECT v.id,
                   substr(s.query, 1, 70) AS query,
                   q.gpt_relevance AS ref_relevance,
                   q.gpt_utilization AS ref_utilization,
                   q.gpt_completeness AS ref_completeness,
                   q.gpt_adherence AS ref_adherence,
                   v.relevance AS our_relevance,
                   v.utilization AS our_utilization,
                   v.completeness AS our_completeness,
                   v.adherence AS our_adherence,
                   v.parse_error
            FROM nextgenrag_v1 v
            JOIN nextgenrag_sample_questions s ON v.id = s.id
            LEFT JOIN nextgenrag_questions q ON s.query = q.query
            WHERE v.session_id = ?
            ORDER BY v.id
            LIMIT ?
            """,
            (sid, limit),
        ).fetchall()
        print(f"session_id: {sid}\n")
        for row in rows:
            print(f"[{row['id']}] {row['query']}...")
            print(f"  ref:  rel={row['ref_relevance']} util={row['ref_utilization']} comp={row['ref_completeness']} adh={row['ref_adherence']}")
            print(f"  ours: rel={row['our_relevance']} util={row['our_utilization']} comp={row['our_completeness']} adh={row['our_adherence']} err={row['parse_error']}")
            print()
    finally:
        conn.close()


def export_csv(db_path: Path, session_id: str | None, output: Path) -> None:
    conn = _connect(db_path)
    try:
        sid = session_id if session_id and session_id != "latest" else _latest_session(conn)
        if not sid:
            raise SystemExit("No sessions in nextgenrag_v1")
        rows = conn.execute(
            """
            SELECT v.*, s.query
            FROM nextgenrag_v1 v
            JOIN nextgenrag_sample_questions s ON v.id = s.id
            WHERE v.session_id = ?
            ORDER BY v.id
            """,
            (sid,),
        ).fetchall()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows([dict(r) for r in rows])
        print(f"Exported {len(rows)} rows to {output}")
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--session", default="latest")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--counts", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--export-csv", type=Path, default=None)
    args = parser.parse_args()

    if args.counts:
        show_counts(args.db)
    elif args.compare:
        show_compare(args.db, args.session, args.limit)
    elif args.export_csv:
        export_csv(args.db, args.session, args.export_csv)
    else:
        show_session(args.db, args.session)


if __name__ == "__main__":
    main()
