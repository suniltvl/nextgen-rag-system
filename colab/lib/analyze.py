"""Analyze SQLite results: compare, CSV export, matplotlib charts."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from colab.lib.experiment_config import ExperimentRun


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def latest_session(db_path: Path) -> str | None:
    conn = _connect(db_path)
    try:
        row = conn.execute(
            "SELECT session_id FROM nextgenrag_v1 ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        return row["session_id"] if row else None
    finally:
        conn.close()


def table_counts(db_path: Path) -> dict[str, int]:
    conn = _connect(db_path)
    try:
        counts = {}
        for table in (
            "nextgenrag_questions",
            "nextgenrag_sample_questions",
            "nextgenrag_v1",
        ):
            row = conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()
            counts[table] = row["n"]
        return counts
    finally:
        conn.close()


def session_summary(db_path: Path, session_id: str | None = None) -> dict:
    conn = _connect(db_path)
    try:
        sid = session_id or latest_session(db_path)
        if not sid:
            return {}
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
        return dict(row) if row else {}
    finally:
        conn.close()


def compare_dataframe(db_path: Path, session_id: str | None = None) -> pd.DataFrame:
    conn = _connect(db_path)
    try:
        sid = session_id or latest_session(db_path)
        if not sid:
            return pd.DataFrame()
        rows = conn.execute(
            """
            SELECT v.id,
                   s.query,
                   q.gpt_relevance AS ref_relevance,
                   q.gpt_utilization AS ref_utilization,
                   q.gpt_completeness AS ref_completeness,
                   q.gpt_adherence AS ref_adherence,
                   v.relevance AS our_relevance,
                   v.utilization AS our_utilization,
                   v.completeness AS our_completeness,
                   v.adherence AS our_adherence,
                   v.parse_error,
                   v.response,
                   v.context
            FROM nextgenrag_v1 v
            JOIN nextgenrag_sample_questions s ON v.id = s.id
            LEFT JOIN nextgenrag_questions q ON s.query = q.query
            WHERE v.session_id = ?
            ORDER BY v.id
            """,
            (sid,),
        ).fetchall()
        return pd.DataFrame([dict(r) for r in rows])
    finally:
        conn.close()


def export_csv(db_path: Path, output: Path, session_id: str | None = None) -> Path:
    df = compare_dataframe(db_path, session_id)
    if df.empty:
        raise ValueError("No results to export")
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    return output


def plot_trace_metrics(
    db_path: Path,
    output: Path,
    session_id: str | None = None,
    *,
    show: bool = True,
) -> Path:
    import matplotlib.pyplot as plt

    summary = session_summary(db_path, session_id)
    if not summary:
        raise ValueError("No session summary available")

    ref_df = compare_dataframe(db_path, session_id)
    metrics = ["relevance", "utilization", "completeness", "adherence"]

    ref_vals = [
        ref_df["ref_relevance"].mean(),
        ref_df["ref_utilization"].mean(),
        ref_df["ref_completeness"].mean(),
        ref_df["ref_adherence"].mean(),
    ]
    our_vals = [
        ref_df["our_relevance"].mean(),
        ref_df["our_utilization"].mean(),
        ref_df["our_completeness"].mean(),
        ref_df["our_adherence"].mean(),
    ]

    x = range(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - width / 2 for i in x], ref_vals, width, label="Reference (RAGBench GPT)")
    ax.bar([i + width / 2 for i in x], our_vals, width, label="Ours")
    ax.set_xticks(list(x))
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Average score")
    ax.set_title(f"TRACe metrics — session {summary['session_id'][:8]}...")
    ax.legend()
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=120)
    if show:
        plt.show()
    plt.close(fig)
    return output


def analyze_run(exp: ExperimentRun, session_id: str | None = None) -> dict:
    counts = table_counts(exp.sqlite_db)
    summary = session_summary(exp.sqlite_db, session_id)
    df = compare_dataframe(exp.sqlite_db, session_id)
    csv_path = export_csv(exp.sqlite_db, exp.csv_path, session_id)
    chart_path = plot_trace_metrics(exp.sqlite_db, exp.chart_path, session_id, show=False)
    return {
        "counts": counts,
        "session_summary": summary,
        "compare_rows": len(df),
        "csv_path": str(csv_path),
        "chart_path": str(chart_path),
        "dataframe": df,
    }
