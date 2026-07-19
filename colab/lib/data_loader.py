"""Load ragbench parquet from Google Drive; HF fallback if missing."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from colab.lib.paths import parquet_path


def deduplicate_data(data: list[dict]) -> dict[str, dict]:
    data_dict: dict[str, dict] = {}
    for item in data:
        document = " ".join(item["documents"])
        if document in data_dict:
            data_dict[document]["docid"].append(item["id"])
        else:
            data_dict[document] = {"docid": [item["id"]]}
    return data_dict


def load_parquet_rows(parquet_file: Path, *, limit: int = 0) -> list[dict]:
    df = pd.read_parquet(parquet_file)
    if limit <= 0:
        rows = df.to_dict(orient="records")
    else:
        rows = df.head(limit).to_dict(orient="records")
    for row in rows:
        row["documents"] = list(row["documents"])
    return rows


def load_ragbench_parquet(
    data_dir: Path,
    dataset: str,
    *,
    split: str = "train",
    limit: int = 0,
) -> tuple[Path, list[dict]]:
    path = parquet_path(data_dir, dataset, split)
    return path, load_parquet_rows(path, limit=limit)


def load_ragbench_hf_fallback(dataset: str, *, split: str = "train", limit: int = 0) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("rungalileo/ragbench", dataset, split=split)
    rows = [dict(r) for r in ds]
    if limit > 0:
        rows = rows[:limit]
    for row in rows:
        row["documents"] = list(row["documents"])
    return rows
