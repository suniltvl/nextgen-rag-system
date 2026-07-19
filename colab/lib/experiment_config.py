"""Load and resolve experiment matrix from YAML."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from colab.lib.paths import (
    chroma_path,
    drive_layout,
    embedder_path,
    parquet_path,
    run_dir,
    sqlite_path,
)

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "experiments.yaml"


@dataclass
class ExperimentRun:
    drive_root: Path
    repo_root: Path
    domain: str
    dataset: str
    preset: str
    chunk_size: int
    chunk_overlap: int
    embed_model_path: str
    model_type: str
    gen_model: str
    eval_model: str
    search_type: str
    search_k: int
    retrieval_type: str
    reranker: str | None
    samples: int
    sample_offset: int
    index_limit: int
    max_workers: int
    judge_retries: int
    llm_retries: int
    rebuild_index: bool
    rebuild_db: bool
    resume: bool
    parquet_split: str
    parquet_file: Path
    embedder_dir: Path
    chroma_dir: Path
    sqlite_db: Path
    summary_dir: Path
    csv_path: Path
    chart_path: Path

    @property
    def search_kwargs(self) -> dict:
        return {"k": self.search_k}


def load_yaml_config(config_path: Path | None = None) -> dict:
    path = config_path or CONFIG_PATH
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_experiment(
    *,
    drive_root: str | None = None,
    preset: str | None = None,
    domain: str | None = None,
    dataset: str | None = None,
    overrides: dict | None = None,
    config_path: Path | None = None,
) -> ExperimentRun:
    cfg = load_yaml_config(config_path)
    defaults = dict(cfg.get("defaults") or {})
    drive_cfg = cfg.get("drive") or {}
    presets = cfg.get("presets") or {}
    domains = cfg.get("domains") or {}

    root = Path(drive_root or drive_cfg.get("root") or defaults.get("drive_root"))
    chosen_domain = domain or defaults.get("domain", "biomedical")
    domain_cfg = domains.get(chosen_domain, {})
    chosen_dataset = dataset or domain_cfg.get("dataset") or defaults.get("dataset", "covidqa")
    chosen_preset = preset or defaults.get("preset", "covidqa_tuned")
    if chosen_preset not in presets:
        raise KeyError(f"Unknown preset: {chosen_preset}")

    chunk_size = presets[chosen_preset]["chunk_size"]
    chunk_overlap = presets[chosen_preset]["chunk_overlap"]

    merged = {**defaults, **domain_cfg, **(overrides or {})}
    merged["domain"] = chosen_domain
    merged["dataset"] = chosen_dataset
    merged["preset"] = chosen_preset
    merged["chunk_size"] = chunk_size
    merged["chunk_overlap"] = chunk_overlap

    layout = drive_layout(root)
    split = merged.get("parquet_split", "train")
    embed_rel = merged.get("embed_model_path", "bge-small-en-v1.5")
    run_output = run_dir(layout["experiments_dir"], chosen_preset, chosen_dataset)

    return ExperimentRun(
        drive_root=layout["drive_root"],
        repo_root=layout["repo_root"],
        domain=chosen_domain,
        dataset=chosen_dataset,
        preset=chosen_preset,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embed_model_path=embed_rel,
        model_type=merged.get("model_type", "openai"),
        gen_model=merged.get("gen_model", "gpt-4o-mini"),
        eval_model=merged.get("eval_model", "gpt-4o-mini"),
        search_type=merged.get("search_type", "similarity"),
        search_k=int(merged.get("search_k", 3)),
        retrieval_type=merged.get("retrieval_type", "dense"),
        reranker=merged.get("reranker"),
        samples=int(merged.get("samples", 50)),
        sample_offset=int(merged.get("sample_offset", 0)),
        index_limit=int(merged.get("index_limit", 0)),
        max_workers=int(merged.get("max_workers", 2)),
        judge_retries=int(merged.get("judge_retries", 3)),
        llm_retries=int(merged.get("llm_retries", 3)),
        rebuild_index=bool(merged.get("rebuild_index", False)),
        rebuild_db=bool(merged.get("rebuild_db", False)),
        resume=bool(merged.get("resume", True)),
        parquet_split=split,
        parquet_file=parquet_path(layout["data_dir"], chosen_dataset, split),
        embedder_dir=embedder_path(layout["models_dir"], embed_rel),
        chroma_dir=chroma_path(layout["database_dir"], chosen_domain, chunk_size, chunk_overlap),
        sqlite_db=sqlite_path(layout["sqldb_dir"], chosen_dataset),
        summary_dir=run_output,
        csv_path=run_output / "results.csv",
        chart_path=run_output / "trace_metrics.png",
    )
