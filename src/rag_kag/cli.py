"""Typer CLI — `rag-kag run --config configs/v1_baseline.yaml`."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from rag_kag.config import ExperimentCfg

app = typer.Typer(add_completion=False, help="RAG-KAG capstone CLI")
console = Console()


@app.command()
def run(
    config: Path = typer.Option(..., "--config", "-c", help="Path to experiment YAML"),
    domain: str | None = typer.Option(
        None, "--domain", help="Override data.subset (e.g. covidqa, finqa)"
    ),
    limit: int | None = typer.Option(None, "--limit", help="Override data.limit"),
    output_dir: Path | None = typer.Option(
        None, "--output-dir", help="Override output_dir"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Print per-example pipeline steps"
    ),
) -> None:
    """Run a configured experiment end-to-end."""
    cfg = ExperimentCfg.from_yaml(config)
    if domain is not None:
        cfg.data.subset = domain
    if limit is not None:
        cfg.data.limit = limit
    if output_dir is not None:
        cfg.output_dir = output_dir

    # Imported here to keep `--help` snappy.
    from rag_kag.pipeline import Pipeline

    console.print(f"[bold]Running:[/] {cfg.name}  subset={cfg.data.subset}  limit={cfg.data.limit}")
    run_dir = Pipeline(cfg, verbose=verbose).run()
    console.print(f"[green]Done.[/] Output: {run_dir}")

    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        _print_summary(summary_path)


@app.command()
def show(run_dir: Path = typer.Argument(..., help="Path to a previous run dir")) -> None:
    """Pretty-print summary.json from a previous run."""
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        console.print(f"[red]No summary.json at {summary_path}[/]")
        raise typer.Exit(1)
    _print_summary(summary_path)


@app.command()
def list_subsets() -> None:
    """List all RAGBench subsets and their domains."""
    from rag_kag.data_loaders import SUBSET_TO_DOMAIN

    table = Table("subset", "domain")
    for subset, domain in sorted(SUBSET_TO_DOMAIN.items(), key=lambda kv: (kv[1], kv[0])):
        table.add_row(subset, domain)
    console.print(table)


def _print_summary(path: Path) -> None:
    data = json.loads(path.read_text())
    table = Table(title=f"{data.get('name')}  ·  {data.get('subset')}  ·  n={data.get('n')}")
    table.add_column("metric")
    table.add_column("ours", justify="right")
    table.add_column("reference", justify="right")
    avg = data.get("avg_trace", {})
    ref = data.get("avg_reference", {})
    pairs = [
        ("context_relevance", "relevance_score"),
        ("context_utilization", "utilization_score"),
        ("completeness", "completeness_score"),
        ("adherence", "adherence_score"),
    ]
    for ours_key, ref_key in pairs:
        ours_val = avg.get(ours_key)
        ref_val = ref.get(ref_key)
        table.add_row(
            ours_key,
            f"{ours_val:.3f}" if ours_val is not None else "-",
            f"{ref_val:.3f}" if ref_val is not None else "-",
        )
    console.print(table)
    if "avg_latency_s" in data:
        console.print(f"avg_latency: {data['avg_latency_s']:.2f}s")


if __name__ == "__main__":
    app()
