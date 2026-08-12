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

    console.print(
        f"[bold]Running:[/] {cfg.name}  subset={cfg.data.subset}  "
        f"limit={cfg.data.limit}  workers={cfg.runtime.max_workers}"
    )
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


@app.command("list-subsets")
def list_subsets() -> None:
    """List all RAGBench subsets and their domains."""
    from rag_kag.data_loaders import SUBSET_TO_DOMAIN

    table = Table("subset", "domain")
    for subset, domain in sorted(SUBSET_TO_DOMAIN.items(), key=lambda kv: (kv[1], kv[0])):
        table.add_row(subset, domain)
    console.print(table)


@app.command("data-status")
def data_status(
    local_root: Path = typer.Option(
        Path("data/raw/ragbench"),
        "--local-root",
        help="Root directory for local parquet files",
    ),
) -> None:
    """Show which phase-1.1 subsets are downloaded locally."""
    from rag_kag.data_loaders.download import list_local_status
    from rag_kag.data_loaders.eda import PHASE1_REPRESENTATIVE_SUBSETS

    status = list_local_status(PHASE1_REPRESENTATIVE_SUBSETS, local_root=local_root)
    table = Table("subset", "train", "validation", "test")
    for subset in PHASE1_REPRESENTATIVE_SUBSETS:
        splits = status[subset]
        table.add_row(
            subset,
            "yes" if splits["train"] else "—",
            "yes" if splits["validation"] else "—",
            "yes" if splits["test"] else "—",
        )
    console.print(table)


@app.command("download-data")
def download_data(
    subset: list[str] = typer.Option(
        None,
        "--subset",
        "-s",
        help="Subset(s) to download; default = all phase-1.1 representative subsets",
    ),
    split: list[str] = typer.Option(
        ["train"],
        "--split",
        help="Splits to download (train, validation, test)",
    ),
    local_root: Path = typer.Option(
        Path("data/raw/ragbench"),
        "--local-root",
        help="Destination root for parquet files",
    ),
    force: bool = typer.Option(False, "--force", help="Re-download even if file exists"),
) -> None:
    """Download RAGBench parquet files from Hugging Face Hub."""
    from rag_kag.data_loaders.download import HF_REPO, download_subsets
    from rag_kag.data_loaders.eda import PHASE1_REPRESENTATIVE_SUBSETS

    subsets = tuple(subset) if subset else PHASE1_REPRESENTATIVE_SUBSETS
    splits_t = tuple(split)
    results = download_subsets(
        subsets, splits=splits_t, local_root=local_root, force=force
    )
    ok = skipped = failed = 0
    for r in results:
        if r.error:
            failed += 1
            console.print(f"[red]FAIL[/] {r.subset}/{r.split}: {r.error}")
            console.print(
                f"       Browser: https://huggingface.co/datasets/{HF_REPO}/tree/main/{r.subset}"
            )
        elif r.skipped:
            skipped += 1
            console.print(f"[dim]skip[/] {r.subset}/{r.split} (already at {r.path})")
        else:
            ok += 1
            console.print(f"[green]ok[/] {r.subset}/{r.split} → {r.path}")
    console.print(f"Downloaded {ok}, skipped {skipped}, failed {failed}")


@app.command()
def eda(
    output: Path = typer.Option(
        Path("docs/eda/ragbench_eda.md"),
        "--output",
        "-o",
        help="Markdown report path",
    ),
    subset: list[str] = typer.Option(
        None,
        "--subset",
        "-s",
        help="Subset(s) to analyze; default = phase-1.1 representative list",
    ),
    split: str = typer.Option("train", "--split", help="RAGBench split"),
    limit: int | None = typer.Option(None, "--limit", help="Max examples per subset"),
    local_root: Path = typer.Option(
        Path("data/raw/ragbench"),
        "--local-root",
        help="Local parquet root",
    ),
    json_out: Path | None = typer.Option(
        None,
        "--json",
        help="Optional path for JSON stats (experiments/eda/ragbench_stats.json)",
    ),
) -> None:
    """Run RAGBench EDA: doc-length stats and per-domain chunk_size recommendations."""
    import json

    from rag_kag.data_loaders.eda import (
        PHASE1_REPRESENTATIVE_SUBSETS,
        subset_stats_json,
        write_eda_report,
    )

    subsets = tuple(subset) if subset else PHASE1_REPRESENTATIVE_SUBSETS
    console.print(
        f"[bold]EDA[/] subsets={len(subsets)} split={split} limit={limit or 'full'}"
    )
    stats = write_eda_report(
        output,
        subsets=subsets,
        split=split,
        limit=limit,
        local_root=local_root,
    )
    console.print(f"[green]Report:[/] {output}")

    errors = [s for s in stats if s.error]
    if errors:
        console.print(f"[yellow]Warning:[/] {len(errors)} subset(s) failed — see report")

    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(subset_stats_json(stats), indent=2))
        console.print(f"[green]JSON:[/] {json_out}")

    table = Table("subset", "n", "doc median", ">256%", "chunk", "overlap")
    for st in stats:
        if st.error:
            table.add_row(st.subset, "err", "-", "-", "-", "-")
        else:
            table.add_row(
                st.subset,
                str(st.n_examples),
                f"{st.doc_words_median:.0f}",
                f"{st.pct_docs_gt_256:.1f}%",
                str(st.recommended_chunk_size),
                str(st.recommended_overlap),
            )
    console.print(table)


@app.command("validate-trace")
def validate_trace(
    domain: str = typer.Option("covidqa", "--domain", "-d", help="RAGBench subset"),
    split: str = typer.Option("train", "--split", help="Dataset split"),
    limit: int = typer.Option(50, "--limit", help="Number of examples to validate"),
    tolerance: float = typer.Option(1e-3, "--tolerance", help="Max |ours - ref| for a match"),
    output: Path = typer.Option(
        Path("docs/findings/trace_validation.md"),
        "--output",
        "-o",
        help="Markdown report path",
    ),
) -> None:
    """Validate TRACe formulas against RAGBench reference scores (dataset responses)."""
    from rag_kag.evaluators.validate import validate_subset, write_validation_markdown

    report = validate_subset(domain, split=split, limit=limit, tolerance=tolerance)
    write_validation_markdown(report, output)
    console.print(f"[green]Report:[/] {output}")

    table = Table("metric", "matches", "avg ours", "avg ref", "avg Δ")
    for metric in (
        "context_relevance",
        "context_utilization",
        "completeness",
        "adherence",
    ):
        table.add_row(
            metric,
            f"{report.matches.get(metric, 0)}/{report.n}",
            f"{report.avg_ours.get(metric, 0):.4f}",
            f"{report.avg_reference.get(metric, 0):.4f}",
            f"{report.avg_delta.get(metric, 0):+.4f}",
        )
    console.print(table)
    if report.mismatches:
        console.print(f"[yellow]{len(report.mismatches)} per-example mismatches — see report[/]")


def _print_summary(path: Path) -> None:
    data = json.loads(path.read_text())
    table = Table(title=f"{data.get('name')}  ·  {data.get('subset')}  ·  n={data.get('n')}")
    table.add_column("metric")
    table.add_column("ours", justify="right")
    table.add_column("reference", justify="right")
    table.add_column("Δ", justify="right")
    avg = data.get("avg_trace", {})
    ref = data.get("avg_reference", {})
    delta = data.get("avg_delta", {})
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
            f"{delta.get(ours_key, 0):+.3f}" if ours_key in delta else "-",
        )
    console.print(table)
    if "avg_latency_s" in data:
        console.print(f"avg_latency: {data['avg_latency_s']:.2f}s")


if __name__ == "__main__":
    app()
