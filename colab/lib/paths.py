"""Path helpers for Google Drive capstone-rag-kag layout."""

from __future__ import annotations

from pathlib import Path

DEFAULT_DRIVE_ROOT = "/content/drive/MyDrive/capstone-rag-kag"


def drive_layout(drive_root: str | Path) -> dict[str, Path]:
    root = Path(drive_root)
    runtime = root / "runtime"
    return {
        "drive_root": root,
        "repo_root": root / "repo",
        "data_dir": root / "data" / "raw" / "ragbench",
        "models_dir": root / "models",
        "runtime_dir": runtime,
        "database_dir": runtime / "database",
        "sqldb_dir": runtime / "sqldb",
        "experiments_dir": runtime / "experiments" / "runs",
        "env_file": root / ".env",
    }


def parquet_path(data_dir: Path, dataset: str, split: str = "train") -> Path:
    dataset_dir = data_dir / dataset
    exact = dataset_dir / f"{split}-00000-of-00001.parquet"
    if exact.exists():
        return exact
    matches = sorted(dataset_dir.glob(f"{split}*.parquet"))
    if not matches:
        raise FileNotFoundError(f"No {split} parquet under {dataset_dir}")
    return matches[0]


def embedder_path(models_dir: Path, embed_model_path: str) -> Path:
    path = models_dir / embed_model_path
    if not path.exists():
        raise FileNotFoundError(f"Embedder not found: {path}")
    return path


def chroma_path(database_dir: Path, domain: str, chunk_size: int, chunk_overlap: int) -> Path:
    return database_dir / f"{domain}_{chunk_size}_{chunk_overlap}"


def sqlite_path(sqldb_dir: Path, dataset: str) -> Path:
    return sqldb_dir / f"{dataset}.db"


def run_dir(experiments_dir: Path, preset: str, dataset: str) -> Path:
    return experiments_dir / f"{preset}__{dataset}"


def validate_drive_paths(layout: dict[str, Path], *, dataset: str, embed_model_path: str) -> None:
    missing: list[str] = []
    if not layout["drive_root"].exists():
        missing.append(str(layout["drive_root"]))
    try:
        parquet_path(layout["data_dir"], dataset)
    except FileNotFoundError as err:
        missing.append(str(err))
    embed = layout["models_dir"] / embed_model_path
    if not embed.exists():
        missing.append(str(embed))
    if missing:
        raise FileNotFoundError(
            "Missing Drive assets:\n  - " + "\n  - ".join(missing)
        )
