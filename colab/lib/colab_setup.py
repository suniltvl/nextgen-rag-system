"""Shared bootstrap for Colab notebooks."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

from colab.lib.experiment_config import ExperimentRun, resolve_experiment
from colab.lib.paths import drive_layout, validate_drive_paths

DEFAULT_DRIVE_ROOT = "/content/drive/MyDrive/capstone-rag-kag"


def mount_drive() -> None:
    try:
        from google.colab import drive

        drive.mount("/content/drive")
    except ImportError:
        print("Not in Colab — skipping drive.mount")


def install_requirements(repo_root: Path) -> None:
    req = repo_root / "colab" / "requirements-colab.txt"
    if req.exists():
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-r", str(req)])


def bootstrap(
    *,
    drive_root: str = DEFAULT_DRIVE_ROOT,
    preset: str | None = None,
    domain: str | None = None,
    dataset: str | None = None,
    overrides: dict | None = None,
    install_deps: bool = True,
    mount: bool = True,
) -> ExperimentRun:
    if mount:
        mount_drive()

    layout = drive_layout(drive_root)
    repo_root = layout["repo_root"]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    if install_deps:
        install_requirements(repo_root)

    env_file = layout["env_file"]
    if env_file.exists():
        load_dotenv(env_file)
    else:
        load_dotenv()

    try:
        from google.colab import userdata

        key = userdata.get("OPENAI_API_KEY")
        if key:
            os.environ["OPENAI_API_KEY"] = key
    except Exception:
        pass

    exp = resolve_experiment(
        drive_root=drive_root,
        preset=preset,
        domain=domain,
        dataset=dataset,
        overrides=overrides,
    )
    validate_drive_paths(
        drive_layout(exp.drive_root),
        dataset=exp.dataset,
        embed_model_path=exp.embed_model_path,
    )
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    return exp


def print_experiment(exp: ExperimentRun) -> None:
    print(f"Drive root:     {exp.drive_root}")
    print(f"Repo root:      {exp.repo_root}")
    print(f"Preset:         {exp.preset} ({exp.chunk_size}/{exp.chunk_overlap})")
    print(f"Domain/Dataset: {exp.domain} / {exp.dataset}")
    print(f"Parquet:        {exp.parquet_file}")
    print(f"Embedder:       {exp.embedder_dir}")
    print(f"Chroma:         {exp.chroma_dir}")
    print(f"SQLite:         {exp.sqlite_db}")
    print(f"Models:         gen={exp.gen_model} eval={exp.eval_model} type={exp.model_type}")
    print(f"OpenAI key set: {bool(os.getenv('OPENAI_API_KEY'))}")
