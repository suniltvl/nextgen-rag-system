"""Download RAGBench parquet files to data/raw/ragbench/<subset>/.

Uses Hugging Face Hub direct parquet URLs (galileo-ai/ragbench). On corporate
networks where the LFS CDN is MITM'd, use browser download per docs/dev_setup.md
or run this script from a network without Zscaler.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rag_kag.data_loaders.eda import PHASE1_REPRESENTATIVE_SUBSETS
from rag_kag.data_loaders.ragbench import RAGBenchLoader

HF_REPO = RAGBenchLoader.HF_REPO
DEFAULT_LOCAL_ROOT = RAGBenchLoader.DEFAULT_LOCAL_ROOT
SPLITS = ("train", "validation", "test")


@dataclass(slots=True)
class DownloadResult:
    subset: str
    split: str
    path: Path | None
    skipped: bool
    error: str | None = None


def _parquet_filename(split: str) -> str:
    return f"{split}-00000-of-00001.parquet"


def local_parquet_path(
    subset: str,
    split: str,
    *,
    local_root: Path = DEFAULT_LOCAL_ROOT,
) -> Path:
    return local_root / subset / _parquet_filename(split)


def is_downloaded(
    subset: str,
    split: str,
    *,
    local_root: Path = DEFAULT_LOCAL_ROOT,
) -> bool:
    return local_parquet_path(subset, split, local_root=local_root).is_file()


def download_subset_split(
    subset: str,
    split: str,
    *,
    local_root: Path = DEFAULT_LOCAL_ROOT,
    force: bool = False,
) -> DownloadResult:
    """Download one parquet file for subset/split."""
    dest = local_parquet_path(subset, split, local_root=local_root)
    if dest.is_file() and not force:
        return DownloadResult(subset=subset, split=split, path=dest, skipped=True)

    dest.parent.mkdir(parents=True, exist_ok=True)
    repo_path = f"{subset}/{_parquet_filename(split)}"

    try:
        import shutil

        from huggingface_hub import hf_hub_download

        cached = hf_hub_download(
            repo_id=HF_REPO,
            repo_type="dataset",
            filename=repo_path,
        )
        shutil.copy2(cached, dest)
        return DownloadResult(subset=subset, split=split, path=dest, skipped=False)
    except Exception as exc:
        return DownloadResult(
            subset=subset,
            split=split,
            path=None,
            skipped=False,
            error=str(exc),
        )


def download_subsets(
    subsets: tuple[str, ...] | list[str] = PHASE1_REPRESENTATIVE_SUBSETS,
    *,
    splits: tuple[str, ...] = ("train",),
    local_root: Path = DEFAULT_LOCAL_ROOT,
    force: bool = False,
) -> list[DownloadResult]:
    results: list[DownloadResult] = []
    for subset in subsets:
        for split in splits:
            results.append(
                download_subset_split(
                    subset, split, local_root=local_root, force=force
                )
            )
    return results


def list_local_status(
    subsets: tuple[str, ...] | list[str] = PHASE1_REPRESENTATIVE_SUBSETS,
    *,
    splits: tuple[str, ...] = SPLITS,
    local_root: Path = DEFAULT_LOCAL_ROOT,
) -> dict[str, dict[str, bool]]:
    out: dict[str, dict[str, bool]] = {}
    for subset in subsets:
        out[subset] = {
            split: is_downloaded(subset, split, local_root=local_root) for split in splits
        }
    return out
