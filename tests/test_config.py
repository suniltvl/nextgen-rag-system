"""ExperimentCfg.from_yaml round-trip + V1 baseline parses cleanly."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_kag.config import ExperimentCfg, SlidingWindowChunkerCfg


REPO_ROOT = Path(__file__).resolve().parent.parent


def test_v1_baseline_parses() -> None:
    cfg = ExperimentCfg.from_yaml(REPO_ROOT / "configs" / "v1_baseline.yaml")
    assert cfg.name == "v1_baseline"
    assert cfg.data.subset == "covidqa"
    assert isinstance(cfg.chunker, SlidingWindowChunkerCfg)
    assert cfg.chunker.chunk_size == 256
    assert cfg.retriever.top_k == 5


def test_chunker_swap_via_yaml(tmp_path: Path) -> None:
    """One-line YAML swap; no Python change needed (proposal §5.1 step 6)."""
    yaml = """
name: v2
data:
  subset: covidqa
chunker:
  kind: sentence_aware
  max_sentences: 5
"""
    p = tmp_path / "v2.yaml"
    p.write_text(yaml)
    cfg = ExperimentCfg.from_yaml(p)
    assert cfg.chunker.kind == "sentence_aware"


def test_unknown_subset_rejected_by_loader() -> None:
    from rag_kag.data_loaders import RAGBenchLoader

    with pytest.raises(ValueError):
        RAGBenchLoader(subset="nonexistent")
