"""Build a chunker from its config. New strategies wire in here."""

from __future__ import annotations

from rag_kag.chunkers.base import Chunker
from rag_kag.chunkers.sliding_window import SlidingWindowChunker
from rag_kag.chunkers.semantic import SemanticChunker
from rag_kag.config import ChunkerCfg, SlidingWindowChunkerCfg, SemanticChunkerCfg


def build_chunker(cfg: ChunkerCfg) -> Chunker:
    if isinstance(cfg, SlidingWindowChunkerCfg):
        return SlidingWindowChunker(chunk_size=cfg.chunk_size, overlap=cfg.overlap)
    elif isinstance(cfg, SemanticChunkerCfg):
      return SemanticChunker(model_name=cfg.model_name, min_sentences=cfg.min_sentences, max_sentences=cfg.max_sentences, breakpoint_threshold=cfg.breakpoint_threshold)
    raise NotImplementedError(
        f"Chunker kind {cfg.kind!r} not yet implemented. "
        "Add it under src/rag_kag/chunkers/ and wire it into build_chunker()."
    )
