"""Build a chunker from its config. New strategies wire in here."""

from __future__ import annotations

from rag_kag.chunkers.base import Chunker
from rag_kag.chunkers.sliding_window import SlidingWindowChunker
from rag_kag.config import ChunkerCfg, SlidingWindowChunkerCfg


def build_chunker(cfg: ChunkerCfg) -> Chunker:
    if isinstance(cfg, SlidingWindowChunkerCfg):
        return SlidingWindowChunker(chunk_size=cfg.chunk_size, overlap=cfg.overlap)
    raise NotImplementedError(
        f"Chunker kind {cfg.kind!r} not yet implemented. "
        "Add it under src/rag_kag/chunkers/ and wire it into build_chunker()."
    )
