"""Build a generator from its config."""

from __future__ import annotations

from rag_kag.config import GeneratorCfg, LLMGeneratorCfg
from rag_kag.generators.base import Generator
from rag_kag.generators.llm import LLMGenerator


def build_generator(cfg: GeneratorCfg) -> Generator:
    if isinstance(cfg, LLMGeneratorCfg):
        return LLMGenerator(
            model=cfg.model,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            prompt_template=cfg.prompt_template,
        )
    raise NotImplementedError(f"Generator kind {cfg.kind!r} not supported.")
