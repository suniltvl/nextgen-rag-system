"""litellm-backed generator. One interface, every provider.

Why litellm: the project requires evaluating 4-5 open-source LLMs for RGB
(noise robustness, negative rejection, etc.). litellm lets us swap between
Ollama / OpenAI / Anthropic / Bedrock / HF endpoints by changing only the
`model` string in YAML — no code per provider.
"""

from __future__ import annotations

import time

from rag_kag.generators.base import Generator
from rag_kag.generators.prompts import format_context, render_prompt
from rag_kag.types import GenerationResult, RetrievedChunk


class LLMGenerator(Generator):
    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1200,
        prompt_template: str = "grounded_default",
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.prompt_template = prompt_template

    def generate(self, question: str, retrieved: list[RetrievedChunk]) -> GenerationResult:
        # Lazy import — litellm pulls a lot at import time.
        import litellm

        context = format_context([r.chunk.text for r in retrieved])
        prompt = render_prompt(self.prompt_template, question=question, context=context)

        start = time.perf_counter()
        response = litellm.completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        latency = time.perf_counter() - start

        # litellm normalizes responses to OpenAI-style dicts.
        choice = response["choices"][0]
        answer = (choice["message"]["content"] or "").strip()
        usage_raw = response.get("usage") or {}
        usage = {
            "prompt_tokens": int(usage_raw.get("prompt_tokens", 0)),
            "completion_tokens": int(usage_raw.get("completion_tokens", 0)),
            "total_tokens": int(usage_raw.get("total_tokens", 0)),
        }

        return GenerationResult(
            answer=answer,
            retrieved=retrieved,
            prompt=prompt,
            model=self.model,
            usage=usage,
            latency_s=latency,
        )
