"""Prompt templates.

The default template enforces grounding ("answer only from CONTEXT") and a
hard rejection clause that the RGB benchmark looks for verbatim. Keeping the
RGB rejection / error-detection strings here means downstream RGB metrics can
match against the same constants without duplication.
"""

from __future__ import annotations

# Phrases the RGB paper expects when the model has nothing to say or has
# detected counterfactual content. RGB matches these substrings.
RGB_REJECTION_PHRASE = "I can not answer the question because of insufficient information in documents"
RGB_ERROR_DETECTION_PHRASE = "There are factual errors in the provided documents"


GROUNDED_DEFAULT = """You are a careful research assistant. Answer the question using ONLY the information in CONTEXT.

Hard rules:
- If the answer is not present in CONTEXT, reply EXACTLY: {rejection}
- If CONTEXT contains factual errors that contradict well-known facts, begin your reply with: {error_detection}
- Do not fabricate citations, dates, names, or numbers.
- Keep the answer concise.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


PROMPT_TEMPLATES: dict[str, str] = {
    "grounded_default": GROUNDED_DEFAULT,
}


def render_prompt(
    template_name: str,
    *,
    question: str,
    context: str,
) -> str:
    template = PROMPT_TEMPLATES.get(template_name)
    if template is None:
        raise KeyError(
            f"Unknown prompt template {template_name!r}. "
            f"Known: {sorted(PROMPT_TEMPLATES)}"
        )
    return template.format(
        question=question,
        context=context,
        rejection=RGB_REJECTION_PHRASE,
        error_detection=RGB_ERROR_DETECTION_PHRASE,
    )


def format_context(retrieved_texts: list[str]) -> str:
    """Numbered context blocks. Numbering helps the model cite implicitly."""
    if not retrieved_texts:
        return "(no documents retrieved)"
    return "\n\n".join(f"[{i + 1}] {t}" for i, t in enumerate(retrieved_texts))
