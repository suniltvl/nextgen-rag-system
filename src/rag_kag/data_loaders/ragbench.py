"""RAGBench loader — adapts the HF dataset into our internal `Example` schema.

Two source modes are supported:
1. **HF Hub (default)** — load from ``galileo-ai/ragbench``. Requires network
   access; in restricted environments the CDN may be unreachable.
2. **Local parquet** — point at ``data/raw/ragbench/<subset>/`` containing the
   parquet files downloaded manually. Discovered automatically when no
   ``cache_dir`` is supplied; explicit override via ``local_root``.

Subset → domain mapping is sourced from the proposal §4.1 table.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from rag_kag.types import Example, Sentence


SUBSET_TO_DOMAIN: dict[str, str] = {
    # biomedical research
    "pubmedqa": "biomedical",
    "covidqa": "biomedical",
    # general knowledge
    "hotpotqa": "general",
    "msmarco": "general",
    "hagrid": "general",
    "expertqa": "general",
    # legal
    "cuad": "legal",
    # customer support
    "delucionqa": "customer_support",
    "emanual": "customer_support",
    "techqa": "customer_support",
    # finance
    "finqa": "finance",
    "tatqa": "finance",
}


# Score field names in RAGBench rows. We capture whichever are present.
_SCORE_FIELDS: tuple[str, ...] = (
    "relevance_score",
    "utilization_score",
    "completeness_score",
    "adherence_score",
    "gpt35_relevance",
    "gpt35_utilization",
    "gpt35_completeness",
    "gpt35_adherence",
)


class RAGBenchLoader:
    """Loader for a single RAGBench subset.

    Resolution order:
    1. If ``local_root`` is provided, load parquet files from
       ``<local_root>/<subset>/*.parquet`` (or ``<split>-*.parquet`` if those
       files exist).
    2. Otherwise, if ``./data/raw/ragbench/<subset>/`` exists in the current
       working directory, use that.
    3. Otherwise, fall back to ``datasets.load_dataset`` against HF Hub.
    """

    HF_REPO = "galileo-ai/ragbench"
    DEFAULT_LOCAL_ROOT = Path("data/raw/ragbench")

    def __init__(
        self,
        subset: str,
        split: str = "test",
        cache_dir: str | Path | None = None,
        local_root: str | Path | None = None,
    ):
        if subset not in SUBSET_TO_DOMAIN:
            raise ValueError(
                f"Unknown RAGBench subset {subset!r}. "
                f"Expected one of: {sorted(SUBSET_TO_DOMAIN)}"
            )
        self.subset = subset
        self.split = split
        self.domain = SUBSET_TO_DOMAIN[subset]
        self.cache_dir = str(cache_dir) if cache_dir else None
        self.local_root = Path(local_root) if local_root else self.DEFAULT_LOCAL_ROOT

    # --- public API ------------------------------------------------------

    def iter_examples(self, limit: int | None = None) -> Iterator[Example]:
        """Yield normalized examples from whichever source is available."""
        local_files = self._find_local_parquets()
        if local_files:
            yield from self._iter_from_parquet(local_files, limit=limit)
        else:
            yield from self._iter_from_hf(limit=limit)

    # --- source backends -------------------------------------------------

    def _find_local_parquets(self) -> list[Path]:
        subset_dir = self.local_root / self.subset
        if not subset_dir.is_dir():
            return []
        # Prefer split-prefixed files when present, else any parquet.
        split_files = sorted(subset_dir.glob(f"{self.split}-*.parquet"))
        if split_files:
            return split_files
        return sorted(subset_dir.glob("*.parquet"))

    def _iter_from_parquet(
        self, files: list[Path], limit: int | None
    ) -> Iterator[Example]:
        # Lazy import to keep package import light when only HF is used.
        import pandas as pd

        emitted = 0
        for f in files:
            df = pd.read_parquet(f)
            for i, row in enumerate(df.itertuples(index=False)):
                if limit is not None and emitted >= limit:
                    return
                yield self._row_to_example(row._asdict(), i)
                emitted += 1

    def _iter_from_hf(self, limit: int | None) -> Iterator[Example]:
        from datasets import load_dataset

        ds = load_dataset(
            self.HF_REPO,
            self.subset,
            split=self.split,
            cache_dir=self.cache_dir,
            streaming=False,
        )
        for i, row in enumerate(ds):
            if limit is not None and i >= limit:
                break
            yield self._row_to_example(row, i)

    def _row_to_example(self, row: dict[str, Any], idx: int) -> Example:
        sentences = self._parse_documents_sentences(row.get("documents_sentences"))

        # `documents` may be list[str], np.ndarray, or absent. Avoid truthiness
        # tests on arrays (raises on ndarray with len > 1).
        raw_documents = row.get("documents")
        documents = (
            list(raw_documents)
            if raw_documents is not None and len(raw_documents) > 0
            else self._reconstruct_documents(sentences)
        )

        scores: dict[str, float] = {}
        for field_name in _SCORE_FIELDS:
            value = row.get(field_name)
            coerced = _to_float(value)
            if coerced is not None:
                scores[field_name] = coerced

        # Use the dataset's own id when available; otherwise synthesize one.
        ex_id = str(row.get("id") or row.get("question_id") or f"{self.subset}-{idx}")

        return Example(
            id=ex_id,
            domain=self.domain,
            subset=self.subset,
            question=str(row.get("question", "")),
            documents=documents,
            documents_sentences=sentences,
            response=row.get("response"),
            all_relevant_sentence_keys=_to_str_list(row.get("all_relevant_sentence_keys")),
            all_utilized_sentence_keys=_to_str_list(row.get("all_utilized_sentence_keys")),
            reference_scores=scores,
            raw=row,
        )

    @staticmethod
    def _parse_documents_sentences(field: Any) -> list[list[Sentence]]:
        """Normalize the nested ``documents_sentences`` structure.

        Accepts list-of-list or numpy ndarrays of [key, text] pairs.
        """
        if field is None:
            return []
        out: list[list[Sentence]] = []
        for doc_idx, doc in enumerate(field):
            doc_sents: list[Sentence] = []
            # `doc` may be ndarray, list, or tuple — iterate uniformly.
            for entry in doc:
                key, text = _key_text(entry)
                if key is None:
                    continue
                doc_sents.append(Sentence(key=str(key), text=str(text), doc_index=doc_idx))
            out.append(doc_sents)
        return out

    @staticmethod
    def _reconstruct_documents(sentences: list[list[Sentence]]) -> list[str]:
        """Rebuild full document text from sentence lists."""
        return [" ".join(s.text for s in doc) for doc in sentences]


# --- module-level helpers -----------------------------------------------


def _to_float(value: Any) -> float | None:
    """Coerce ints / floats / numpy scalars / numpy bools to float."""
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        v = float(value)
        # Reject NaN — a NaN reference score is "missing", not 0.0.
        return v if v == v else None
    # numpy scalars expose .item().
    if hasattr(value, "item"):
        try:
            v = float(value.item())
            return v if v == v else None
        except (TypeError, ValueError):
            return None
    return None


def _to_str_list(value: Any) -> list[str]:
    """Coerce ndarray / list / None into a list[str]."""
    if value is None:
        return []
    return [str(x) for x in value]


def _key_text(entry: Any) -> tuple[Any, Any]:
    """Extract (key, text) from one sentence entry, regardless of container type."""
    if isinstance(entry, dict):
        return entry.get("key") or entry.get("id"), entry.get("text") or entry.get("sentence", "")
    # list / tuple / ndarray — treat as positional.
    try:
        if len(entry) >= 2:
            return entry[0], entry[1]
    except TypeError:
        pass
    return None, None
