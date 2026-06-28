"""TRACe reference validation — must match RAGBench labels on dataset responses."""

from __future__ import annotations

from rag_kag.evaluators.validate import validate_subset


def test_validate_covidqa_train_matches_reference() -> None:
    report = validate_subset("covidqa", split="train", limit=200, tolerance=1e-3)
    assert report.n == 200
    # Allow rare float edge cases; expect near-perfect on covidqa.
    for metric in (
        "context_relevance",
        "context_utilization",
        "completeness",
        "adherence",
    ):
        assert report.matches[metric] >= 195, (
            f"{metric}: {report.matches[metric]}/{report.n} mismatches"
        )
