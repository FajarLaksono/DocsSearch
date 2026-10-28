"""Pure evaluation arithmetic.

Nothing here touches Postgres or a model, so the tests run in milliseconds and
the definitions are readable in one screen.
"""

from __future__ import annotations

import statistics
from typing import Any, Iterable

from src.retrieve import Hit


def matches(
    hit: Hit, expected_document: str, expected_phrase: str | None
) -> bool:
    """A hit is the right document *and* (optionally) the right passage."""
    if hit.document != expected_document:
        return False
    if not expected_phrase:
        return True
    return expected_phrase.lower() in hit.content.lower()


def is_hit(
    hits: list[Hit],
    expected_document: str | None,
    expected_phrase: str | None,
) -> bool:
    if not expected_document:
        return False
    return any(matches(hit, expected_document, expected_phrase) for hit in hits)


def reciprocal_rank(
    hits: list[Hit],
    expected_document: str | None,
    expected_phrase: str | None,
) -> float:
    """1 / rank of the first correct chunk; 0.0 when it is absent."""
    if not expected_document:
        return 0.0
    for rank, hit in enumerate(hits, start=1):
        if matches(hit, expected_document, expected_phrase):
            return 1.0 / rank
    return 0.0


def mean(values: Iterable[float | None]) -> float | None:
    """Average the numbers that exist. All-missing returns None, not 0."""
    clean = [float(value) for value in values if value is not None]
    return round(sum(clean) / len(clean), 3) if clean else None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Turn per-question rows into the table you report."""
    answerable = [row for row in rows if row.get("answerable")]
    unanswerable = [row for row in rows if not row.get("answerable")]

    return {
        "n": len(rows),
        "n_answerable": len(answerable),
        "hit_at_k": mean(1.0 if row.get("hit") else 0.0 for row in answerable),
        "mrr": mean(row.get("reciprocal_rank") for row in answerable),
        "refusal_recall": mean(
            1.0 if row.get("refused") else 0.0 for row in unanswerable
        ),
        "false_refusal_rate": mean(
            1.0 if row.get("refused") else 0.0 for row in answerable
        ),
        "faithfulness": mean(row.get("faithfulness") for row in rows),
        "correctness": mean(row.get("correct") for row in answerable),
        "p50_latency_ms": (
            statistics.median(row["latency_ms"] for row in rows) if rows else None
        ),
    }

def threshold_curve(
    rows: list[dict[str, Any]], thresholds: list[float]
) -> list[dict[str, Any]]:
    """Gate-only refusal trade-off, computed from stored top scores.

    Assumes the answer step is skipped when top_score < t, which is exactly what
    the agent's gate does. For a single-pass row the same arithmetic is a
    counterfactual: the model may also refuse on its own.
    """
    answerable = [row for row in rows if row["answerable"]]
    unanswerable = [row for row in rows if not row["answerable"]]

    def score(row: dict[str, Any]) -> float:
        value = row.get("top_score")
        return 0.0 if value is None else float(value)

    curve: list[dict[str, Any]] = []
    for threshold in thresholds:
        false_refusal = (
            sum(1 for row in answerable if score(row) < threshold) / len(answerable)
            if answerable
            else None
        )
        caught = (
            sum(1 for row in unanswerable if score(row) < threshold) / len(unanswerable)
            if unanswerable
            else None
        )
        curve.append(
            {
                "threshold": threshold,
                "false_refusal_rate": (
                    round(false_refusal, 4) if false_refusal is not None else None
                ),
                "gate_refusal_recall": (
                    round(caught, 4) if caught is not None else None
                ),
            }
        )
    return curve
