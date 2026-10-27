"""The tools the agent may call. One tool, one job.

`search_docs` wraps `retrieve.search` so the loop never builds SQL or touches an
embedding client. `format_observation` renders a result for both the model and
the human reading the trace.

The preview is deliberately short. The planner only needs enough text to decide
whether a chunk is relevant; the full text goes to `answer_from_hits` at the end,
where it is actually cited.
"""

from __future__ import annotations

from src import retrieve
from src.retrieve import Hit

PREVIEW_CHARS = 300


def search_docs(query: str, *, k: int | None = None) -> list[Hit]:
    """Retrieve chunks for one query. An empty query returns nothing."""
    query = " ".join(query.split())
    if not query:
        return []
    return retrieve.search(query, k=k)


def preview(text: str, limit: int = PREVIEW_CHARS) -> str:
    """One line, cut at a word boundary, with an ellipsis when shortened."""
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    cut = flat[:limit].rsplit(" ", 1)[0].rstrip(" ,;:")
    return f"{cut}..."


def format_observation(
    query: str, hits: list[Hit], changes: dict[int, str]
) -> str:
    """What the model is told after a search.

    `hits` is the merged, sorted, capped list the model will see next, not the
    raw result. `changes` maps chunk id to "new" or "improved". The distinction
    matters: a search can return four chunks and improve nothing, or return one
    chunk already present with a much better score.
    """
    if not hits:
        return f'Search "{query}" returned no chunks. Try different words.'

    if not changes:
        return (
            f'Search "{query}" found nothing better than the sources you have. '
            "Do not repeat this query. Choose answer, or search with different words."
        )

    lines = [f'Search "{query}" updated {len(changes)} source(s).']
    for rank, hit in enumerate(hits, start=1):
        label = changes.get(hit.chunk_id)
        if label is None:
            continue
        lines.append(
            f"[{rank}] {hit.score:.3f}  {hit.document}  chunk {hit.chunk_index} "
            f"({label})"
        )
        lines.append(f"    {preview(hit.content)}")
    return "\n".join(lines)
