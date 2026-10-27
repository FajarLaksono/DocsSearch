"""Persist and reload runs. The trace is a row, not a log file."""

from __future__ import annotations

import sys
from typing import Any

from psycopg.types.json import Jsonb

from src import db
from src.config import settings
from src.generate import Answer
from src.retrieve import Hit

RUN_COLUMNS = (
    "id",
    "created_at",
    "question",
    "answer",
    "llm_model",
    "embed_model",
    "steps",
    "metrics",
    "latency_ms",
    "tokens_in",
    "tokens_out",
)

INSERT_SQL = """
INSERT INTO runs
  (question, answer, llm_model, embed_model, steps, metrics,
   latency_ms, tokens_in, tokens_out)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
RETURNING id
"""


def compute_metrics(
    scores: list[float], citations: list[int], refused: bool
) -> dict[str, Any]:
    """Derive the numbers a reader needs to judge this retrieval.

    Pure function: no database, no model. Rounded so two runs compare cleanly.
    """
    return {
        "top_score": round(scores[0], 6) if scores else None,
        "mean_score": round(sum(scores) / len(scores), 6) if scores else None,
        "score_gap": round(scores[0] - scores[1], 6) if len(scores) > 1 else None,
        "cited_sources": len(citations),
        "refused": refused,
    }


def record(answer: Answer, *, llm_model: str, embed_model: str | None = None) -> int:
    """Write one run and return its id."""
    steps = [
        {
            "type": "retrieval",
            "query": answer.question,
            "chunks": [
                {
                    "chunk_id": hit.chunk_id,
                    "document": hit.document,
                    "chunk_index": hit.chunk_index,
                    "score": round(hit.score, 6),
                }
                for hit in answer.hits
            ],
        },
        {
            "type": "answer",
            "text": answer.text,
            "citations": answer.citations,
            "refused": answer.refused,
        },
    ]
    metrics = compute_metrics(
        [hit.score for hit in answer.hits], answer.citations, answer.refused
    )

    with db.connect() as conn:
        row = conn.execute(
            INSERT_SQL,
            (
                answer.question,
                answer.text,
                llm_model,
                embed_model or settings.embed_model,
                Jsonb(steps),
                Jsonb(metrics),
                answer.latency_ms,
                answer.tokens_in,
                answer.tokens_out,
            ),
        ).fetchone()
    return int(row[0])


def _run_from_row(row: tuple) -> dict[str, Any]:
    return dict(zip(RUN_COLUMNS, row))


def load_run(run_id: int) -> dict[str, Any]:
    with db.connect() as conn:
        row = conn.execute(
            f"SELECT {', '.join(RUN_COLUMNS)} FROM runs WHERE id = %s", (run_id,)
        ).fetchone()
    if row is None:
        raise SystemExit(f"run {run_id} not found")
    return _run_from_row(row)


def load_latest() -> dict[str, Any]:
    with db.connect() as conn:
        row = conn.execute(
            f"SELECT {', '.join(RUN_COLUMNS)} FROM runs ORDER BY id DESC LIMIT 1"
        ).fetchone()
    if row is None:
        raise SystemExit("no runs yet - ask a question first")
    return _run_from_row(row)


def recent_runs(limit: int = 20) -> list[dict[str, Any]]:
    with db.connect() as conn:
        rows = conn.execute(
            """
            SELECT id, created_at, question, llm_model, latency_ms
            FROM runs
            ORDER BY id DESC
            LIMIT %s
            """,
            (limit,),
        ).fetchall()
    keys = ("id", "created_at", "question", "llm_model", "latency_ms")
    return [dict(zip(keys, row)) for row in rows]


def _chunk_contents(chunk_ids: list[int]) -> dict[int, str]:
    if not chunk_ids:
        return {}
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT id, content FROM chunks WHERE id = ANY(%s)", (chunk_ids,)
        ).fetchall()
    return {int(chunk_id): content for chunk_id, content in rows}


def answer_from_run(run: dict[str, Any]) -> Answer:
    """Rebuild the Answer object from stored steps, joining chunk text by id."""
    steps = run.get("steps") or []
    retrieval = next((s for s in steps if s.get("type") == "retrieval"), {})
    answer_step = next((s for s in steps if s.get("type") == "answer"), {})
    chunks = retrieval.get("chunks", [])
    contents = _chunk_contents([int(c["chunk_id"]) for c in chunks])

    hits = [
        Hit(
            chunk_id=int(chunk["chunk_id"]),
            document=chunk["document"],
            chunk_index=int(chunk["chunk_index"]),
            content=contents.get(
                int(chunk["chunk_id"]), "[chunk is no longer in the index]"
            ),
            score=float(chunk["score"]),
        )
        for chunk in chunks
    ]

    return Answer(
        question=run["question"],
        text=run.get("answer") or "",
        hits=hits,
        citations=list(answer_step.get("citations", [])),
        refused=bool(answer_step.get("refused", False)),
        latency_ms=int(run.get("latency_ms") or 0),
        tokens_in=int(run.get("tokens_in") or 0),
        tokens_out=int(run.get("tokens_out") or 0),
    )


def main() -> None:
    run = load_run(int(sys.argv[1])) if len(sys.argv) > 1 else load_latest()
    answer = answer_from_run(run)

    print(
        f"run #{run['id']}  {run['created_at']:%Y-%m-%d %H:%M:%S}  "
        f"{run['llm_model']}  {run['latency_ms']} ms"
    )
    print(f"Q: {run['question']}")
    print(f"A: {answer.text}")
    print(f"metrics: {run['metrics']}")
    print("sources:")
    for rank, hit in enumerate(answer.hits, start=1):
        mark = "*" if rank in answer.citations else " "
        print(f" {mark}[{rank}] {hit.score:.3f}  {hit.document}  chunk {hit.chunk_index}")


if __name__ == "__main__":
    main()
