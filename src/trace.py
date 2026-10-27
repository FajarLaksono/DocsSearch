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


def _default_steps(answer: Answer) -> list[dict[str, Any]]:
    return [
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
        }
    ]


def record(
    answer: Answer,
    *,
    llm_model: str,
    embed_model: str | None = None,
    steps: list[dict[str, Any]] | None = None,
    metrics: dict[str, Any] | None = None,
) -> int:
    """Write one run and return its id.

    `steps` lets the agent supply its thoughts, tool calls, and observations.
    The answer step is always appended here, from the `Answer` object, so the
    stored text and citations cannot drift from what was returned.

    `metrics` lets the agent add fields such as `reason` and `searches`. When it
    is None the Phase 2 arithmetic is used.
    """
    steps = list(steps) if steps is not None else _default_steps(answer)
    steps.append(
        {
            "type": "answer",
            "text": answer.text,
            "citations": answer.citations,
            "refused": answer.refused,
            # The exact rank order the model saw. Without this, a reload has to
            # recompute order, and a score tie or a filtered chunk breaks links.
            "chunks": [hit.chunk_id for hit in answer.hits],
        }
    )

    if metrics is None:
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


def _chunks_from_steps(steps: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Best copy of every chunk mentioned by a retrieval or observation step."""
    found: dict[int, dict[str, Any]] = {}

    for step in steps:
        if step.get("type") not in ("retrieval", "observation"):
            continue
        for chunk in step.get("chunks", []):
            chunk_id = int(chunk["chunk_id"])
            previous = found.get(chunk_id)
            if previous is None or float(chunk["score"]) > float(previous["score"]):
                found[chunk_id] = chunk

    return found


def answer_from_run(run: dict[str, Any]) -> Answer:
    """Rebuild the Answer object from stored steps, joining chunk text by id."""
    steps = run.get("steps") or []
    answer_step = next((s for s in steps if s.get("type") == "answer"), {})
    chunks_by_id = _chunks_from_steps(steps)

    order = [int(chunk_id) for chunk_id in answer_step.get("chunks", [])]
    if order:
        chunks = [chunks_by_id[cid] for cid in order if cid in chunks_by_id]
    else:
        chunks = sorted(
            chunks_by_id.values(),
            key=lambda chunk: (-float(chunk["score"]), int(chunk["chunk_id"])),
        )

    contents = _chunk_contents([int(chunk["chunk_id"]) for chunk in chunks])

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


def format_steps(steps: list[dict[str, Any]]) -> list[str]:
    """Render stored steps. Reads JSONB only: no agent import, no model."""
    lines: list[str] = []
    for step in steps or []:
        kind = step.get("type")
        if kind == "thought":
            lines.append(f"  step {step.get('step')}  thought: {step.get('text', '')}")
        elif kind == "tool_call":
            name = step.get("name", "?")
            if name == "answer":
                lines.append(f"  step {step.get('step')}  action: answer")
            else:
                lines.append(
                    f"  step {step.get('step')}  action: {name}({step.get('input', '')!r})"
                )
        elif step.get("error"):
            lines.append(f"           protocol: {step['error']}")
        elif kind == "observation":
            lines.append(f"           result: {step.get('summary', '')}")
    return lines



def main() -> None:
    run = load_run(int(sys.argv[1])) if len(sys.argv) > 1 else load_latest()
    answer = answer_from_run(run)

    print(
        f"run #{run['id']}  {run['created_at']:%Y-%m-%d %H:%M:%S}  "
        f"{run['llm_model']}  {run['latency_ms']} ms"
    )
    print(f"Q: {run['question']}")
    if run.get("steps"):
        print("steps:")
        for line in format_steps(run["steps"]):
            print(line)
    print(f"A: {answer.text}")
    print(f"metrics: {run['metrics']}")
    print("sources:")
    for rank, hit in enumerate(answer.hits, start=1):
        mark = "*" if rank in answer.citations else " "
        print(f" {mark}[{rank}] {hit.score:.3f}  {hit.document}  chunk {hit.chunk_index}")


if __name__ == "__main__":
    main()
