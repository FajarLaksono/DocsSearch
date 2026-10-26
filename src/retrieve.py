"""Top-k retrieval over the chunks table."""

from __future__ import annotations

import sys
from dataclasses import dataclass

from pgvector import Vector

from src import db, llm
from src.config import settings

SEARCH_SQL = """
SELECT c.id,
       d.name,
       c.chunk_index,
       c.content,
       1 - (c.embedding <=> %s) AS score
FROM chunks c
JOIN documents d ON d.id = c.document_id
WHERE c.embed_model = %s
ORDER BY c.embedding <=> %s
LIMIT %s
"""


@dataclass(frozen=True)
class Hit:
    chunk_id: int
    document: str
    chunk_index: int
    content: str
    score: float


def search(query: str, k: int | None = None, embed_model: str | None = None) -> list[Hit]:
    limit = settings.top_k if k is None else k
    model = embed_model or settings.embed_model
    vector = Vector(llm.get_client().embed([query], model=model)[0])

    with db.connect() as conn:
        rows = conn.execute(SEARCH_SQL, (vector, model, vector, limit)).fetchall()

    return [Hit(*row) for row in rows]


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit('usage: python -m src.retrieve "your question"')
    query = " ".join(sys.argv[1:])
    hits = search(query)

    print(f"Query: {query}")
    if not hits:
        sys.exit("No chunks retrieved. Run `make ingest` first.")

    for rank, hit in enumerate(hits, start=1):
        preview = " ".join(hit.content.split())[:120]
        print(f"[{rank}] {hit.score:.3f}  {hit.document}  chunk {hit.chunk_index}")
        print(f"    {preview}...")


if __name__ == "__main__":
    main()
