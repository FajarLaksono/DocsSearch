"""Load documents, chunk them, embed them, and store them in Postgres."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from pgvector import Vector

from src import db, llm
from src.chunking import chunk_text, token_target_to_words
from src.config import settings

DATA_DIR = Path("data")
SUFFIXES = {".md", ".markdown", ".txt"}
EMBED_BATCH = 32

def find_documents(data_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(data_dir.rglob("*"))
        if path.is_file()
        and path.suffix.lower() in SUFFIXES
        and not any(part.startswith(".") for part in path.relative_to(data_dir).parts)
    ]


def ingest_file(
    conn,
    path: Path,
    data_dir: Path,
    client,
    max_words: int,
    overlap_words: int,
    embed_model: str,
) -> int:
    name = path.relative_to(data_dir).as_posix()
    text = path.read_text(encoding="utf-8", errors="replace")
    pieces = chunk_text(text, max_words=max_words, overlap_words=overlap_words)
    if not pieces:
        print(f"[ingest] {name}: empty, skipped")
        return 0

    document_id = conn.execute(
        """
        INSERT INTO documents (name, source_path)
        VALUES (%s, %s)
        ON CONFLICT (name) DO UPDATE SET source_path = EXCLUDED.source_path
        RETURNING id
        """,
        (name, str(path)),
    ).fetchone()[0]

    conn.execute(
        "DELETE FROM chunks WHERE document_id = %s AND embed_model = %s",
        (document_id, embed_model),
    )

    rows = []
    for start in range(0, len(pieces), EMBED_BATCH):
        batch = pieces[start : start + EMBED_BATCH]
        vectors = client.embed(batch, model=embed_model)
        if len(vectors) != len(batch):
            raise RuntimeError(
                f"embedding returned {len(vectors)} vectors for {len(batch)} chunks"
            )
        for offset, (piece, vector) in enumerate(zip(batch, vectors)):
            if len(vector) != settings.embed_dim:
                raise RuntimeError(
                    f"{embed_model} returned {len(vector)}-d vectors but "
                    f"EMBED_DIM={settings.embed_dim}. Fix .env and db/init.sql."
                )
            rows.append(
                (document_id, start + offset, piece, embed_model, Vector(vector))
            )

    with conn.cursor() as cursor:
        cursor.executemany(
            """
            INSERT INTO chunks (document_id, chunk_index, content, embed_model, embedding)
            VALUES (%s, %s, %s, %s, %s)
            """,
            rows,
        )
    print(f"[ingest] {name} -> {len(rows)} chunks")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into Postgres")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="delete all documents and chunks before ingesting",
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    if not args.data_dir.is_dir():
        sys.exit(f"[ingest] data directory not found: {args.data_dir.resolve()}")

    files = find_documents(args.data_dir)
    if not files:
        sys.exit(f"[ingest] no .md/.txt files under {args.data_dir.resolve()}")

    max_words = token_target_to_words(settings.chunk_tokens)
    overlap_words = token_target_to_words(settings.chunk_overlap)
    client = llm.get_client()

    print(f"[ingest] data dir: {args.data_dir.resolve()}")
    print(
        f"[ingest] embed {settings.embed_model} ({settings.embed_dim}-d), "
        f"chunk ~{settings.chunk_tokens} tokens ({max_words} words), "
        f"overlap {overlap_words} words"
    )

    db.wait_for_db()
    started = time.perf_counter()
    total_chunks = 0

    with db.connect() as conn:
        if args.reset:
            conn.execute("TRUNCATE chunks, documents RESTART IDENTITY CASCADE")
            print("[ingest] --reset: cleared documents and chunks")

        for path in files:
            total_chunks += ingest_file(
                conn,
                path,
                args.data_dir,
                client,
                max_words,
                overlap_words,
                settings.embed_model,
            )
            conn.commit()  # one bad file must not roll back the good ones

    elapsed = time.perf_counter() - started
    print(
        f"[ingest] done: {len(files)} documents, {total_chunks} chunks "
        f"in {elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()