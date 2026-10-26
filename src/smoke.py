from __future__ import annotations

from src import db, llm
from src.config import settings

def main() -> None:
    db.wait_for_db()
    with db.connect() as conn:
        documents, chunks, runs = conn.execute(
            "SELECT (SELECT count(*) FROM documents),"
            "       (SELECT count(*) FROM chunks),"
            "       (SELECT count(*) FROM runs)"
        ).fetchone()
    print(f"[smoke] db ok — documents={documents} chunks={chunks} runs={runs}")
    client = llm.get_client()
    chat = client.generate(
        [{"role": "user", "content": "Reply with exactly: smoke ok"}],
        temperature=0.0,
    )
    print(
        f"[smoke] chat {settings.llm_model} -> {chat['text'].strip()!r} "
        f"({chat['tokens_out']} tokens out)"
    )

    vectors = client.embed(["a short smoke test"])
    dimension = len(vectors[0])
    print(f"[smoke] embed {settings.embed_model} -> {dimension}-d")
    if dimension != settings.embed_dim:
        raise SystemExit(
            f"[smoke] EMBED_DIM={settings.embed_dim} but the model returned "
            f"{dimension}. Fix .env, then run `make reset-db` if the schema "
            f"dimension also has to change."
        )

    print("[smoke] all good")


if __name__ == "__main__":
    main()