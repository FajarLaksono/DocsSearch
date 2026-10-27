"""Ask a question over the ingested corpus from the command line."""

from __future__ import annotations

import argparse

from . import trace
from src.config import settings
from src.generate import answer_question


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question over the corpus")
    parser.add_argument("question", nargs="+")
    parser.add_argument("--k", type=int, default=None, help="override TOP_K")
    parser.add_argument("--model", default=None, help="override LLM_MODEL")
    args = parser.parse_args()

    question = " ".join(args.question)
    model = args.model or settings.llm_model

    answer = answer_question(question, k=args.k, model=model)
    run_id = trace.record(answer, llm_model=model)

    print(f"Question: {answer.question}")
    print()
    print(answer.text)
    print()

    if not answer.hits:
        print("Sources: none — the corpus returned no chunks. Run `make ingest`.")
    else:
        print(f"Sources ({len(answer.hits)} retrieved, * = cited):")
        for rank, hit in enumerate(answer.hits, start=1):
            marker = "*" if rank in answer.citations else " "
            preview = " ".join(hit.content.split())[:110]
            print(
                f" {marker}[{rank}] {hit.score:.3f}  "
                f"{hit.document}  chunk {hit.chunk_index}"
            )
            print(f"      {preview}...")

    if not answer.refused and not answer.citations:
        print("\n[warning] the answer cited no sources; treat it as ungrounded.")

    print(
        f"\nrun #{run_id}  [{answer.latency_ms} ms, "
        f"{answer.tokens_in} tokens in / {answer.tokens_out} out]"
    )


if __name__ == "__main__":
    main()
