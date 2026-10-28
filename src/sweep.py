"""Run the experiment matrix, one label per configuration.

Two kinds of experiment live here:

  in-place    k, thinking, model - a flag to `evaluate.run`
  re-ingest   chunk size - needs a fresh index, so it shells out to `src.ingest`

The chunk-size sweep is destructive: ids are reused, so runs recorded before it
stop pointing at the same text. It ends by re-ingesting the configured default.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

from src import evaluate, explain, judge
from src.config import settings

NORAG_PATH = Path("reports/norag.json")


def run_ingest(*, chunk_tokens: int, chunk_overlap: int) -> None:
    """Re-ingest with a different chunk budget. Settings are read at import."""
    env = {
        **os.environ,
        "CHUNK_TOKENS": str(chunk_tokens),
        "CHUNK_OVERLAP": str(chunk_overlap),
    }
    print(f"  re-ingesting: CHUNK_TOKENS={chunk_tokens} CHUNK_OVERLAP={chunk_overlap}")
    subprocess.run([sys.executable, "-m", "src.ingest", "--reset"], env=env, check=True)


def chunk_plan(sizes: list[int]) -> list[tuple[str, int, int]]:
    """Label, tokens, overlap. Overlap scales, so the ratio stays constant."""
    return [(f"chunk{size}", size, max(0, size // 6)) for size in sizes]


def sweep_chunks(sizes: list[int], *, use_judge: bool, restore: bool = True) -> None:
    for label, tokens, overlap in chunk_plan(sizes):
        print(f"[{label}]")
        run_ingest(chunk_tokens=tokens, chunk_overlap=overlap)
        summary = evaluate.run(label=label, use_judge=use_judge)
        evaluate.print_summary(label, summary)

    if restore:
        print("[restore]")
        run_ingest(
            chunk_tokens=settings.chunk_tokens,
            chunk_overlap=settings.chunk_overlap,
        )


def sweep_k(values: list[int], *, use_judge: bool) -> None:
    for value in values:
        label = f"k{value}"
        print(f"[{label}]")
        summary = evaluate.run(label=label, k=value, use_judge=use_judge)
        evaluate.print_summary(label, summary)


def sweep_thinking(modes: list[str]) -> None:
    for mode in modes:
        label = "agent" if mode == "prompted" else f"agent-{mode}"
        print(f"[{label}]")
        summary = evaluate.run(label=label, use_agent=True, thinking=mode)
        evaluate.print_summary(label, summary)


def sweep_models(models: list[str], *, use_judge: bool) -> None:
    for model in models:
        label = f"model-{model.replace(':', '-')}"
        print(f"[{label}]")
        summary = evaluate.run(label=label, model=model, use_judge=use_judge)
        evaluate.print_summary(label, summary)


def sweep_norag(*, limit: int | None = None) -> None:
    """Answer every gold question with no sources, and judge correctness only."""
    gold = evaluate.load_gold()
    if limit:
        gold = gold[:limit]

    rows: list[dict[str, Any]] = []
    for index, item in enumerate(gold, start=1):
        plain = explain.without_rag(item["question"])
        verdict = judge.judge_answer(
            item["question"],
            plain["text"],
            "(no sources were provided)",
            gold_answer=item.get("gold_answer"),
        )
        rows.append(
            {
                "id": item["id"],
                "answerable": bool(item["answerable"]),
                "question": item["question"],
                "answer": plain["text"],
                "correct": verdict.correct,
                "latency_ms": plain["latency_ms"],
                "tokens_in": plain["tokens_in"],
                "tokens_out": plain["tokens_out"],
                "judge_error": verdict.error,
            }
        )
        print(f"[{index:02}/{len(gold)}] {item['id']}  correct={verdict.correct}")

    NORAG_PATH.parent.mkdir(parents=True, exist_ok=True)
    NORAG_PATH.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")

    answerable = [row for row in rows if row["answerable"]]
    unanswerable = [row for row in rows if not row["answerable"]]
    judged = [row for row in answerable if row["correct"] is not None]
    print()
    print(f"no-RAG correctness      {sum(1 for r in judged if r['correct']) / len(judged):.3f}")
    print(f"answered unanswerable   {len(unanswerable)}/{len(unanswerable)}")
    print(f"p50 latency (ms)        {statistics.median([r['latency_ms'] for r in rows]):.0f}")
    print(f"wrote {NORAG_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the experiment matrix")
    parser.add_argument(
        "--only",
        default="chunk,k,thinking,model,norag",
        help="comma list of: chunk,k,thinking,model,norag",
    )
    parser.add_argument("--chunks", default="200,300,400")
    parser.add_argument("--ks", default="2,4,8")
    parser.add_argument("--thinking", default="off,prompted")
    parser.add_argument("--models", default="llama3.2:1b,llama3.2:3b")
    parser.add_argument("--no-judge", action="store_true", help="retrieval metrics only")
    parser.add_argument("--no-restore", action="store_true", help="leave the last index in place")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    wanted = [part.strip() for part in args.only.split(",") if part.strip()]
    plan: list[str] = []
    if "chunk" in wanted:
        plan += [label for label, _, _ in chunk_plan(_ints(args.chunks))]
    if "k" in wanted:
        plan += [f"k{value}" for value in _ints(args.ks)]
    if "thinking" in wanted:
        plan += ["agent" if mode == "prompted" else f"agent-{mode}" for mode in args.thinking.split(",")]
    if "model" in wanted:
        plan += [f"model-{model.replace(':', '-')}" for model in args.models.split(",")]
    if "norag" in wanted:
        plan += [f"norag -> {NORAG_PATH}"]

    print("plan:")
    for label in plan:
        print(f"  {label}")

    if args.dry_run:
        return

    judge_on = not args.no_judge
    if "chunk" in wanted:
        sweep_chunks(_ints(args.chunks), use_judge=judge_on, restore=not args.no_restore)
    if "k" in wanted:
        sweep_k(_ints(args.ks), use_judge=judge_on)
    if "thinking" in wanted:
        sweep_thinking(args.thinking.split(","))
    if "model" in wanted:
        sweep_models(args.models.split(","), use_judge=judge_on)
    if "norag" in wanted:
        sweep_norag()

    print()
    print("done. run `make report` to write reports/results.md")


def _ints(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part.strip()]


if __name__ == "__main__":
    main()
