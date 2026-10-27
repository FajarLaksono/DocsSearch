"""Run the gold set and write one `eval_results` row per question.

The loop is deliberately boring: ask, score, judge, insert, commit. Every row
carries the configuration that produced it, so a number can always be traced
back to the setup behind it.
"""

from __future__ import annotations

import time
import argparse
import json
from pathlib import Path
from typing import Any

from psycopg.types.json import Jsonb

from src import agent, db, judge, metrics, trace
from src.config import settings
from src.generate import answer_question


GOLD_PATH = Path("eval/gold.jsonl")
REQUIRED = {"id", "question", "answerable"}
RESULT_COLUMNS = (
    "question_id",
    "hit",
    "faithfulness",
    "correct",
    "mrr",
    "refused",
    "latency_ms",
    "details",
)


def load_gold(path: Path = GOLD_PATH) -> list[dict[str, Any]]:
    """Parse and shape-check the gold file. No database, no model."""
    if not path.exists():
        raise SystemExit(f"{path} is missing; write the gold set first")

    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{number}: {exc}") from exc
        missing = REQUIRED - row.keys()
        if missing:
            raise SystemExit(f"{path}:{number}: missing {sorted(missing)}")
        if row["answerable"] and not row.get("expected_document"):
            raise SystemExit(f"{path}:{number}: answerable row needs expected_document")
        rows.append(row)

    seen: set[str] = set()
    for row in rows:
        if row["id"] in seen:
            raise SystemExit(f"{path}: duplicate id {row['id']!r}")
        seen.add(row["id"])
    if not rows:
        raise SystemExit(f"{path} is empty")
    return rows


def chunks_by_document() -> dict[str, list[str]]:
    """Every chunk text, grouped by document name."""
    with db.connect() as conn:
        rows = conn.execute(
            """
            SELECT d.name, c.content
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            ORDER BY d.name, c.chunk_index
            """
        ).fetchall()

    grouped: dict[str, list[str]] = {}
    for name, content in rows:
        grouped.setdefault(name, []).append(content)
    return grouped


def normalise(text: str) -> str:
    return " ".join(text.split()).lower()


def check_gold(path: Path = GOLD_PATH) -> int:
    """Validate the gold set against the *current* index. Returns a failure count."""
    gold = load_gold(path)
    grouped = chunks_by_document()
    failures = 0

    for row in gold:
        if not row["answerable"]:
            if row.get("expected_document") or row.get("expected_phrase"):
                print(f"{row['id']}: unanswerable row should not name a document")
                failures += 1
            continue

        document = row["expected_document"]
        phrase = row.get("expected_phrase")
        if not phrase:
            print(f"{row['id']}: answerable row needs expected_phrase")
            failures += 1
            continue
        contents = grouped.get(document)
        if contents is None:
            print(f"{row['id']}: {document} is not in the index")
            failures += 1
            continue
        needle = normalise(phrase)
        if not any(needle in normalise(text) for text in contents):
            print(f"{row['id']}: phrase not found in {document}: {phrase!r}")
            failures += 1

    print(f"{len(gold)} rows checked, {failures} problem(s)")
    return failures


def config_snapshot(
    *,
    model: str | None = None,
    k: int | None = None,
    use_judge: bool = True,
    use_agent: bool = False,
    max_steps: int | None = None,
    thinking: str | None = None,
) -> dict[str, Any]:
    """The settings that make this row's numbers reproducible."""
    return {
        "mode": "agent" if use_agent else "single",
        "llm_model": model or settings.llm_model,
        "judge_model": settings.judge_model if use_judge else None,
        "embed_model": settings.embed_model,
        "chunk_tokens": settings.chunk_tokens,
        "chunk_overlap": settings.chunk_overlap,
        "top_k": k or settings.top_k,
        "min_top_score": settings.min_top_score,
        "num_ctx": settings.num_ctx,
        "max_steps": (max_steps or settings.max_steps) if use_agent else None,
        "thinking": (thinking or settings.thinking) if use_agent else None,
    }



def insert_result(row: dict[str, Any]) -> None:
    """One transaction per question: a crash costs the unfinished question only."""
    with db.connect() as conn:
        conn.execute(
            """
            INSERT INTO eval_results
                (eval_label, question_id, hit, faithfulness, correct,
                 run_id, mrr, refused, latency_ms, details)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                row["eval_label"],
                row["question_id"],
                row["hit"],
                row["faithfulness"],
                row["correct"],
                row["run_id"],
                row["mrr"],
                row["refused"],
                row["latency_ms"],
                Jsonb(row["details"]),
            ),
        )


def save_row(
    *,
    label: str,
    item: dict[str, Any],
    answer: Any,
    run_id: int,
    hit: bool | None,
    reciprocal_rank: float | None,
    verdict: judge.JudgeResult,
    model: str | None,
    k: int | None,
    use_judge: bool,
    use_agent: bool = False,
    agent_metrics: dict[str, Any] | None = None,
) -> None:
    details = {
        "answerable": bool(item["answerable"]),
        "question": item["question"],
        "expected_document": item.get("expected_document"),
        "expected_phrase": item.get("expected_phrase"),
        "gold_answer": item.get("gold_answer"),
        "answer": answer.text,
        "citations": answer.citations,
        "top_score": answer.hits[0].score if answer.hits else None,
        "judge_claims": verdict.claims,
        "judge_error": verdict.error,
        "agent": agent_metrics,
        "config": config_snapshot(
            model=model,
            k=k,
            use_judge=use_judge,
            use_agent=use_agent,
            max_steps=(agent_metrics or {}).get("max_steps"),
            thinking=(agent_metrics or {}).get("thinking"),
        ),
    }
    insert_result(
        {
            "eval_label": label,
            "question_id": item["id"],
            "hit": hit,
            "faithfulness": verdict.faithfulness,
            "correct": verdict.correct,
            "run_id": run_id,
            "mrr": reciprocal_rank,
            "refused": answer.refused,
            "latency_ms": answer.latency_ms,
            "details": details,
        }
    )


def labels() -> list[dict[str, Any]]:
    """Every configuration that has been run, newest first."""
    with db.connect() as conn:
        rows = conn.execute(
            """
            SELECT eval_label, count(*) AS n, max(created_at) AS last_run
            FROM eval_results
            GROUP BY eval_label
            ORDER BY max(created_at) DESC NULLS LAST
            """
        ).fetchall()
    return [
        {"label": label, "n": int(n), "last_run": last_run}
        for label, n, last_run in rows
    ]


def load_results(label: str) -> list[dict[str, Any]]:
    with db.connect() as conn:
        rows = conn.execute(
            """
            SELECT question_id, hit, faithfulness, correct, mrr, refused,
                   latency_ms, details
            FROM eval_results
            WHERE eval_label = %s
            ORDER BY question_id
            """,
            (label,),
        ).fetchall()
    return [dict(zip(RESULT_COLUMNS, row)) for row in rows]


def metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Adapt stored rows to the shape `metrics.summarize` expects."""
    return [
        {
            "answerable": bool((row["details"] or {}).get("answerable", True)),
            "hit": row["hit"],
            "reciprocal_rank": row["mrr"],
            "refused": row["refused"],
            "faithfulness": row["faithfulness"],
            "correct": row["correct"],
            "latency_ms": row["latency_ms"] or 0,
        }
        for row in rows
    ]


def summarize_label(label: str) -> dict[str, Any]:
    return metrics.summarize(metric_rows(load_results(label)))


def run(
    *,
    label: str,
    limit: int | None = None,
    use_judge: bool = True,
    model: str | None = None,
    k: int | None = None,
    resume: bool = False,
    use_agent: bool = False,
    max_steps: int | None = None,
    thinking: str | None = None,
) -> dict[str, Any]:
    gold = load_gold()
    if limit:
        gold = gold[:limit]

    existing = {row["question_id"] for row in load_results(label)}

    if existing and not resume:
        with db.connect() as conn:
            conn.execute("DELETE FROM eval_results WHERE eval_label = %s", (label,))
        print(f"replaced {len(existing)} existing rows for label {label!r}")
        existing = set()
    elif existing:
        print(f"resuming {label!r}: {len(existing)} rows already done")

    total = len(gold)
    for index, item in enumerate(gold, start=1):
        if item["id"] in existing:
            continue

        agent_metrics: dict[str, Any] | None = None
        if use_agent:
            result = agent.run_agent(
                item["question"],
                model=model,
                k=k,
                max_steps=max_steps,
                thinking=thinking,
            )
            answer = result.answer
            run_id = trace.record(
                answer,
                llm_model=model or settings.llm_model,
                steps=result.steps,
                metrics=result.metrics,
            )
            agent_metrics = {
                key: result.metrics.get(key)
                for key in (
                    "reason",
                    "stop_reason",
                    "steps",
                    "searches",
                    "llm_calls",
                    "thinking",
                    "max_steps",
                )
            }
        else:
            answer = answer_question(item["question"], k=k, model=model)
            run_id = trace.record(answer, llm_model=model or settings.llm_model)

        answerable = bool(item["answerable"])
        hit = (
            metrics.is_hit(
                answer.hits,
                item.get("expected_document"),
                item.get("expected_phrase"),
            )
            if answerable
            else None
        )
        reciprocal_rank = (
            metrics.reciprocal_rank(
                answer.hits,
                item.get("expected_document"),
                item.get("expected_phrase"),
            )
            if answerable
            else None
        )

        verdict = judge.JudgeResult(faithfulness=None, correct=None)
        if use_judge:
            # Skip judging correctly refused unanswerable questions to save load
            if not answerable and answer.refused:
                verdict = judge.JudgeResult(faithfulness=None, correct=None)
            else:
                verdict = judge.judge_answer(
                    item["question"],
                    answer.text,
                    judge.format_context(answer.hits),
                    gold_answer=item.get("gold_answer"),
                )

        save_row(
            label=label,
            item=item,
            answer=answer,
            run_id=run_id,
            hit=hit,
            reciprocal_rank=reciprocal_rank,
            verdict=verdict,
            model=model,
            k=k,
            use_judge=use_judge,
            use_agent=use_agent,
            agent_metrics=agent_metrics,
        )
        time.sleep(2)

        faith = "  n/a" if verdict.faithfulness is None else f"{verdict.faithfulness:.3f}"
        rank = "  n/a" if reciprocal_rank is None else f"{reciprocal_rank:5.3f}"
        tag = f"  {agent_metrics['reason']:<20}" if agent_metrics else ""
        print(
            f"[{index:02}/{total}] {item['id']}  hit={str(hit):5}  rr={rank}  "
            f"faith={faith}{tag}  {answer.latency_ms:5} ms"
        )

    return summarize_label(label)


def agent_stats(label: str) -> dict[str, Any]:
    """Averages over the loop itself: searches, decisions, and why it stopped."""
    agents = [
        (row["details"] or {}).get("agent")
        for row in load_results(label)
    ]
    agents = [entry for entry in agents if entry]
    if not agents:
        return {}

    n = len(agents)

    def mean(key: str) -> float:
        return round(sum(float(entry.get(key) or 0) for entry in agents) / n, 3)

    def rate(key: str, value: str) -> float:
        return round(sum(1 for entry in agents if entry.get(key) == value) / n, 3)

    return {
        "n": n,
        "mean_searches": mean("searches"),
        "mean_decisions": mean("steps"),
        "mean_calls": mean("llm_calls"),
        "step_limit_rate": rate("stop_reason", "step_limit"),
        "invalid_json_rate": rate("stop_reason", "invalid_json"),
        "below_threshold_rate": rate("reason", "below_min_top_score"),
    }


def print_summary(label: str, summary: dict[str, Any]) -> None:
    def show(value: Any, digits: int = 3) -> str:
        return "n/a" if value is None else f"{float(value):.{digits}f}"

    print()
    print(f"label: {label}")
    print(
        f"n={summary['n']}  ({summary['n_answerable']} answerable, "
        f"{summary['n'] - summary['n_answerable']} unanswerable)"
    )
    print()
    print(f"{'hit@k':22} {show(summary['hit_at_k'])}")
    print(f"{'MRR':22} {show(summary['mrr'])}")
    print(f"{'refusal recall':22} {show(summary['refusal_recall'])}")
    print(f"{'false refusal rate':22} {show(summary['false_refusal_rate'])}")
    print(f"{'faithfulness':22} {show(summary['faithfulness'])}")
    print(f"{'correctness':22} {show(summary['correctness'])}")
    print(f"{'p50 latency (ms)':22} {show(summary['p50_latency_ms'], 0)}")

    loop = agent_stats(label)
    if loop:
        print()
        print("loop")
        print(f"{'mean searches':22} {loop['mean_searches']:.2f}")
        print(f"{'mean decisions':22} {loop['mean_decisions']:.2f}")
        print(f"{'mean model calls':22} {loop['mean_calls']:.2f}")
        print(f"{'hit step limit':22} {loop['step_limit_rate'] * 100:.1f}%")
        print(f"{'invalid JSON':22} {loop['invalid_json_rate'] * 100:.1f}%")
        print(f"{'below MIN_TOP_SCORE':22} {loop['below_threshold_rate'] * 100:.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the gold set over the corpus")
    parser.add_argument("--check-gold", action="store_true", help="validate only")
    parser.add_argument("--limit", type=int, default=None, help="first N questions")
    parser.add_argument("--label", default="baseline", help="name this configuration")
    parser.add_argument("--no-judge", action="store_true", help="skip the judge")
    parser.add_argument("--resume", action="store_true", help="skip finished ids")
    parser.add_argument("--model", default=None, help="override LLM_MODEL")
    parser.add_argument("--k", type=int, default=None, help="override TOP_K")
    parser.add_argument("--agent", action="store_true", help="use the agent loop")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--thinking", choices=("off", "prompted"), default=None)
    args = parser.parse_args()

    if args.check_gold:
        raise SystemExit(1 if check_gold() else 0)

    summary = run(
        label=args.label,
        limit=args.limit,
        use_judge=not args.no_judge,
        model=args.model,
        k=args.k,
        resume=args.resume,
    )
    print_summary(args.label, summary)


if __name__ == "__main__":
    main()
