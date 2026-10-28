"""Turn `eval_results` into the table the README quotes.

Reads only. Writes one markdown file. Every row in it can be traced back to a
label and a `config` snapshot.
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path
from typing import Any

from src import db, evaluate, metrics

DEFAULT_OUT = Path("reports/results.md")
CURVE = [0.25, 0.35, 0.45, 0.55]


def label_rows(label: str) -> dict[str, Any]:
    rows = evaluate.load_results(label)
    summary = metrics.summarize(evaluate.metric_rows(rows))
    config = (rows[0]["details"] or {}).get("config", {}) if rows else {}
    return {"label": label, "rows": rows, "summary": summary, "config": config}


def token_means(label: str) -> tuple[float, float]:
    with db.connect() as conn:
        row = conn.execute(
            """
            SELECT avg(r.tokens_in), avg(r.tokens_out)
            FROM eval_results e
            JOIN runs r ON r.id = e.run_id
            WHERE e.eval_label = %s
            """,
            (label,),
        ).fetchone()
    return (float(row[0] or 0), float(row[1] or 0))


def cell(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def table(entries: list[dict[str, Any]]) -> str:
    header = (
        "| label | mode | chunk | k | model | n | hit@k | MRR | refusal recall "
        "| false refusal | faithfulness | correctness | p50 ms | mean tokens in/out |"
    )
    rule = "|" + "---|" * 14
    lines = [header, rule]
    for entry in entries:
        summary = entry["summary"]
        config = entry["config"]
        tin, tout = token_means(entry["label"])
        lines.append(
            "| {label} | {mode} | {chunk}+{overlap} | {k} | {model} | {n} "
            "| {hit} | {mrr} | {recall} | {false} | {faith} | {correct} "
            "| {p50} | {tin:.0f}/{tout:.0f} |".format(
                label=entry["label"],
                mode=config.get("mode", "single"),
                chunk=config.get("chunk_tokens", "?"),
                overlap=config.get("chunk_overlap", "?"),
                k=config.get("top_k", "?"),
                model=config.get("llm_model", "?"),
                n=summary["n"],
                hit=cell(summary["hit_at_k"]),
                mrr=cell(summary["mrr"]),
                recall=cell(summary["refusal_recall"]),
                false=cell(summary["false_refusal_rate"]),
                faith=cell(summary["faithfulness"]),
                correct=cell(summary["correctness"]),
                p50=cell(summary["p50_latency_ms"], 0),
                tin=tin,
                tout=tout,
            )
        )
    return "\n".join(lines)


def deltas(entries: list[dict[str, Any]], baseline: str) -> str:
    base = next((entry for entry in entries if entry["label"] == baseline), None)
    if base is None:
        return f"_baseline `{baseline}` was not found among the labels._"

    keys = [
        ("hit@k", "hit_at_k"),
        ("MRR", "mrr"),
        ("false refusal", "false_refusal_rate"),
        ("faithfulness", "faithfulness"),
        ("correctness", "correctness"),
        ("p50 ms", "p50_latency_ms"),
    ]
    lines = ["| label | " + " | ".join(name for name, _ in keys) + " |"]
    lines.append("|" + "---|" * (len(keys) + 1))
    for entry in entries:
        if entry["label"] == baseline:
            continue
        cells = []
        for _, key in keys:
            before = base["summary"].get(key)
            after = entry["summary"].get(key)
            if before is None or after is None:
                cells.append("n/a")
                continue
            digits = 0 if key == "p50_latency_ms" else 3
            cells.append(f"{after - before:+.{digits}f}")
        lines.append(f"| {entry['label']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def failures(entries: list[dict[str, Any]]) -> str:
    lines = ["| label | retrieval miss | false refusal | should have refused |"]
    lines.append("|---|---|---|---|")
    for entry in entries:
        misses, false_refusals, missed_refusals = [], [], []
        for row in entry["rows"]:
            detail = row["details"] or {}
            answerable = bool(detail.get("answerable", True))
            if answerable and row["hit"] is False:
                misses.append(row["question_id"])
            if answerable and row["refused"]:
                false_refusals.append(row["question_id"])
            if not answerable and not row["refused"]:
                missed_refusals.append(row["question_id"])
        lines.append(
            f"| {entry['label']} | {', '.join(misses) or '—'} "
            f"| {', '.join(false_refusals) or '—'} "
            f"| {', '.join(missed_refusals) or '—'} |"
        )
    return "\n".join(lines)


def curve_table(label: str) -> str:
    rows = [
        {
            "answerable": bool((row["details"] or {}).get("answerable", True)),
            "top_score": (row["details"] or {}).get("top_score"),
        }
        for row in evaluate.load_results(label)
    ]
    if not rows:
        return f"_no rows for `{label}`._"

    lines = ["| MIN_TOP_SCORE | false refusal | gate refusal recall |"]
    lines.append("|---|---|---|")
    for row in metrics.threshold_curve(rows, CURVE):
        lines.append(
            f"| {row['threshold']:.2f} | {cell(row['false_refusal_rate'])} "
            f"| {cell(row['gate_refusal_recall'])} |"
        )
    return "\n".join(lines)


def norag_section() -> str:
    path = Path("reports/norag.json")
    if not path.exists():
        return "_`reports/norag.json` not found. Run `make sweep \"ARGS=--only norag\"`._"

    import json

    rows = json.loads(path.read_text(encoding="utf-8"))
    answerable = [row for row in rows if row["answerable"]]
    judged = [row for row in answerable if row["correct"] is not None]
    correct = sum(1 for row in judged if row["correct"]) / len(judged) if judged else None
    unanswerable = [row for row in rows if not row["answerable"]]

    lines = [
        "| no-RAG measurement | value |",
        "|---|---|",
        f"| correctness on answerable | {cell(correct)} |",
        f"| unanswerable questions answered anyway | {len(unanswerable)}/{len(unanswerable)} |",
        f"| p50 latency (ms) | {statistics.median([row['latency_ms'] for row in rows]):.0f} |",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write reports/results.md")
    parser.add_argument("--labels", default=None, help="comma list; default: every label")
    parser.add_argument("--baseline", default="baseline")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    names = (
        [part.strip() for part in args.labels.split(",") if part.strip()]
        if args.labels
        else [row["label"] for row in evaluate.labels()]
    )
    entries = [label_rows(name) for name in names]
    entries = [entry for entry in entries if entry["rows"]]
    if not entries:
        raise SystemExit("no eval_results rows; run `make eval` first")

    body = "\n\n".join(
        [
            "# Results",
            "Generated by `make report`. Do not edit by hand.",
            "## Configurations",
            table(entries),
            f"## Delta against `{args.baseline}`",
            deltas(entries, args.baseline),
            "## Failures by label",
            failures(entries),
            f"## Gate-only threshold curve (`{args.baseline}`)",
            "Recomputed from stored `top_score` values. No model calls.",
            curve_table(args.baseline),
            "## RAG vs no RAG",
            norag_section(),
        ]
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(body.splitlines())} lines)")
    if args.labels is None:
        print("tip: pass --labels to keep scratch labels out of the table")


if __name__ == "__main__":
    main()
