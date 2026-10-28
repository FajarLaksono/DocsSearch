"""Explain an answer using the material it was built from.

Three checks, none of which needs a new model for the first one:

    highlight()      which source sentences overlap each answer sentence
    leave_one_out()  what the answer looks like with one chunk removed
    without_rag()    what the model says with no sources at all

The first is lexical and instant. The other two call the model, because the only
way to see a counterfactual is to generate it.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from dataclasses import dataclass
from html import escape
from typing import Any

from src import llm, trace
from src.config import settings
from src.generate import answer_from_hits, answer_question
from src.retrieve import Hit

STOPWORDS = frozenset(
    """
    a an and are as at be been but by can could did do does for from had has have
    how i if in into is it its of on or that the their them then there these they
    this to was were what when where which who why will with would you your
    """.split()
)

SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
WORD_RE = re.compile(r"[a-z0-9]+")

# One shared content word in five is enough to call it a match. Below this the
# table fills with sentences that share only "retrieval".
MIN_OVERLAP = 0.2

# An answer that keeps 85% of its words after a chunk is removed did not need
# that chunk. Below this, the chunk was doing work.
SUPPORT_THRESHOLD = 0.85

NO_RAG_PROMPT = """Answer the question from your own knowledge.

There are no sources. Do not cite anything, and do not pretend to quote a
document. If you are not sure, say so in one sentence. Answer in at most 80
words.
"""


@dataclass(frozen=True)
class Influence:
    removed_rank: int
    chunk_id: int
    document: str
    score: float
    kept: float
    citations: list[int]
    answer: str

    @property
    def changed(self) -> bool:
        return self.kept < SUPPORT_THRESHOLD


def split_sentences(text: str) -> list[str]:
    """Flatten whitespace, then split after . ! or ?"""
    flat = " ".join(text.split())
    if not flat:
        return []
    parts = [part.strip() for part in SENTENCE_RE.split(flat) if part.strip()]
    return parts or [flat]


def content_words(text: str) -> set[str]:
    """Lowercase words longer than two characters, minus stopwords."""
    return {
        word
        for word in WORD_RE.findall(text.lower())
        if len(word) > 2 and word not in STOPWORDS
    }


def overlap(answer_sentence: str, source_sentence: str) -> float:
    """Fraction of the answer sentence's content words found in the source."""
    wanted = content_words(answer_sentence)
    if not wanted:
        return 0.0
    return round(len(wanted & content_words(source_sentence)) / len(wanted), 4)


def similarity(left: str, right: str) -> float:
    """Jaccard over content words. Symmetric, so variants compare cleanly."""
    a, b = content_words(left), content_words(right)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return round(len(a & b) / len(a | b), 4)


def highlight(
    answer_text: str,
    hits: list[Hit],
    *,
    threshold: float = MIN_OVERLAP,
    per_sentence: int = 3,
) -> list[dict[str, Any]]:
    """For each answer sentence, the source sentences it shares words with."""
    sources: list[tuple[int, Hit, str]] = []
    for rank, hit in enumerate(hits, start=1):
        for sentence in split_sentences(hit.content):
            sources.append((rank, hit, sentence))

    rows: list[dict[str, Any]] = []
    for sentence in split_sentences(answer_text):
        matches: list[dict[str, Any]] = []
        for rank, hit, source_sentence in sources:
            score = overlap(sentence, source_sentence)
            if score < threshold:
                continue
            matches.append(
                {
                    "rank": rank,
                    "chunk_id": hit.chunk_id,
                    "document": hit.document,
                    "sentence": source_sentence,
                    "score": score,
                }
            )
        matches.sort(key=lambda match: (-match["score"], match["rank"], match["chunk_id"]))
        rows.append({"sentence": sentence, "matches": matches[:per_sentence]})
    return rows


def marked_sentences(rows: list[dict[str, Any]], rank: int) -> list[str]:
    """The source sentences to highlight for one rank, best first."""
    seen: list[str] = []
    for row in rows:
        for match in row["matches"]:
            if match["rank"] == rank and match["sentence"] not in seen:
                seen.append(match["sentence"])
    return seen


def highlight_html(content: str, sentences: list[str]) -> str:
    """Escape first, then wrap known sentences. Never inject raw model text."""
    flat = " ".join(content.split())
    html = escape(flat)
    for sentence in sorted(sentences, key=len, reverse=True):
        needle = escape(sentence)
        if needle and needle in html:
            html = html.replace(needle, f"<mark>{needle}</mark>", 1)
    return html


def leave_one_out(
    question: str,
    answer: Any,
    hits: list[Hit],
    *,
    model: str | None = None,
) -> list[Influence]:
    """Re-answer with each chunk removed, and measure how much survived."""
    rows: list[Influence] = []
    for rank, hit in enumerate(hits, start=1):
        subset = [other for other in hits if other.chunk_id != hit.chunk_id]
        if not subset:
            continue
        variant = answer_from_hits(question, subset, model=model)
        rows.append(
            Influence(
                removed_rank=rank,
                chunk_id=hit.chunk_id,
                document=hit.document,
                score=hit.score,
                kept=similarity(answer.text, variant.text),
                citations=list(variant.citations),
                answer=variant.text,
            )
        )
    return rows


def without_rag(question: str, *, model: str | None = None) -> dict[str, Any]:
    """The model alone. No sources, no citations, no retrieval."""
    started = time.perf_counter()
    result = llm.get_client().generate(
        [
            {"role": "system", "content": NO_RAG_PROMPT},
            {"role": "user", "content": question},
        ],
        model=model,
        temperature=0.0,
        num_ctx=settings.num_ctx,
    )
    return {
        "text": str(result.get("text") or "").strip(),
        "latency_ms": int((time.perf_counter() - started) * 1000),
        "tokens_in": int(result.get("tokens_in") or 0),
        "tokens_out": int(result.get("tokens_out") or 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Explain one answer")
    parser.add_argument("question", nargs="?", default="")
    parser.add_argument("--run", type=int, default=None, help="explain a stored run")
    parser.add_argument("--model", default=None, help="override LLM_MODEL")
    parser.add_argument("--k", type=int, default=None, help="override TOP_K")
    parser.add_argument(
        "--no-counterfactual",
        action="store_true",
        help="skip leave-one-out and no-RAG (no model calls)",
    )
    args = parser.parse_args()

    if args.run:
        run = trace.load_run(args.run)
        answer = trace.answer_from_run(run)
        question = answer.question
        print(f"run #{run['id']}  {run['llm_model']}  {run['latency_ms']} ms")
    elif args.question.strip():
        question = " ".join(args.question.split())
        answer = answer_question(question, k=args.k, model=args.model)
    else:
        raise SystemExit("give a question or --run ID")

    print(f"Q: {question}")
    print(f"A: {answer.text}")
    print()

    rows = highlight(answer.text, answer.hits)
    matched = sum(1 for row in rows if row["matches"])
    print(f"Highlights ({len(rows)} answer sentences, {matched} matched):")
    for number, row in enumerate(rows, start=1):
        best = row["matches"][0] if row["matches"] else None
        if best is None:
            print(f"  [{number}] no source sentence shares its words")
            continue
        print(
            f"  [{number}] {best['score']:.2f}  source [{best['rank']}] "
            f"{best['document']}"
        )
        print(f"       answer: {row['sentence']}")
        print(f"       source: {best['sentence']}")

    if args.no_counterfactual:
        return

    print()
    print("Leave-one-out (kept 1.000 = the answer did not change):")
    for row in leave_one_out(question, answer, answer.hits, model=args.model):
        mark = "  <- influential" if row.changed else ""
        print(
            f"  removed [{row.removed_rank}] {row.score:.3f}  "
            f"chunk {row.chunk_id:<4} {row.document:<32} "
            f"kept {row.kept:.3f}{mark}"
        )

    print()
    plain = without_rag(question, model=args.model)
    print("Without RAG (no sources, no citations):")
    print(f"  {plain['text']}")
    print(
        f"  [{plain['latency_ms']} ms  {plain['tokens_in']} in / "
        f"{plain['tokens_out']} out]"
    )


if __name__ == "__main__":
    sys.exit(main())
