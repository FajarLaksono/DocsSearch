"""Build the grounded prompt, call the model, parse citations."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

from src import llm
from src.retrieve import Hit, search

SYSTEM_PROMPT = """You answer questions using ONLY the numbered sources provided.

Rules:
1. Use only the sources. Do not use outside knowledge.
2. Cite each claim with the source number in square brackets, like [1] or [2][3].
3. If the sources do not contain the answer, reply exactly: I don't know based on the provided documents.
4. Answer in at most 150 words.
"""

CITATION_RE = re.compile(r"\[(\d+)\]")
REFUSAL_RE = re.compile(r"i (?:do not|don't) know", re.IGNORECASE)

REFUSAL_TEXT = "I don't know based on the provided documents."


@dataclass
class Answer:
    question: str
    text: str
    hits: list[Hit]
    citations: list[int]
    refused: bool
    latency_ms: int
    tokens_in: int
    tokens_out: int


def build_messages(question: str, hits: list[Hit]) -> list[dict]:
    sources = "\n\n".join(
        f"[{rank}] (document: {hit.document}, chunk {hit.chunk_index})\n{hit.content}"
        for rank, hit in enumerate(hits, start=1)
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {question}\n\nSources:\n\n{sources}"},
    ]


def parse_citations(text: str, n_sources: int) -> list[int]:
    """Return valid citation ranks in order of first appearance."""
    found: list[int] = []
    for match in CITATION_RE.finditer(text):
        rank = int(match.group(1))
        if 1 <= rank <= n_sources and rank not in found:
            found.append(rank)
    return found


def is_refusal(text: str) -> bool:
    return REFUSAL_RE.search(text) is not None


def answer_question(
    question: str,
    k: int | None = None,
    model: str | None = None,
) -> Answer:
    started = time.perf_counter()
    hits = search(question, k=k)

    if not hits:
        elapsed = int((time.perf_counter() - started) * 1000)
        return Answer(question, REFUSAL_TEXT, [], [], True, elapsed, 0, 0)

    result = llm.get_client().generate(build_messages(question, hits), model=model)
    text = result["text"].strip()

    return Answer(
        question=question,
        text=text,
        hits=hits,
        citations=parse_citations(text, len(hits)),
        refused=is_refusal(text),
        latency_ms=int((time.perf_counter() - started) * 1000),
        tokens_in=result["tokens_in"],
        tokens_out=result["tokens_out"],
    )
