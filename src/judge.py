"""LLM-as-judge for answer quality.

The judge is a second model call with a rubric. It returns one JSON object: the
answer's atomic claims, each marked supported or not, plus a correctness
verdict against the gold answer.

Three rules keep it honest:

* It sees only the question, the answer, and the retrieved context. Not the
  system prompt, not the trace, not the expected phrase.
* A malformed reply becomes `error`, never a score. `faithfulness=None` means
  "not judged", and the summary skips it instead of counting it as zero.
* The judge is an instrument, not an oracle. It is a 3B model grading a 3B
  model; the numbers are a relative signal, not a certificate.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from src import llm
from src.config import settings

JUDGE_SYSTEM = """You grade answers produced by a retrieval-augmented system.

You are strict and literal. You do not use outside knowledge. The retrieved
context is the only evidence you are allowed to consider.

Return one JSON object and nothing else. Its shape is:

{"claims": [{"claim": "one atomic factual statement", "supported": true}],
 "correct": true}

Rules:
- Split the answer into atomic factual claims. One sentence may contain several.
- Set supported to true only when the context states the claim or directly
  entails it. Partial support is false.
- Ignore greetings, hedges, and citation markers such as [1].
- If the answer contains no factual claims (for example a refusal), use [].
- correct compares the answer with the gold answer. Use false when the answer
  contradicts the gold answer or misses its point. Use null when no gold answer
  is given.
"""

JUDGE_USER = """Question:
{question}

Retrieved context:
{context}

Answer to grade:
{answer}

Gold answer (may be absent):
{gold}

Return the JSON object."""

FENCE_RE = re.compile(r"`{3}(?:json)?\s*(.*?)`{3}", re.S)


@dataclass(frozen=True)
class JudgeResult:
    faithfulness: float | None
    correct: bool | None
    claims: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None


def format_context(hits: list[Any]) -> str:
    """Render retrieved chunks the way the answering prompt numbered them."""
    if not hits:
        return "(no chunks were retrieved)"
    return "\n\n".join(
        f"[{rank}] {hit.document} (chunk {hit.chunk_index})\n{hit.content}"
        for rank, hit in enumerate(hits, start=1)
    )


def build_messages(
    question: str,
    answer_text: str,
    context: str,
    gold_answer: str | None = None,
    *,
    strict: bool = False,
) -> list[dict[str, str]]:
    user = JUDGE_USER.format(
        question=question,
        context=context,
        answer=answer_text,
        gold=gold_answer or "(none; set correct to null)",
    )
    if strict:
        user += "\n\nReturn only the JSON object. No prose, no markdown fences."
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": user},
    ]


def extract_json(text: str) -> dict[str, Any]:
    """Find one JSON object in a model reply, fences and prose included."""
    candidate = text.strip()

    fenced = FENCE_RE.search(candidate)
    if fenced:
        candidate = fenced.group(1).strip()
    else:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end < start:
            raise ValueError(f"no JSON object in judge reply: {text[:200]!r}")
        candidate = candidate[start : end + 1]

    payload = json.loads(candidate)
    if not isinstance(payload, dict):
        raise ValueError("judge reply is not a JSON object")
    return payload


def as_bool(value: Any) -> bool | None:
    """Accept a real boolean, or the strings a small model sometimes emits."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
    return None


def faithfulness(claims: list[dict[str, Any]]) -> float | None:
    """Share of claims marked supported. No claims means 'not judged'."""
    if not claims:
        return None
    supported = sum(1 for claim in claims if claim.get("supported") is True)
    return round(supported / len(claims), 3)


def parse_verdict(payload: dict[str, Any], *, has_gold: bool) -> JudgeResult:
    """Turn a decoded payload into a verdict, discarding malformed claims."""
    raw_claims = payload.get("claims")
    claims: list[dict[str, Any]] = []
    if isinstance(raw_claims, list):
        for item in raw_claims:
            if not isinstance(item, dict):
                continue
            text = item.get("claim")
            if not isinstance(text, str) or not text.strip():
                continue
            claims.append(
                {"claim": text.strip(), "supported": as_bool(item.get("supported")) is True}
            )

    correct = as_bool(payload.get("correct"))
    if not has_gold:
        correct = None

    return JudgeResult(
        faithfulness=faithfulness(claims),
        correct=correct,
        claims=claims,
    )


def judge_answer(
    question: str,
    answer_text: str,
    context: str,
    *,
    gold_answer: str | None = None,
    model: str | None = None,
) -> JudgeResult:
    """Grade one answer. One retry with a stricter instruction, then give up."""
    client = llm.get_client()
    last_error = "judge never ran"

    for attempt in (1, 2):
        messages = build_messages(
            question,
            answer_text,
            context,
            gold_answer,
            strict=(attempt == 2),
        )
        try:
            result = client.generate(
                messages,
                model=model or settings.judge_model,
                temperature=0.0,
                fmt="json",
                num_ctx=settings.num_ctx,
            )
            # The client returns a dict, not an object.
            return parse_verdict(
                extract_json(result["text"]), has_gold=bool(gold_answer)
            )
        except Exception as exc:
            last_error = f"{exc.__class__.__name__}: {exc}"

    return JudgeResult(
        faithfulness=None,
        correct=None,
        claims=[],
        error=last_error,
    )
