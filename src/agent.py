"""The agent loop: search, look, decide, answer.

The model is a planner, not a writer. It chooses `search_docs` or `answer`; the
final prose still comes from `generate.answer_from_hits`, the same grounded
prompt Phase 3 measured. That keeps one answer format and makes the agent-vs-
baseline comparison honest: only retrieval changes.

Protocol, one JSON object per reply:

    {"thought": "...", "action": "search_docs", "input": "query"}
    {"thought": "...", "action": "answer", "input": ""}

The loop is bounded by MAX_STEPS, and the answer is gated by MIN_TOP_SCORE.
Both are deterministic. Neither trusts the model to police itself.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass, replace
from typing import Any

from src import llm, tools, trace
from src.config import settings
from src.generate import REFUSAL_TEXT, Answer, answer_from_hits, is_refusal
from src.retrieve import Hit

ACTIONS = ("search_docs", "answer")
FENCE_RE = re.compile(r"`{3}(?:json)?\s*(.*?)`{3}", re.S)

SYSTEM_PROMPT = """You plan retrieval over a small document corpus.

Each turn you reply with exactly one JSON object and nothing else. There are two
actions.

Search the corpus:
{"thought": "one short sentence", "action": "search_docs", "input": "a search query"}

Stop searching:
{"thought": "one short sentence", "action": "answer", "input": ""}

Rules:
- Start with a search that reuses the question's own words.
- Read the sources you already have. If they contain the answer, choose
  "answer". Do not search just to confirm.
- If they do not, search again with different words: synonyms, or the vocabulary
  the documents use. Never repeat a query you already tried.
- "answer" does not write the answer. A separate grounded prompt writes it from
  the sources you collected. Your job is to decide when the evidence is enough.
- Never invent a source number or a fact.
"""

THOUGHT_REQUIRED = """
Always include a "thought" of at most 20 words. It is shown to the user, so it
must explain the decision, not restate the question.
"""

NO_THOUGHT = """
Thinking is off for this run: omit the "thought" field. Return only "action" and
"input".
"""


@dataclass(frozen=True)
class Action:
    name: str
    input: str
    thought: str = ""


@dataclass
class AgentResult:
    answer: Answer
    steps: list[dict[str, Any]]
    metrics: dict[str, Any]


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
            raise ValueError(f"no JSON object in reply: {text[:120]!r}")
        candidate = candidate[start : end + 1]

    payload = json.loads(candidate)
    if not isinstance(payload, dict):
        raise ValueError("reply is not a JSON object")
    return payload


def parse_action(text: str) -> Action:
    """Validate one planner reply. Every rejection is a ValueError."""
    payload = extract_json(text)

    name = payload.get("action")
    if not isinstance(name, str) or name not in ACTIONS:
        raise ValueError(f"unknown action {name!r}; use search_docs or answer")

    raw_input = payload.get("input", "")
    if raw_input is None:
        raw_input = ""
    if not isinstance(raw_input, str):
        raise ValueError("input must be a string")

    thought = payload.get("thought", "")
    if thought is None:
        thought = ""
    if not isinstance(thought, str):
        raise ValueError("thought must be a string")

    query = " ".join(raw_input.split())
    if name == "search_docs" and not query:
        raise ValueError("search_docs needs a non-empty input query")

    return Action(name=name, input=query, thought=thought.strip())


def build_system_prompt(*, max_steps: int, thinking: str) -> str:
    mode = THOUGHT_REQUIRED if thinking == "prompted" else NO_THOUGHT
    return "\n".join(
        [
            SYSTEM_PROMPT.strip(),
            f"You have at most {max_steps} decisions. Reserve the last one for answer.",
            mode.strip(),
        ]
    )


def _source_block(hits: list[Hit]) -> str:
    if not hits:
        return "No sources yet. Your first action must be search_docs."
    lines: list[str] = []
    for rank, hit in enumerate(hits, start=1):
        lines.append(
            f"[{rank}] {hit.score:.3f}  {hit.document}  chunk {hit.chunk_index}"
        )
        lines.append(f"    {tools.preview(hit.content)}")
    return "\n".join(lines)


def _history(steps: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for step in steps:
        if step.get("type") == "tool_call" and step.get("name") == "search_docs":
            lines.append(f'- search_docs("{step.get("input", "")}")')
        elif step.get("type") == "observation":
            if step.get("error"):
                lines.append(f"- protocol error: {step['error']}")
            else:
                lines.append(f"- result: {step.get('summary', 'no summary')}")
    return "\n".join(lines) if lines else "(none yet)"


def build_messages(
    question: str,
    hits: list[Hit],
    steps: list[dict[str, Any]],
    *,
    step_no: int,
    max_steps: int,
    thinking: str,
) -> list[dict[str, str]]:
    user = (
        f"Question: {question}\n\n"
        f"Current sources ({len(hits)}):\n{_source_block(hits)}\n\n"
        f"Previous actions:\n{_history(steps)}\n\n"
        f"Decision {step_no} of {max_steps}. Reply with one JSON object."
    )
    if steps and steps[-1].get("type") == "observation" and steps[-1].get("error"):
        user += (
            "\n\nYour previous reply could not be parsed: "
            f"{steps[-1]['error']}. Return exactly one JSON object."
        )
    return [
        {
            "role": "system",
            "content": build_system_prompt(max_steps=max_steps, thinking=thinking),
        },
        {"role": "user", "content": user},
    ]


def merge_hits(existing: list[Hit], found: list[Hit], *, k: int) -> list[Hit]:
    """Best copy of each chunk, sorted, capped at k.

    The agent keeps a leaderboard, not a transcript. Three searches can see twelve
    chunks; the answerer sees the best four. That keeps the prompt inside
    NUM_CTX and makes the citation ranks stable.
    """
    best: dict[int, Hit] = {hit.chunk_id: hit for hit in existing}
    for hit in found:
        current = best.get(hit.chunk_id)
        if current is None or hit.score > current.score:
            best[hit.chunk_id] = hit
    ordered = sorted(best.values(), key=lambda hit: (-hit.score, hit.chunk_id))
    return ordered[:k]


def gate(hits: list[Hit]) -> str | None:
    """Return a refusal reason, or None when the evidence clears the floor."""
    if not hits:
        return "no_sources"
    if hits[0].score < settings.min_top_score:
        return "below_min_top_score"
    return None


def _chunk(hit: Hit) -> dict[str, Any]:
    return {
        "chunk_id": hit.chunk_id,
        "document": hit.document,
        "chunk_index": hit.chunk_index,
        "score": round(hit.score, 6),
    }


def _refusal(question: str, hits: list[Hit]) -> Answer:
    return Answer(question, REFUSAL_TEXT, hits, [], True, 0, 0, 0)


def run_agent(
    question: str,
    *,
    model: str | None = None,
    k: int | None = None,
    max_steps: int | None = None,
    thinking: str | None = None,
) -> AgentResult:
    """Plan, search, and answer. No database writes: the caller records."""
    question = " ".join(question.split())
    if not question:
        raise ValueError("question is empty")

    limit = settings.top_k if k is None else k
    if limit < 1:
        raise ValueError("k must be at least 1")
    budget = settings.max_steps if max_steps is None else max_steps
    if budget < 1:
        raise ValueError("max_steps must be at least 1")
    mode = (thinking or settings.thinking).lower()
    if mode not in ("off", "prompted"):
        raise ValueError(f"thinking must be off or prompted, not {mode!r}")

    client = llm.get_client()
    started = time.perf_counter()

    steps: list[dict[str, Any]] = []
    hits: list[Hit] = []
    searches = 0
    calls = 0
    tokens_in = 0
    tokens_out = 0
    invalid_streak = 0
    stop = "step_limit"
    step_no = 0

    while step_no < budget:
        messages = build_messages(
            question,
            hits,
            steps,
            step_no=step_no + 1,
            max_steps=budget,
            thinking=mode,
        )
        try:
            result = client.generate(
                messages,
                model=model,
                temperature=0.0,
                fmt="json",
                num_ctx=settings.num_ctx,
            )
        except Exception as exc:
            steps.append(
                {
                    "type": "observation",
                    "summary": "model call failed",
                    "error": f"{exc.__class__.__name__}: {exc}",
                }
            )
            stop = "model_error"
            break

        calls += 1
        tokens_in += int(result.get("tokens_in") or 0)
        tokens_out += int(result.get("tokens_out") or 0)
        raw = str(result.get("text") or "")

        try:
            action = parse_action(raw)
        except ValueError as exc:
            invalid_streak += 1
            steps.append(
                {
                    "type": "observation",
                    "summary": "invalid JSON",
                    "error": str(exc),
                    "raw": raw[:300],
                }
            )
            if invalid_streak >= 2:
                stop = "invalid_json"
                break
            continue  # a retry does not consume a decision

        invalid_streak = 0
        if mode == "prompted" and action.thought:
            steps.append(
                {"type": "thought", "step": step_no + 1, "text": action.thought}
            )
        steps.append(
            {
                "type": "tool_call",
                "step": step_no + 1,
                "name": action.name,
                "input": action.input,
            }
        )

        if action.name == "answer":
            step_no += 1
            stop = "answer"
            break

        before = {hit.chunk_id: hit.score for hit in hits}
        found = tools.search_docs(action.input, k=limit)
        searches += 1
        merged = merge_hits(hits, found, k=limit)
        changes: dict[int, str] = {}
        for hit in merged:
            if hit.chunk_id not in before:
                changes[hit.chunk_id] = "new"
            elif hit.score > before[hit.chunk_id]:
                changes[hit.chunk_id] = "improved"
        hits = merged
        steps.append(
            {
                "type": "observation",
                "step": step_no + 1,
                "query": action.input,
                "chunks": [_chunk(hit) for hit in found],
                "changed_ids": sorted(changes),
                "summary": f"{len(changes)} source(s) changed",
                "text": tools.format_observation(action.input, hits, changes),
            }
        )
        step_no += 1

    blocked = gate(hits)
    if blocked:
        answer = _refusal(question, hits)
        reason = blocked
    else:
        answer = answer_from_hits(question, hits, model=model)
        calls += 1
        tokens_in += answer.tokens_in
        tokens_out += answer.tokens_out
        reason = "model_refused" if is_refusal(answer.text) else "answered"

    elapsed = int((time.perf_counter() - started) * 1000)
    answer = replace(
        answer,
        latency_ms=elapsed,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
    )

    metrics = trace.compute_metrics(
        [hit.score for hit in hits], answer.citations, answer.refused
    )
    metrics.update(
        {
            "reason": reason,
            "stop_reason": stop,
            "steps": step_no,
            "searches": searches,
            "llm_calls": calls,
            "thinking": mode,
            "max_steps": budget,
        }
    )
    return AgentResult(answer=answer, steps=steps, metrics=metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the agent loop")
    parser.add_argument("question", nargs="+")
    parser.add_argument("--k", type=int, default=None, help="override TOP_K")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--thinking", choices=("off", "prompted"), default=None)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    question = " ".join(args.question)
    result = run_agent(
        question,
        model=args.model,
        k=args.k,
        max_steps=args.max_steps,
        thinking=args.thinking,
    )
    answer = result.answer

    print(f"Question: {answer.question}")
    print()
    print("Decisions:")
    if not result.steps:
        print("  (none: the loop never produced a valid decision)")
    for step in result.steps:
        kind = step.get("type")
        if kind == "thought":
            print(f"  step {step.get('step')}  thought: {step['text']}")
        elif kind == "tool_call":
            if step["name"] == "answer":
                print(f"  step {step.get('step')}  action: answer")
            else:
                print(
                    f"  step {step.get('step')}  action: "
                    f"search_docs({step['input']!r})"
                )
        elif step.get("error"):
            print(f"           protocol: {step['error']}")
        else:
            print(f"           result: {step.get('summary', '')}")

    print()
    print(answer.text)
    print()

    if answer.hits:
        print(f"Sources ({len(answer.hits)} retrieved, * = cited):")
        for rank, hit in enumerate(answer.hits, start=1):
            marker = "*" if rank in answer.citations else " "
            preview_text = " ".join(hit.content.split())[:100]
            print(
                f" {marker}[{rank}] {hit.score:.3f}  {hit.document}  "
                f"chunk {hit.chunk_index}"
            )
            print(f"      {preview_text}...")
    else:
        print("Sources: none.")

    print(
        f"\n[reason={result.metrics['reason']}  "
        f"stop={result.metrics['stop_reason']}  "
        f"searches={result.metrics['searches']}  "
        f"decisions={result.metrics['steps']}  "
        f"{answer.latency_ms} ms  "
        f"{answer.tokens_in} in / {answer.tokens_out} out]"
    )

    run_id = trace.record(
        answer,
        llm_model=args.model or settings.llm_model,
        steps=result.steps,
        metrics=result.metrics,
    )
    print(f"run #{run_id} recorded. Inspect it with `make trace ID={run_id}` or the UI.")


if __name__ == "__main__":
    main()
