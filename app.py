"""DocsSearch - Phase 2: ask, inspect, explain.

The page is a view over one row in `runs`. Everything it shows -- answer,
chunks, scores, steps, metrics -- is read back from that row. Nothing is
recomputed from the current index, so an old run keeps explaining itself even
after the corpus changes.
"""

from __future__ import annotations

import html
import re
from datetime import datetime
from typing import Any

import streamlit as st

from src import llm, trace, agent, explain, evaluate, judge
from src.config import settings
from src.generate import Answer, answer_question

CITATION_RE = re.compile(r"\[(\d+)\]")
EMBED_HINTS = ("embed", "minilm", "bge", "e5-", "nomic-embed")

st.set_page_config(page_title="DocsSearch", page_icon="📚", layout="wide")


@st.cache_data(ttl=30, show_spinner=False)
def installed_models() -> list[str]:
    """Ask Ollama once per 30 s, not once per widget interaction."""
    return llm.get_client().list_models()


def chat_models(names: list[str]) -> list[str]:
    """Drop embedding models from the chat picker."""
    chat = [name for name in names if not any(hint in name.lower() for hint in EMBED_HINTS)]
    return chat or [settings.llm_model]


def jsonable(value: Any) -> Any:
    """Make a row safe for st.json: datetimes become ISO strings."""
    if isinstance(value, datetime):
        return value.isoformat(timespec="seconds")
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def render_answer(text: str, n_sources: int) -> str:
    """Escape model text, then link each valid [n] to its source card."""

    def link(match: re.Match) -> str:
        rank = int(match.group(1))
        if 1 <= rank <= n_sources:
            return f'<a href="#source-{rank}">[{rank}]</a>'
        return match.group(0)

    return CITATION_RE.sub(link, html.escape(text))


def confidence_flags(answer: Answer, threshold: float) -> list[tuple[str, str]]:
    """Rule-based warnings from stored scores. No model call, no guessing."""
    scores = [hit.score for hit in answer.hits]
    if not scores:
        return [("error", "No chunks were retrieved. Run `make ingest`.")]

    flags: list[tuple[str, str]] = []
    if answer.refused:
        flags.append(
            ("info", "The model refused: the sources did not answer the question.")
        )
    if scores[0] < threshold:
        flags.append(
            (
                "error",
                f"Top score {scores[0]:.3f} is below MIN_TOP_SCORE {threshold:.2f}; "
                "retrieval is probably a miss.",
            )
        )
    if len(scores) > 1 and scores[0] - scores[1] < 0.02:
        flags.append(
            (
                "warning",
                f"The top two scores are {scores[0] - scores[1]:.3f} apart; "
                "the retriever cannot separate them.",
            )
        )

    score_by_rank = {rank: score for rank, score in enumerate(scores, start=1)}
    weak = [rank for rank in answer.citations if score_by_rank.get(rank, 1.0) < threshold]
    if weak:
        flags.append(("warning", f"Cited source(s) {weak} scored below the threshold."))
    if not answer.refused and not answer.citations:
        flags.append(("error", "The answer cites nothing; treat it as ungrounded."))
    if not flags:
        flags.append(
            ("success", "No red flags: retrieval cleared the threshold and the answer cites sources.")
        )
    return flags


def agent_flags(metrics: dict[str, Any]) -> list[tuple[str, str]]:
    """Warnings that only exist because there was a loop."""
    flags: list[tuple[str, str]] = []
    stop = metrics.get("stop_reason")
    reason = metrics.get("reason")
    searches = int(metrics.get("searches") or 0)

    if stop == "step_limit":
        flags.append(
            (
                "warning",
                "The planner used every decision without choosing answer; "
                "this answer came from a partial search history.",
            )
        )
    if stop == "invalid_json":
        flags.append(
            (
                "error",
                "Two consecutive planner replies could not be parsed; "
                "the answer fell back to the sources collected so far.",
            )
        )
    if stop == "model_error":
        flags.append(("error", "A model call raised; the loop stopped early."))
    if reason == "answered" and searches >= 2:
        flags.append(
            (
                "info",
                f"The planner needed {searches} searches; "
                "the question's own words were not enough.",
            )
        )
    return flags


# ------------------------------------------------------------------ sidebar ---
with st.sidebar:
    st.title("DocsSearch")
    st.caption(f"Embedding model is fixed at `{settings.embed_model}`; stored vectors belong to it.")

    try:
        model_names = chat_models(installed_models())
        model = st.selectbox("Chat model", model_names)
    except Exception as exc:
        st.error(f"Cannot list Ollama models: {exc}")
        model = st.text_input("Chat model", settings.llm_model)

    top_k = st.slider("Chunks to retrieve (k)", 1, 8, settings.top_k)

    st.divider()
    st.subheader("Pipeline")
    mode = st.radio(
        "Mode",
        ["Single pass", "Agent"],
        horizontal=True,
        help="Single pass: one retrieval. Agent: up to N search decisions.",
    )
    use_agent = mode == "Agent"
    thinking = "off"
    budget = settings.max_steps
    if use_agent:
        thinking = st.selectbox("Thinking", ["off", "prompted"], index=0)
        budget = st.slider("Max decisions", 1, 5, settings.max_steps)


    st.divider()
    st.subheader("Confidence")
    threshold = st.slider(
        "MIN_TOP_SCORE (flag only)",
        0.0,
        1.0,
        float(settings.min_top_score),
        0.01,
    )
    st.caption("The agent's gate uses this value to refuse before answering.")


    st.divider()
    st.subheader("History")
    try:
        history = trace.recent_runs(limit=10)
    except Exception as exc:
        history = []
        st.caption(f"History unavailable: {exc.__class__.__name__}")

    if history:
        labels = {
            row["id"]: f"#{row['id']}  {row['question'][:34]}" for row in history
        }
        chosen = st.selectbox("Load a run", list(labels), format_func=labels.get)
        if st.button("Load", use_container_width=True):
            loaded = trace.load_run(int(chosen))
            st.session_state["run"] = loaded
            st.session_state["answer"] = trace.answer_from_run(loaded)
    else:
        st.caption("No runs yet. Ask a question below.")

# --------------------------------------------------------------------- main ---
st.header("Ask the corpus")
st.caption("Every question writes a row to Postgres. Everything below is read back from that row.")

with st.form("ask"):
    question = st.text_area(
        "Question",
        value="What is hit rate at k?",
        height=80,
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button("Ask", type="primary")

if submitted:
    if not question.strip():
        st.warning("Type a question first.")
    else:
        with st.spinner(f"Asking {model}…"):
            try:
                if use_agent:
                    result = agent.run_agent(
                        question.strip(),
                        model=model,
                        k=top_k,
                        max_steps=budget,
                        thinking=thinking,
                    )
                    answer = result.answer
                    run_id = trace.record(
                        answer,
                        llm_model=model,
                        steps=result.steps,
                        metrics=result.metrics,
                    )
                else:
                    answer = answer_question(question.strip(), k=top_k, model=model)
                    run_id = trace.record(answer, llm_model=model)
                st.session_state["answer"] = answer
                st.session_state["run"] = trace.load_run(run_id)
            except Exception as exc:
                st.error(f"The question failed: {exc}")


answer: Answer | None = (
    st.session_state["answer"] if "answer" in st.session_state else None
)
run: dict[str, Any] | None = st.session_state["run"] if "run" in st.session_state else None

if answer is None:
    st.info("Ask a question, or load one from the sidebar.")
    st.stop()

metrics = (run or {}).get("metrics") or trace.compute_metrics(
    [hit.score for hit in answer.hits], answer.citations, answer.refused
)

columns = st.columns(4)
columns[0].metric("Run", f"#{run['id']}" if run else "unsaved")
columns[1].metric(
    "Top score",
    f"{metrics['top_score']:.3f}" if metrics.get("top_score") is not None else "n/a",
)
columns[2].metric("Latency", f"{answer.latency_ms} ms")
columns[3].metric("Tokens", f"{answer.tokens_in} in / {answer.tokens_out} out")

if metrics.get("reason"):
    st.caption(
        f"agent · {metrics['reason']} · stop={metrics['stop_reason']} · "
        f"{metrics['steps']} decision(s) · {metrics['searches']} search(es) · "
        f"{metrics['llm_calls']} model call(s) · thinking={metrics.get('thinking')}"
    )

st.subheader("Answer")
st.markdown(render_answer(answer.text, len(answer.hits)), unsafe_allow_html=True)

if run and run.get("llm_model") != model:
    st.caption(f"This run was produced by `{run['llm_model']}`, not `{model}`.")

# sources_tab, trace_tab, explain_tab, confidence_tab, evaluation_tab, raw_tab = st.tabs(
#     ["Sources", "Trace", "Explain", "Confidence", "Evaluation", "Raw"]
# )

sources_tab, trace_tab, explain_tab, confidence_tab, raw_tab = st.tabs(
    ["Sources", "Trace", "Explain", "Confidence", "Raw"]
)

with sources_tab:
    if not answer.hits:
        st.warning("No sources were retrieved; this answer is a refusal by construction.")
    for rank, hit in enumerate(answer.hits, start=1):
        st.markdown(f'<div id="source-{rank}"></div>', unsafe_allow_html=True)
        header = (
            f"[{rank}] {hit.score:.3f}  {hit.document} · chunk {hit.chunk_index}"
            + ("  ✓ cited" if rank in answer.citations else "")
        )
        with st.expander(header, expanded=(rank == 1)):
            if hit.score < threshold:
                st.warning(f"Below MIN_TOP_SCORE ({threshold:.2f}).")
            st.write(hit.content)
            st.caption(f"chunk_id={hit.chunk_id}")

with trace_tab:
    steps = (run or {}).get("steps") or []
    if not steps:
        st.info("No stored steps for this answer.")
    
    if any(
        step.get("type") in ("thought", "tool_call", "observation")
        for step in steps
    ):
        st.caption("Agent decisions in order. Read top to bottom.")

    for index, step in enumerate(steps, start=1):
        kind = step.get("type", "unknown")
        number = step.get("step", index)

        if kind == "thought":
            st.markdown(f"**Step {number} — thought**\n\n> {step.get('text', '')}")
        elif kind == "tool_call":
            name = step.get("name", "?")
            if name == "answer":
                st.markdown(f"**Step {number} — action** `answer`")
            else:
                st.markdown(
                    f"**Step {number} — action** `{name}({step.get('input', '')})`"
                )
        elif kind == "observation":
            if step.get("error"):
                st.warning(f"Protocol error: {step['error']}")
            else:
                st.write(step.get("text") or step.get("summary", ""))
                chunks = step.get("chunks", [])
                if chunks:
                    with st.expander(f"{len(chunks)} chunk(s) returned"):
                        st.dataframe(chunks, hide_index=True, use_container_width=True)
        elif kind == "retrieval":
            st.write(f"Embedded query: `{step.get('query')}`")
            chunks = step.get("chunks", [])
            if chunks:
                st.dataframe(chunks, hide_index=True, use_container_width=True)
        elif kind == "answer":
            st.write(step.get("text", ""))
            st.caption(f"citations={step.get('citations')} refused={step.get('refused')}")
        else:
            st.json(jsonable(step))

    if run:
        st.caption(
            f"llm_model={run['llm_model']} · embed_model={run['embed_model']} · "
            f"created_at={jsonable(run['created_at'])}"
        )

with confidence_tab:
    st.caption("Rule-based flags from the stored scores. No model is called.")
    for level, message in confidence_flags(answer, threshold) + agent_flags(metrics):
        getattr(st, level)(message)
    st.divider()
    st.json(jsonable(metrics))


with raw_tab:
    if run:
        st.json(jsonable(run))
    else:
        st.info("This answer was not stored.")

with explain_tab:
    st.caption(
        "Highlighting is lexical and free. The two counterfactuals call the model "
        "and are never run on page load."
    )

    rows = explain.highlight(answer.text, answer.hits)
    matched = sum(1 for row in rows if row["matches"])

    if not answer.hits:
        st.warning("No sources were retrieved; there is nothing to attribute.")
    else:
        st.markdown(
            f"**Sentence coverage** — {matched} of {len(rows)} answer sentences "
            "share content words with a source."
        )
        st.dataframe(
            [
                {
                    "answer sentence": row["sentence"][:90],
                    "source": (
                        f"[{row['matches'][0]['rank']}] "
                        f"{row['matches'][0]['document']}"
                        if row["matches"]
                        else "—"
                    ),
                    "overlap": (
                        row["matches"][0]["score"] if row["matches"] else 0.0
                    ),
                }
                for row in rows
            ],
            hide_index=True,
            use_container_width=True,
        )

        by_rank = {
            rank: explain.marked_sentences(rows, rank)
            for rank in range(1, len(answer.hits) + 1)
        }
        st.markdown("**Marked sources** — the sentences behind each citation.")
        for rank, hit in enumerate(answer.hits, start=1):
            if not by_rank[rank]:
                continue
            with st.expander(
                f"[{rank}] {hit.score:.3f}  {hit.document} · chunk {hit.chunk_index}",
                expanded=(rank == 1),
            ):
                st.markdown(
                    explain.highlight_html(hit.content, by_rank[rank]),
                    unsafe_allow_html=True,
                )
        if not any(by_rank.values()):
            st.info("No answer sentence shares content words with a source. That is a citation to distrust.")

    st.divider()
    st.markdown("**Leave-one-out** — remove one chunk, re-answer, compare.")
    loo_key = f"loo_{(run or {}).get('id', answer.question)}"
    if st.button(
        f"Run leave-one-out ({len(answer.hits)} model calls)",
        key="loo_button",
        disabled=len(answer.hits) < 2,
    ):
        with st.spinner("Re-answering with each chunk removed…"):
            st.session_state[loo_key] = explain.leave_one_out(
                answer.question, answer, answer.hits
            )

    influences = st.session_state.get(loo_key)
    if influences:
        st.dataframe(
            [
                {
                    "removed": f"[{row.removed_rank}]",
                    "chunk_id": row.chunk_id,
                    "document": row.document,
                    "score": row.score,
                    "kept": row.kept,
                    "verdict": "influential" if row.changed else "dispensable",
                    "citations": ",".join(f"[{c}]" for c in row.citations) or "none",
                }
                for row in influences
            ],
            hide_index=True,
            use_container_width=True,
        )
        worst = min(influences, key=lambda row: row.kept)
        st.caption(
            f"Most influential: chunk {worst.chunk_id} (kept {worst.kept:.3f}). "
            "Low `kept` means the answer depended on it."
        )
        with st.expander("Read the counterfactual answers"):
            for row in influences:
                st.markdown(f"**Without [{row.removed_rank}]** — kept {row.kept:.3f}")
                st.write(row.answer)

    st.divider()
    st.markdown("**Without RAG** — the same question, no corpus.")
    plain_key = f"plain_{(run or {}).get('id', answer.question)}"
    if st.button("Answer without RAG (1 model call)", key="plain_button"):
        with st.spinner("Asking the model with no sources…"):
            st.session_state[plain_key] = explain.without_rag(
                answer.question, model=model
            )

    plain = st.session_state.get(plain_key)
    if plain:
        left, right = st.columns(2)
        with left:
            st.markdown("**With retrieval**")
            st.write(answer.text)
            st.caption(f"{answer.latency_ms} ms · cites {answer.citations or 'nothing'}")
        with right:
            st.markdown("**Without retrieval**")
            st.write(plain["text"])
            st.caption(f"{plain['latency_ms']} ms · cites nothing")
        st.caption(
            "The no-RAG answer is not stored: it is an exhibit, not a product answer."
        )


# with evaluation_tab:
#     with st.expander("Evaluate current answer (ad-hoc)"):
#             st.caption("Saves a row to eval_results for this run. Uses gold-like fields if provided.")
#             with st.form("ad_hoc_eval"):
#                 label = st.text_input("Eval label", value=f"ui-{datetime.now().strftime('%Y%m%d-%H%M%S')}")
#                 expected_doc = st.text_input("Expected document (e.g. sample/rag-basics.md)", value="")
#                 expected_phrase = st.text_input("Expected phrase (optional)", value="")
#                 gold_answer = st.text_area("Gold answer (optional, helps the judge)", value="")
#                 use_j = st.checkbox("Use LLM judge", value=True)
#                 submitted_eval = st.form_submit_button("Save evaluation")
    
#             if submitted_eval and answer is not None:
#                 if not run or "id" not in run:
#                     st.warning("This answer isn't saved to runs yet. Ask the question first so it has a run_id.")
#                     st.stop()
    
#                 item = {
#                     "id": f"ui-{run['id']}",
#                     "question": answer.question,
#                     "answerable": bool(expected_doc.strip()),
#                     "expected_document": expected_doc.strip() or None,
#                     "expected_phrase": expected_phrase.strip() or None,
#                     "gold_answer": gold_answer.strip() or None,
#                 }
    
#                 answerable = bool(item["answerable"])
#                 hit = None
#                 rr = None
#                 if answerable and item["expected_document"]:
#                     hit = evaluate.metrics.is_hit(answer.hits, item["expected_document"], item["expected_phrase"]) if False else __import__("src.metrics", fromlist=[""]).is_hit(answer.hits, item["expected_document"], item["expected_phrase"])
#                     rr = __import__("src.metrics", fromlist=[""]).reciprocal_rank(answer.hits, item["expected_document"], item["expected_phrase"])
    
#                 verdict = judge.JudgeResult(faithfulness=None, correct=None)
#                 if use_j:
#                     if not answerable and answer.refused:
#                         verdict = judge.JudgeResult(faithfulness=None, correct=None)
#                     else:
#                         verdict = judge.judge_answer(
#                             item["question"],
#                             answer.text,
#                             judge.format_context(answer.hits),
#                             gold_answer=item.get("gold_answer"),
#                         )
    
#                 try:
#                     evaluate.save_row(
#                         label=label.strip() or f"ui-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
#                         item=item,
#                         answer=answer,
#                         run_id=run["id"],
#                         hit=hit,
#                         reciprocal_rank=rr,
#                         verdict=verdict,
#                         model=run.get("llm_model") or model,
#                         k=top_k,
#                         use_judge=use_j,
#                         use_agent=use_agent,
#                         agent_metrics=metrics if use_agent else None,
#                     )
#                     st.success(f"Saved to eval_results under label '{label}'")
#                 except Exception as exc:
#                     st.error(f"Failed to save evaluation: {exc}")

#     st.caption("Results from `eval_results`. Run evaluations via `python -m src.evaluate --label <name>`.")
#     try:
#         labels = evaluate.labels()
#     except Exception as exc:
#         st.error(f"Cannot load evaluation labels: {exc}")
#         labels = []

#     if not labels:
#         st.info("No evaluation runs found yet. Run `python -m src.evaluate --label baseline` first.")
#     else:
#         label_map = {}
#         for row in labels:
#             last = row.get("last_run")
#             last_str = last.strftime("%Y-%m-%d %H:%M") if last else "unknown"
#             label_map[row["label"]] = f"{row['label']} · {row['n']} q · {last_str}"

#         chosen = st.selectbox("Eval label", list(label_map.keys()), format_func=label_map.get)
#         if not chosen:
#             st.stop()

#         try:
#             summary = evaluate.summarize_label(chosen)
#         except Exception as exc:
#             st.error(f"Failed to summarize {chosen}: {exc}")
#             st.stop()

#         st.subheader("Summary")
#         c1, c2, c3, c4 = st.columns(4)
#         c1.metric("N", summary.get("n"))
#         c1.caption(f"{summary.get('n_answerable')} answerable / {summary.get('n', 0) - summary.get('n_answerable', 0)} unanswerable")
#         c2.metric("Hit@k", f"{summary['hit_at_k']:.3f}" if summary.get("hit_at_k") is not None else "n/a")
#         c3.metric("MRR", f"{summary['mrr']:.3f}" if summary.get("mrr") is not None else "n/a")
#         c4.metric("Faithfulness", f"{summary['faithfulness']:.3f}" if summary.get("faithfulness") is not None else "n/a")

#         c5, c6, c7, c8 = st.columns(4)
#         c5.metric("Correctness", f"{summary['correctness']:.3f}" if summary.get("correctness") is not None else "n/a")
#         c6.metric("Refusal recall", f"{summary['refusal_recall']:.3f}" if summary.get("refusal_recall") is not None else "n/a")
#         c7.metric("False refusal rate", f"{summary['false_refusal_rate']:.3f}" if summary.get("false_refusal_rate") is not None else "n/a")
#         c8.metric("p50 latency (ms)", f"{summary['p50_latency_ms']:.0f}" if summary.get("p50_latency_ms") is not None else "n/a")

#         astats = evaluate.agent_stats(chosen)
#         if astats:
#             st.divider()
#             st.subheader("Agent loop stats")
#             a1, a2, a3, a4 = st.columns(4)
#             a1.metric("Mean searches", f"{astats.get('mean_searches', 0):.2f}")
#             a2.metric("Mean decisions", f"{astats.get('mean_decisions', 0):.2f}")
#             a3.metric("Mean model calls", f"{astats.get('mean_calls', 0):.2f}")
#             a4.metric("Hit step limit", f"{astats.get('step_limit_rate', 0) * 100:.1f}%")
#             a5, a6, a7, a8 = st.columns(4)
#             a5.metric("Invalid JSON", f"{astats.get('invalid_json_rate', 0) * 100:.1f}%")
#             a6.metric("Below MIN_TOP_SCORE", f"{astats.get('below_threshold_rate', 0) * 100:.1f}%")

#         st.divider()
#         st.subheader("Per-question results")
#         try:
#             rows = evaluate.load_results(chosen)
#         except Exception as exc:
#             st.error(f"Failed to load results for {chosen}: {exc}")
#             rows = []

#         if rows:
#             table = []
#             for r in rows:
#                 details = r.get("details") or {}
#                 table.append(
#                     {
#                         "question_id": r["question_id"],
#                         "answerable": details.get("answerable"),
#                         "hit": r["hit"],
#                         "mrr": r["mrr"],
#                         "faithfulness": r["faithfulness"],
#                         "correct": r["correct"],
#                         "refused": r["refused"],
#                         "latency_ms": r["latency_ms"],
#                     }
#                 )
#             st.dataframe(table, hide_index=True, use_container_width=True)

#             with st.expander("Full details (JSON)"):
#                 st.json(jsonable(rows))
#         else:
#             st.caption("No rows for this label.")