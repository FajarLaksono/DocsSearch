import pytest

from src import agent, trace
from src.generate import Answer
from src.retrieve import Hit


def hit(
    chunk_id: int = 1,
    score: float = 0.9,
    document: str = "sample/rag-basics.md",
    chunk_index: int = 0,
    content: str = "text",
) -> Hit:
    return Hit(
        chunk_id=chunk_id,
        document=document,
        chunk_index=chunk_index,
        content=content,
        score=score,
    )


class FakeClient:
    """Returns canned replies; records the messages it was shown."""

    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.calls: list[dict] = []

    def generate(self, messages, **kwargs):
        self.calls.append({"messages": messages, **kwargs})
        reply = self.replies.pop(0)
        return {
            "text": reply,
            "tokens_in": 10,
            "tokens_out": 5,
            "done_reason": "stop",
        }


def install(monkeypatch, replies, *, hits=None, text="The answer [1].", refused=False):
    client = FakeClient(replies)
    captured: dict = {}

    monkeypatch.setattr(agent.llm, "get_client", lambda: client)
    monkeypatch.setattr(
        agent.tools, "search_docs", lambda query, k=None: list(hits or [])
    )

    def fake_answer(question, hit_list, *, model=None):
        captured["question"] = question
        captured["hits"] = list(hit_list)
        return Answer(
            question,
            text,
            list(hit_list),
            [1] if hit_list and not refused else [],
            refused,
            7,
            3,
            2,
        )

    monkeypatch.setattr(agent, "answer_from_hits", fake_answer)
    return client, captured


def test_parse_action_reads_search():
    action = agent.parse_action(
        '{"thought": "  try the document words  ", '
        '"action": "search_docs", "input": "  hit   rate "}'
    )
    assert action.name == "search_docs"
    assert action.input == "hit rate"
    assert action.thought == "try the document words"


def test_parse_action_reads_answer():
    action = agent.parse_action('{"action": "answer"}')
    assert action.name == "answer"
    assert action.input == ""
    assert action.thought == ""


def test_parse_action_rejects_unknown_and_empty():
    with pytest.raises(ValueError, match="unknown action"):
        agent.parse_action('{"action": "calculator", "input": "1+1"}')
    with pytest.raises(ValueError, match="non-empty"):
        agent.parse_action('{"action": "search_docs", "input": "   "}')
    with pytest.raises(ValueError, match="JSON"):
        agent.parse_action("I would search for that.")


def test_merge_hits_dedupes_sorts_and_caps():
    existing = [hit(1, 0.40), hit(2, 0.80)]
    found = [hit(2, 0.90, content="better copy"), hit(3, 0.20), hit(4, 0.60)]

    merged = agent.merge_hits(existing, found, k=2)

    assert [h.chunk_id for h in merged] == [2, 4]
    assert merged[0].content == "better copy"
    assert merged[0].score == 0.9


def test_search_then_answer(monkeypatch):
    client, captured = install(
        monkeypatch,
        [
            '{"thought": "look up the term", "action": "search_docs", '
            '"input": "  hit   rate "}',
            '{"thought": "the first source defines it", "action": "answer", '
            '"input": ""}',
        ],
        hits=[hit(1, 0.81)],
    )

    result = agent.run_agent("What is hit rate at k?", thinking="prompted")

    assert len(client.calls) == 2
    assert captured["hits"][0].chunk_id == 1
    assert [step["type"] for step in result.steps] == [
        "thought",
        "tool_call",
        "observation",
        "thought",
        "tool_call",
    ]
    assert result.steps[1]["input"] == "hit rate"
    assert result.metrics["reason"] == "answered"
    assert result.metrics["stop_reason"] == "answer"
    assert result.metrics["searches"] == 1
    assert result.metrics["steps"] == 2
    assert result.metrics["llm_calls"] == 3
    assert result.answer.tokens_in == 23
    assert result.answer.tokens_out == 12
    assert result.answer.text == "The answer [1]."


def test_invalid_json_is_retried(monkeypatch):
    client, captured = install(
        monkeypatch,
        [
            '{"action": "search_docs", "input": "hit rate"}',
            "Sure! I would search again.",
            '{"action": "answer", "input": ""}',
        ],
        hits=[hit(1, 0.81)],
    )

    result = agent.run_agent("What is hit rate at k?", thinking="off")

    assert len(client.calls) == 3
    assert result.metrics["stop_reason"] == "answer"
    assert result.metrics["steps"] == 2
    assert any(step.get("error") for step in result.steps)
    assert "could not be parsed" in client.calls[2]["messages"][1]["content"]
    assert captured["hits"][0].chunk_id == 1


def test_invalid_json_twice_falls_back(monkeypatch):
    client, captured = install(
        monkeypatch,
        [
            '{"action": "search_docs", "input": "hit rate"}',
            "not json",
            "still not json",
        ],
        hits=[hit(1, 0.81)],
    )

    result = agent.run_agent("What is hit rate at k?", thinking="off")

    assert len(client.calls) == 3
    assert result.metrics["stop_reason"] == "invalid_json"
    assert result.metrics["steps"] == 1
    assert result.metrics["reason"] == "answered"
    assert captured["hits"][0].chunk_id == 1
    assert sum(1 for step in result.steps if step.get("error")) == 2


def test_below_threshold_refuses_without_answer_call(monkeypatch):
    client, captured = install(
        monkeypatch,
        [
            '{"action": "search_docs", "input": "fifa"}',
            '{"action": "answer", "input": ""}',
        ],
        hits=[hit(1, 0.20)],
    )

    result = agent.run_agent("Who won the 2018 World Cup?", thinking="off")

    assert len(client.calls) == 2
    assert captured == {}
    assert result.answer.refused is True
    assert result.answer.citations == []
    assert result.answer.text == "I don't know based on the provided documents."
    assert result.metrics["reason"] == "below_min_top_score"
    assert result.metrics["stop_reason"] == "answer"


def test_step_limit_stops_searching(monkeypatch):
    client, captured = install(
        monkeypatch,
        [
            '{"action": "search_docs", "input": "first"}',
            '{"action": "search_docs", "input": "second"}',
            '{"action": "search_docs", "input": "third"}',
        ],
        hits=[hit(1, 0.81)],
    )

    result = agent.run_agent("What is hit rate at k?", max_steps=2, thinking="off")

    assert len(client.calls) == 2
    assert result.metrics["searches"] == 2
    assert result.metrics["steps"] == 2
    assert result.metrics["stop_reason"] == "step_limit"
    assert result.metrics["reason"] == "answered"
    assert captured["hits"][0].chunk_id == 1


def test_thinking_off_records_no_thought(monkeypatch):
    client, _ = install(
        monkeypatch,
        [
            '{"thought": "hidden", "action": "search_docs", "input": "hit rate"}',
            '{"thought": "hidden too", "action": "answer", "input": ""}',
        ],
        hits=[hit(1, 0.81)],
    )

    result = agent.run_agent("What is hit rate at k?", thinking="off")

    assert not any(step["type"] == "thought" for step in result.steps)
    assert result.metrics["thinking"] == "off"
    assert 'omit the "thought"' in client.calls[0]["messages"][0]["content"]


def test_answer_from_run_keeps_stored_order(monkeypatch):
    steps = [
        {
            "type": "observation",
            "query": "first",
            "chunks": [
                {"chunk_id": 4, "document": "d.md", "chunk_index": 1, "score": 0.5},
                {"chunk_id": 9, "document": "d.md", "chunk_index": 2, "score": 0.9},
            ],
        },
        {
            "type": "answer",
            "text": "nine then four [1][2]",
            "citations": [1, 2],
            "refused": False,
            "chunks": [9, 4],
        },
    ]
    run = {
        "question": "q",
        "answer": "nine then four [1][2]",
        "steps": steps,
        "latency_ms": 10,
        "tokens_in": 1,
        "tokens_out": 2,
    }
    monkeypatch.setattr(trace, "_chunk_contents", lambda ids: {9: "nine", 4: "four"})

    answer = trace.answer_from_run(run)

    assert [h.chunk_id for h in answer.hits] == [9, 4]
    assert [h.content for h in answer.hits] == ["nine", "four"]
