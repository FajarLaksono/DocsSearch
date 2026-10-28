import pytest

from src import explain
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


def answer(text: str, hits: list[Hit], citations: list[int] | None = None) -> Answer:
    return Answer(
        question="q",
        text=text,
        hits=hits,
        citations=citations if citations is not None else [1],
        refused=False,
        latency_ms=1,
        tokens_in=1,
        tokens_out=1,
    )


def test_split_sentences_flattens_and_splits():
    text = "First one.\n\nSecond   one! Third one?"
    assert explain.split_sentences(text) == [
        "First one.",
        "Second one!",
        "Third one?",
    ]
    assert explain.split_sentences("   ") == []


def test_content_words_drops_short_and_stop_words():
    assert explain.content_words("The cat is at a window") == {"cat", "window"}


def test_overlap_is_containment():
    sentence = "Hit rate measures retrieval quality"
    source = "Hit rate measures retrieval quality across the gold set and more"
    assert explain.overlap(sentence, source) == 1.0
    assert explain.overlap("Hit rate measures retrieval precision", source) == 0.8
    assert explain.overlap("the is at", source) == 0.0


def test_similarity_is_symmetric_and_range_bound():
    left = "The gate refuses below the floor."
    right = "The gate refuses when the score is below the floor."
    assert explain.similarity(left, right) == explain.similarity(right, left)
    assert 0.0 < explain.similarity(left, right) < 1.0
    assert explain.similarity(left, left) == 1.0


def test_highlight_ranks_best_first():
    hits = [
        hit(1, 0.81, content="Unrelated text about chunk overlap. Chunking is quiet."),
        hit(2, 0.60, content="A retrieval miss means the answer is not in the top chunks."),
    ]
    rows = explain.highlight(
        "A retrieval miss means the answer is missing.", hits
    )

    assert len(rows) == 1
    assert rows[0]["matches"][0]["rank"] == 2
    assert rows[0]["matches"][0]["score"] >= 0.8


def test_highlight_caps_matches_per_sentence():
    content = " ".join(
        f"Chunk overlap number {n} keeps context across the boundary." for n in range(6)
    )
    rows = explain.highlight("Chunk overlap keeps context.", [hit(1, content=content)])

    assert len(rows) == 1
    assert len(rows[0]["matches"]) == 3


def test_highlight_html_escapes_and_marks():
    content = "Ignore <script>alert(1)</script>. The gate refuses."
    html = explain.highlight_html(content, ["The gate refuses."])

    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "<mark>The gate refuses.</mark>" in html


def test_leave_one_out_flags_the_chunk_that_mattered(monkeypatch):
    hits = [hit(1, 0.82), hit(2, 0.41)]
    original = answer("The gate refuses any answer below the configured floor [1].", hits)
    replies = {
        1: "Chunking decides what can be retrieved at all [1].",
        2: "The gate refuses any answer below the configured score floor [1].",
    }

    def fake_answer_from_hits(question, subset, *, model=None):
        removed = {item.chunk_id for item in hits} - {
            item.chunk_id for item in subset
        }
        return answer(replies[removed.pop()], subset)

    monkeypatch.setattr(explain, "answer_from_hits", fake_answer_from_hits)

    rows = explain.leave_one_out(original.question, original, hits)

    assert [row.removed_rank for row in rows] == [1, 2]
    assert rows[0].changed is True
    assert rows[1].changed is False
    assert rows[0].kept < rows[1].kept


def test_leave_one_out_skips_a_single_hit():
    only = [hit(1)]
    assert explain.leave_one_out("q", answer("text [1].", only), only) == []


def test_without_rag_has_no_sources(monkeypatch):
    captured: dict = {}

    class FakeClient:
        def generate(self, messages, **kwargs):
            captured["messages"] = messages
            captured["kwargs"] = kwargs
            return {
                "text": "  France won.  ",
                "tokens_in": 12,
                "tokens_out": 3,
                "done_reason": "stop",
            }

    monkeypatch.setattr(explain.llm, "get_client", lambda: FakeClient())

    result = explain.without_rag("Who won the 2018 World Cup?")

    assert result["text"] == "France won."
    assert result["tokens_out"] == 3
    assert len(captured["messages"]) == 2
    assert "no sources" in captured["messages"][0]["content"].lower()
    assert captured["messages"][1]["content"] == "Who won the 2018 World Cup?"
