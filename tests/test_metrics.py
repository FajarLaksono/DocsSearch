import pytest

from src.metrics import is_hit, matches, mean, reciprocal_rank, summarize
from src.retrieve import Hit


def hit(document="sample/rag-basics.md", content="A chunk about chunking.", score=0.8):
    return Hit(
        chunk_id=1,
        document=document,
        chunk_index=0,
        content=content,
        score=score,
    )


def test_matches_document_and_phrase():
    assert matches(hit(), "sample/rag-basics.md", "about chunking")


def test_matches_requires_the_phrase():
    assert not matches(hit(), "sample/rag-basics.md", "about embeddings")


def test_matches_wrong_document():
    assert not matches(hit(), "sample/rag-evaluation.md", "about chunking")


def test_is_hit_needs_a_document():
    assert is_hit([hit()], None, None) is False


def test_reciprocal_rank_first():
    assert reciprocal_rank([hit()], "sample/rag-basics.md", "chunking") == 1.0


def test_reciprocal_rank_third():
    hits = [
        hit(document="a.md", content="nothing"),
        hit(document="b.md", content="nothing"),
        hit(document="sample/rag-basics.md", content="about chunking"),
    ]
    assert reciprocal_rank(hits, "sample/rag-basics.md", "chunking") == pytest.approx(1 / 3)


def test_reciprocal_rank_miss():
    assert reciprocal_rank([hit()], "sample/rag-evaluation.md", "gold set") == 0.0


def test_summarize_mixed_rows():
    rows = [
        {"answerable": True, "hit": True, "reciprocal_rank": 1.0,
         "refused": False, "faithfulness": 1.0, "correct": True, "latency_ms": 1000},
        {"answerable": True, "hit": False, "reciprocal_rank": 0.0,
         "refused": True, "faithfulness": 0.5, "correct": False, "latency_ms": 3000},
        {"answerable": False, "hit": None, "reciprocal_rank": None,
         "refused": True, "faithfulness": None, "correct": None, "latency_ms": 2000},
    ]
    summary = summarize(rows)
    assert summary["n"] == 3
    assert summary["n_answerable"] == 2
    assert summary["hit_at_k"] == 0.5
    assert summary["mrr"] == 0.5
    assert summary["refusal_recall"] == 1.0
    assert summary["false_refusal_rate"] == 0.5
    assert summary["faithfulness"] == 0.75
    assert summary["correctness"] == 0.5
    assert summary["p50_latency_ms"] == 2000


def test_summarize_ignores_missing_judge_values():
    rows = [
        {"answerable": True, "hit": True, "reciprocal_rank": 1.0,
         "refused": False, "faithfulness": None, "correct": None, "latency_ms": 1000},
        {"answerable": True, "hit": False, "reciprocal_rank": 0.0,
         "refused": False, "faithfulness": None, "correct": None, "latency_ms": 2000},
        {"answerable": False, "hit": None, "reciprocal_rank": None,
         "refused": True, "faithfulness": None, "correct": None, "latency_ms": 1500},
    ]
    summary = summarize(rows)
    assert summary["faithfulness"] is None
    assert summary["correctness"] is None
    assert summary["hit_at_k"] == 0.5
    assert summary["refusal_recall"] == 1.0
    assert summary["false_refusal_rate"] == 0.0


def test_summarize_empty():
    summary = summarize([])
    assert summary["n"] == 0
    assert summary["hit_at_k"] is None
    assert summary["mrr"] is None
    assert summary["p50_latency_ms"] is None


def test_mean_skips_none():
    assert mean([1.0, None, 0.0]) == 0.5
