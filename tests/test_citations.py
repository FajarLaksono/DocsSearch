from src.generate import is_refusal, parse_citations


def test_parse_citations_returns_ranks_in_order():
    assert parse_citations("Hit rate is [1] and MRR is [3].", 3) == [1, 3]


def test_parse_citations_drops_out_of_range():
    assert parse_citations("See [0], [1], and [9].", 2) == [1]


def test_parse_citations_dedupes():
    assert parse_citations("[2] then [2] again, then [1]", 2) == [2, 1]


def test_parse_citations_handles_no_citations():
    assert parse_citations("No numbers here.", 4) == []


def test_refusal_detection():
    assert is_refusal("I don't know based on the provided documents.")
    assert is_refusal("I do not know.")
    assert not is_refusal("Hit rate measures whether the correct source is retrieved.")
