import pytest

from src.judge import extract_json, faithfulness, parse_verdict


def fence(body: str, language: str = "json") -> str:
    """Build a markdown fence without typing one inside this file's own fence."""
    ticks = "`" * 3
    return f"{ticks}{language}\n{body}\n{ticks}"


def test_extract_json_plain():
    payload = extract_json('{"claims": [], "correct": true}')
    assert payload["correct"] is True


def test_extract_json_fenced():
    body = '{"claims": [{"claim": "x", "supported": true}], "correct": null}'
    assert len(extract_json(fence(body))["claims"]) == 1


def test_extract_json_from_prose():
    reply = 'Here is my verdict:\n{"claims": [], "correct": false}\nHope that helps.'
    assert extract_json(reply)["correct"] is False


def test_extract_json_rejects_garbage():
    with pytest.raises(ValueError):
        extract_json("I cannot grade this.")


def test_faithfulness_counts_supported_claims():
    claims = [
        {"claim": "a", "supported": True},
        {"claim": "b", "supported": True},
        {"claim": "c", "supported": False},
    ]
    assert faithfulness(claims) == 0.667
    assert faithfulness([]) is None


def test_parse_verdict_drops_correct_without_gold():
    payload = {"claims": [{"claim": "a", "supported": "true"}], "correct": "false"}
    result = parse_verdict(payload, has_gold=False)
    assert result.correct is None
    assert result.claims[0]["supported"] is True
    assert result.faithfulness == 1.0
