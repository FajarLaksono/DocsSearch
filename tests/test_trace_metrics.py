from trace import compute_metrics


def test_metrics_with_several_scores():
    metrics = compute_metrics([0.80, 0.60, 0.40], [1, 2], refused=False)
    assert metrics["top_score"] == 0.8
    assert metrics["mean_score"] == 0.6
    assert metrics["score_gap"] == 0.2
    assert metrics["cited_sources"] == 2
    assert metrics["refused"] is False


def test_metrics_with_one_score_has_no_gap():
    metrics = compute_metrics([0.42], [], refused=False)
    assert metrics["top_score"] == 0.42
    assert metrics["mean_score"] == 0.42
    assert metrics["score_gap"] is None


def test_metrics_with_no_scores():
    metrics = compute_metrics([], [], refused=True)
    assert metrics["top_score"] is None
    assert metrics["mean_score"] is None
    assert metrics["score_gap"] is None
    assert metrics["cited_sources"] == 0
    assert metrics["refused"] is True


def test_metrics_round_to_six_places():
    metrics = compute_metrics([0.1234567, 0.1000001], [1], refused=False)
    assert metrics["top_score"] == 0.123457
    assert metrics["score_gap"] == 0.023457
    assert metrics["mean_score"] == 0.111728
