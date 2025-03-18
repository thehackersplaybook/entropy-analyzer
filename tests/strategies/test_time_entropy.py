import pytest
from entropy_analyzer.strategies.time_entropy import TimeEntropy


def test_time_entropy_normal():
    analyzer = TimeEntropy()
    regular_times = [
        "2023-01-01T12:00:00",
        "2023-01-01T13:00:00",
        "2023-01-01T14:00:00",
    ]
    irregular_times = [
        "2023-01-01T12:00:00",
        "2023-01-01T12:05:00",
        "2023-01-01T14:00:00",
    ]
    assert analyzer.compute_entropy(regular_times) < analyzer.compute_entropy(
        irregular_times
    )


def test_time_entropy_edge_cases():
    analyzer = TimeEntropy()
    assert analyzer.compute_entropy(None) == 0.0
    assert analyzer.compute_entropy([]) == 0.0
    assert analyzer.compute_entropy(["2023-01-01T12:00:00"]) == 0.0


def test_time_entropy_same_times():
    analyzer = TimeEntropy()
    same_times = ["2023-01-01T12:00:00"] * 3
    assert analyzer.compute_entropy(same_times) == 0.0


def test_time_entropy_invalid_input():
    analyzer = TimeEntropy()
    with pytest.raises(ValueError):
        analyzer.compute_entropy("2023-01-01T12:00:00")
    with pytest.raises(ValueError):
        analyzer.compute_entropy([123, "2023-01-01T12:00:00"])
    with pytest.raises(ValueError):
        analyzer.compute_entropy(["invalid_time"])
