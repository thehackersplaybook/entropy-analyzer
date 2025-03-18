import pytest
from entropy_analyzer.strategies.search_entropy import SearchEngineEntropy


def test_search_entropy_normal():
    analyzer = SearchEngineEntropy()
    similar_results = ["apple fruit healthy", "apple fruit good", "apple fruit tasty"]
    diverse_results = [
        "apple computer technology",
        "apple fruit healthy",
        "apple records music",
    ]
    assert analyzer.compute_entropy(similar_results) < analyzer.compute_entropy(
        diverse_results
    )


def test_search_entropy_edge_cases():
    analyzer = SearchEngineEntropy()
    assert analyzer.compute_entropy(None) == 0.0
    assert analyzer.compute_entropy([]) == 0.0
    assert analyzer.compute_entropy(["single result"]) < 1.0


def test_search_entropy_identical_results():
    analyzer = SearchEngineEntropy()
    assert analyzer.compute_entropy(["same"] * 3) == 0.0


def test_search_entropy_invalid_input():
    analyzer = SearchEngineEntropy()
    with pytest.raises(ValueError):
        analyzer.compute_entropy("not a list")
    with pytest.raises(ValueError):
        analyzer.compute_entropy([1, 2, 3])
    with pytest.raises(ValueError):
        analyzer.compute_entropy(["text", None, "text"])
