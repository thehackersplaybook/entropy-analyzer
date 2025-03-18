import pytest
import numpy as np
from entropy_analyzer.strategies.numerical_entropy import NumericalEntropy


def test_numerical_entropy_normal():
    analyzer = NumericalEntropy()
    assert 0 < analyzer.compute_entropy([1, 2, 3, 4, 5]) < 1
    assert analyzer.compute_entropy([1, 1, 1]) < analyzer.compute_entropy([1, 2, 3])


def test_numerical_entropy_edge_cases():
    analyzer = NumericalEntropy()
    assert analyzer.compute_entropy(None) == 0.0
    assert analyzer.compute_entropy([]) == 0.0
    assert analyzer.compute_entropy([1]) == 0.0


def test_numerical_entropy_mixed_types():
    analyzer = NumericalEntropy()
    assert 0 < analyzer.compute_entropy([1, 2.5, 3, 4.7]) < 1
    assert analyzer.compute_entropy([0, 0.0, 1, 1.0]) > 0.0


def test_numerical_entropy_invalid_input():
    analyzer = NumericalEntropy()
    with pytest.raises(ValueError):
        analyzer.compute_entropy("123")
    with pytest.raises(ValueError):
        analyzer.compute_entropy([1, "2", 3])
    with pytest.raises(ValueError):
        analyzer.compute_entropy([1, float("inf"), 3])
    with pytest.raises(ValueError):
        analyzer.compute_entropy([1, float("nan"), 3])
