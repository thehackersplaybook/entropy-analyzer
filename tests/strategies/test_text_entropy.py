import pytest
from entropy_analyzer.strategies.text_entropy import TextEntropy


def test_text_entropy_normal():
    analyzer = TextEntropy()
    assert 0 < analyzer.compute_entropy("hello world") < 1
    assert analyzer.compute_entropy("aaaa") < analyzer.compute_entropy("abcd")


def test_text_entropy_edge_cases():
    analyzer = TextEntropy()
    assert analyzer.compute_entropy(None) == 0.0
    assert analyzer.compute_entropy("") == 0.0
    assert analyzer.compute_entropy(" ") == 0.0


def test_text_entropy_special_chars():
    analyzer = TextEntropy()
    assert analyzer.compute_entropy("!@#$%^&*()") > 0.0
    assert analyzer.compute_entropy("αβγδε") > 0.0
    assert analyzer.compute_entropy("👍🎉🔥") > 0.0


def test_text_entropy_invalid_input():
    analyzer = TextEntropy()
    with pytest.raises(ValueError):
        analyzer.compute_entropy(123)
    with pytest.raises(ValueError):
        analyzer.compute_entropy(["text"])
