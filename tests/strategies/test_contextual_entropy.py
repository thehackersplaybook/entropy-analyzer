import pytest
from unittest.mock import patch, MagicMock
from entropy_analyzer.strategies.contextual_entropy import ContextualEntropy


def test_contextual_entropy_edge_cases():
    analyzer = ContextualEntropy()
    assert analyzer.compute_entropy(None) == 0.0
    assert analyzer.compute_entropy("") == 0.0


def test_contextual_entropy_invalid_input():
    analyzer = ContextualEntropy()
    with pytest.raises(ValueError):
        analyzer.compute_entropy(123)
    with pytest.raises(ValueError):
        analyzer.compute_entropy(["text"])


@patch("openai.OpenAI")
def test_contextual_entropy_api_success(mock_openai):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="5"))]
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai.return_value = mock_client

    analyzer = ContextualEntropy()
    score = analyzer.compute_entropy("test text")
    assert 0 <= score <= 1
    assert score == 0.5


@patch("openai.OpenAI")
def test_contextual_entropy_api_failure(mock_openai):
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception("API Error")
    mock_openai.return_value = mock_client

    analyzer = ContextualEntropy()
    score = analyzer.compute_entropy("test text")
    assert 0 <= score <= 1  # Should fall back to TextEntropy
