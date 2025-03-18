import pytest
from unittest.mock import patch, MagicMock, Mock
from pydantic import BaseModel
from openai import OpenAI
from entropy_analyzer.strategies.contextual_entropy import ContextualEntropy


SAMPLE_MESSAGE = {"role": "user", "content": "Hello"}
SAMPLE_CHAT_INPUT = {"messages": [SAMPLE_MESSAGE], "model": "gpt-3.5-turbo"}

MOCK_API_RESPONSE = Mock(
    choices=[
        Mock(
            message=Mock(
                model_dump=lambda: {"role": "assistant", "content": "Hi there!"}
            )
        )
    ],
    created=1234567890,
    model="gpt-3.5-turbo",
    usage=Mock(model_dump=lambda: {"total_tokens": 10}),
)


class TestSchema(BaseModel):
    name: str
    value: int
    tags: list[str]


MOCK_STRUCTURED_RESPONSE = Mock(
    choices=[
        Mock(
            message=Mock(parsed={"name": "test", "value": 42, "tags": ["tag1", "tag2"]})
        )
    ]
)


@pytest.fixture
def mock_openai():
    client = Mock(spec=OpenAI)
    client.chat = Mock()
    client.chat.completions = Mock()
    client.chat.completions.create = Mock(return_value=MOCK_API_RESPONSE)
    return client


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
def test_contextual_entropy_api_success(mock_openai_cls):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MOCK_API_RESPONSE
    mock_openai_cls.return_value = mock_client
    analyzer = ContextualEntropy()
    score = analyzer.compute_entropy("test text", client=mock_client)
    assert 0 <= score <= 1
    mock_client.chat.completions.create.assert_called_once()


@patch("openai.OpenAI")
def test_contextual_entropy_api_failure(mock_openai_cls):
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception("API Error")
    mock_openai_cls.return_value = mock_client

    analyzer = ContextualEntropy()
    score = analyzer.compute_entropy("test text", client=mock_client)
    assert 0 <= score <= 1
