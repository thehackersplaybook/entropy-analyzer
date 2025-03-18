import pytest
from unittest.mock import Mock, AsyncMock
from openai import OpenAI, AsyncOpenAI
from entropy_analyzer.system.llm import LLM, AsyncLLM, ChatInput, Message, ChatResponse

# Test data
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


@pytest.fixture
def mock_openai():
    client = Mock(spec=OpenAI)
    client.chat = Mock()
    client.chat.completions = Mock()
    client.chat.completions.create = Mock(return_value=MOCK_API_RESPONSE)
    return client


@pytest.fixture
def mock_async_openai():
    client = AsyncMock(spec=AsyncOpenAI)
    client.chat = AsyncMock()
    client.chat.completions = AsyncMock()
    client.chat.completions.create = AsyncMock(return_value=MOCK_API_RESPONSE)
    return client


class TestLLM:
    def test_successful_response_generation(self, mock_openai):
        llm = LLM(client=mock_openai)
        response = llm.generate_response(SAMPLE_CHAT_INPUT)

        assert isinstance(response, ChatResponse)
        assert response.model == "gpt-3.5-turbo"
        assert len(response.choices) == 1
        mock_openai.chat.completions.create.assert_called_once()

    def test_input_validation(self, mock_openai):
        llm = LLM(client=mock_openai)

        # Test invalid temperature
        invalid_input = SAMPLE_CHAT_INPUT.copy()
        invalid_input["temperature"] = 3.0

        with pytest.raises(ValueError):
            llm.generate_response(invalid_input)

    def test_api_error_handling(self, mock_openai):
        mock_openai.chat.completions.create.side_effect = Exception("API Error")
        llm = LLM(client=mock_openai)

        with pytest.raises(RuntimeError, match="Failed to generate chat completion"):
            llm.generate_response(SAMPLE_CHAT_INPUT)

    def test_verbose_logging(self, mock_openai, caplog):
        llm = LLM(client=mock_openai, verbose=True)
        llm.generate_response(SAMPLE_CHAT_INPUT)

        assert len(caplog.records) > 0


class TestAsyncLLM:
    @pytest.mark.asyncio
    async def test_successful_async_response_generation(self, mock_async_openai):
        async_llm = AsyncLLM(client=mock_async_openai)
        response = await async_llm.generate_response(SAMPLE_CHAT_INPUT)

        assert isinstance(response, ChatResponse)
        assert response.model == "gpt-3.5-turbo"
        assert len(response.choices) == 1
        mock_async_openai.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_input_validation(self, mock_async_openai):
        async_llm = AsyncLLM(client=mock_async_openai)

        # Test invalid presence_penalty
        invalid_input = SAMPLE_CHAT_INPUT.copy()
        invalid_input["presence_penalty"] = 3.0

        with pytest.raises(ValueError):
            await async_llm.generate_response(invalid_input)

    @pytest.mark.asyncio
    async def test_async_api_error_handling(self, mock_async_openai):
        mock_async_openai.chat.completions.create.side_effect = Exception("API Error")
        async_llm = AsyncLLM(client=mock_async_openai)

        with pytest.raises(RuntimeError, match="Failed to generate chat completion"):
            await async_llm.generate_response(SAMPLE_CHAT_INPUT)

    def test_chat_input_model_validation(self):
        # Test message validation
        valid_message = Message(role="user", content="test")
        assert valid_message.role == "user"

        # Test ChatInput validation
        valid_input = ChatInput(
            messages=[valid_message], model="gpt-3.5-turbo", temperature=0.7
        )
        assert valid_input.model == "gpt-3.5-turbo"
        assert len(valid_input.messages) == 1

    def test_edge_cases(self, mock_openai):
        llm = LLM(client=mock_openai)

        # Test empty messages list
        with pytest.raises(ValueError):
            llm.generate_response({"messages": [], "model": "gpt-3.5-turbo"})

        # Test with all optional parameters
        complex_input = {
            "messages": [SAMPLE_MESSAGE],
            "model": "gpt-3.5-turbo",
            "temperature": 0.7,
            "max_tokens": 100,
            "top_p": 0.9,
            "frequency_penalty": 0.5,
            "presence_penalty": 0.5,
            "stop": ["stop"],
            "user": "test_user",
        }
        response = llm.generate_response(complex_input)
        assert isinstance(response, ChatResponse)
