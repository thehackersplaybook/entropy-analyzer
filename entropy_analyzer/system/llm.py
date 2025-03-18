"""
LLM Module.

This module provides a clean facade over OpenAI's chat completion API with comprehensive
parameter support, type safety, and error handling. It simplifies the interaction with
language models while maintaining full control over the underlying API parameters.

The module is designed to be:
- Type-safe: Using Pydantic models for request/response validation.
- Comprehensive: Supporting all non-streaming chat completion parameters.
- Error-resistant: Built-in error handling and logging.
- Developer-friendly: Clear documentation and intuitive interface.

Author: Aditya Patange (AdiPat)
License: MIT
"""

import traceback
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field
from openai import OpenAI, AsyncOpenAI
from .printer import Printer


class Message(BaseModel):
    """
    Chat message model for the LLM module.

    A structured representation of a chat message with role and content information.
    Used for maintaining a consistent format for message exchanges with the language model.

    Attributes:
        role (str): The role of the message sender (e.g., 'user', 'assistant', 'system').
        content (str): The text content of the message.
    """

    role: str
    content: str


class ChatInput(BaseModel):
    """
    Chat input model for the LLM module.

    Attributes:
    - messages: A list of chat messages.
    - model: The language model to use.
    - frequency_penalty: Frequency penalty for the model.
    - presence_penalty: Presence penalty for the model.
    - max_tokens: Maximum number of tokens to generate.
    - n: Number of completions to generate.
    - temperature: Sampling temperature for the model.
    - top_p: Top-p sampling parameter.
    - stop: Stop sequence for the model.
    - tool_choice: Tool choice for the model.
    - tools: List of tools for the model.
    - user: User identifier.
    - seed: Random seed for the model.
    - response_format: Response format for the model.
    - logit_bias: Logit bias for the model.
    - metadata: Metadata for the model.
    """

    messages: List[Message]
    model: str
    frequency_penalty: Optional[float] = Field(0, ge=-2.0, le=2.0)
    presence_penalty: Optional[float] = Field(0, ge=-2.0, le=2.0)
    max_tokens: Optional[int] = None
    n: Optional[int] = 1
    temperature: Optional[float] = Field(1, ge=0, le=2)
    top_p: Optional[float] = Field(1, ge=0, le=1)
    stop: Optional[Union[str, List[str]]] = None
    tool_choice: Optional[Union[str, Dict]] = None
    tools: Optional[List[Dict]] = None
    user: Optional[str] = None
    seed: Optional[int] = None
    response_format: Optional[Dict] = None
    logit_bias: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, str]] = None


class ChatResponse(BaseModel):
    """
    Chat response model for the EasyLLM module.

    Attributes:
    - choices: List of chat completion choices.
    - created: Timestamp of the response creation.
    - model: The language model used for the response.
    - usage: Usage statistics for the response.
    """

    choices: List[Dict]
    created: int
    model: str
    usage: Dict


class BaseLLM:
    """
    Base class for LLM implementations with shared functionality.

    This class provides core functionality for both synchronous and asynchronous LLM implementations.
    It handles input validation, response formatting, logging, and error management.

    Attributes:
        verbose (bool): Flag to enable detailed logging of operations.
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize a new LLM instance.

        Args:
            verbose: Flag to enable verbose logging.
        """
        self.verbose = verbose

    def _prepare_request(self, input_data: Union[ChatInput, Dict]) -> ChatInput:
        """
        Prepare and validate the chat input request.

        Converts dictionary input to ChatInput model if needed and ensures all data is valid.

        Args:
            input_data: Raw input data either as a ChatInput instance or dictionary.

        Returns:
            ChatInput: A validated ChatInput instance ready for API submission.

        Raises:
            ValueError: If input data fails validation requirements.
        """
        return (
            ChatInput(**input_data)
            if not isinstance(input_data, ChatInput)
            else input_data
        )

    def _prepare_structured_request(
        self, input_data: Union[ChatInput, Dict], response_format: BaseModel
    ) -> ChatInput:
        """
        Prepare and validate the chat input request for structured outputs.

        Args:
            input_data: Raw input data either as a ChatInput instance or dictionary.
            response_format: A Pydantic model defining the response structure.

        Returns:
            ChatInput: A validated ChatInput instance ready for API submission.

        Raises:
            ValueError: If input data fails validation requirements.
        """
        return self._prepare_request(input_data)

    def _create_response(self, response: Any) -> ChatResponse:
        """
        Create a validated chat completion response object.

        Transforms the raw API response into a structured ChatResponse model.

        Args:
            response: Raw response data from the OpenAI API.

        Returns:
            ChatResponse: A validated response object containing completion data.

        Raises:
            ValueError: If response data cannot be properly formatted.
        """
        response_dict = {
            "choices": [{"message": c.message.model_dump()} for c in response.choices],
            "created": response.created,
            "model": response.model,
            "usage": response.usage.model_dump(),
        }
        return ChatResponse(**response_dict)

    def _log_request(self, chat_input: ChatInput) -> None:
        """
        Log details about the chat completion request.

        Provides detailed logging of request parameters when verbose mode is enabled.

        Args:
            chat_input: The validated chat input being sent to the API.
        """
        Printer.verbose_logger(
            self.verbose,
            Printer.print_light_grey_message,
            f"Generating chat completion with model: {chat_input.model}",
        )

    def _log_response(self, chat_response: ChatResponse) -> None:
        """
        Log details about the chat completion response.

        Provides detailed logging of response data when verbose mode is enabled.

        Args:
            chat_response: The validated response received from the API.
        """
        Printer.verbose_logger(
            self.verbose,
            Printer.print_light_grey_message,
            f"Successfully generated response: {chat_response.model_dump()}",
        )

    def _handle_error(self, error: Exception, context: str) -> None:
        """
        Handle errors during chat completion generation.

        Args:
            error: Exception raised during completion generation.
            context: Context of the error.

        Returns:
            None

        Raises:
            ValueError: If the error context is related to input validation.
            RuntimeError: If the error context is related to completion generation
            Exception: For any other unexpected errors.
        """
        error_msg = f"{context}: {str(error)}"
        Printer.print_red_message(error_msg)
        traceback.print_exc()
        raise (
            ValueError(error_msg)
            if context.startswith("Error in")
            else RuntimeError(error_msg)
        )


class LLM(BaseLLM):
    """
    A facade class for OpenAI's synchronous chat completion API.

    Provides a simplified interface for making synchronous chat completion requests
    while maintaining full control over API parameters and error handling.

    Attributes:
        client (OpenAI): OpenAI client for making API requests.
        verbose (bool): Inherited flag for detailed logging.
    """

    def __init__(self, client: OpenAI, verbose: bool = False):
        super().__init__(verbose)
        self.client = client

    def generate_response(self, input_data: Union[ChatInput, Dict]) -> ChatResponse:
        """
        Generate a chat completion response synchronously.

        Processes the input data, makes the API request, and returns a validated response.

        Args:
            input_data: Either a ChatInput instance or a dictionary matching ChatInput schema.

        Returns:
            ChatResponse: A validated response containing the completion data.

        Raises:
            ValueError: If input validation fails.
            RuntimeError: If API request fails or response processing fails.
            Exception: For any other unexpected errors during execution.
        """
        try:
            chat_input = self._prepare_request(input_data)
            self._log_request(chat_input)

            try:
                response = self.client.chat.completions.create(
                    **chat_input.model_dump(exclude_none=True)
                )
                chat_response = self._create_response(response)
                self._log_response(chat_response)
                return chat_response
            except Exception as e:
                self._handle_error(e, "Failed to generate chat completion")
        except Exception as e:
            self._handle_error(e, "Error in generate_response")

    def generate_response_structured(
        self, input_data: Union[ChatInput, Dict], response_format: BaseModel
    ) -> Any:
        """
        Generate a structured chat completion response synchronously.

        Processes the input data, makes the API request, and returns a validated response
        that matches the provided Pydantic model schema.

        Args:
            input_data: Either a ChatInput instance or a dictionary matching ChatInput schema.
            response_format: A Pydantic model defining the response structure.

        Returns:
            Any: A validated response object matching the provided schema.

        Raises:
            ValueError: If input validation fails.
            RuntimeError: If API request fails or response processing fails.
            Exception: For any other unexpected errors during execution.
        """
        try:
            chat_input = self._prepare_structured_request(input_data, response_format)
            self._log_request(chat_input)

            try:
                response = self.client.chat.completions.parse(
                    **chat_input.model_dump(exclude_none=True),
                    response_format=response_format,
                )
                return response.choices[0].message.parsed
            except Exception as e:
                self._handle_error(e, "Failed to generate structured chat completion")
        except Exception as e:
            self._handle_error(e, "Error in generate_response_structured")


class AsyncLLM(BaseLLM):
    """
    A facade class for OpenAI's asynchronous chat completion API.

    Provides a simplified interface for making asynchronous chat completion requests
    while maintaining full control over API parameters and error handling.

    Attributes:
        client (AsyncOpenAI): AsyncOpenAI client for making async API requests.
        verbose (bool): Inherited flag for detailed logging.
    """

    def __init__(self, client: AsyncOpenAI, verbose: bool = False):
        super().__init__(verbose)
        self.client = client

    async def generate_response(
        self, input_data: Union[ChatInput, Dict]
    ) -> ChatResponse:
        """
        Generate a chat completion response asynchronously.

        Processes the input data, makes the async API request, and returns a validated response.

        Args:
            input_data: Either a ChatInput instance or a dictionary matching ChatInput schema.

        Returns:
            ChatResponse: A validated response containing the completion data.

        Raises:
            ValueError: If input validation fails.
            RuntimeError: If API request fails or response processing fails.
            Exception: For any other unexpected errors during execution.
        """
        try:
            chat_input = self._prepare_request(input_data)
            self._log_request(chat_input)

            try:
                response = await self.client.chat.completions.create(
                    **chat_input.model_dump(exclude_none=True)
                )
                chat_response = self._create_response(response)
                self._log_response(chat_response)
                return chat_response
            except Exception as e:
                self._handle_error(e, "Failed to generate chat completion")
        except Exception as e:
            self._handle_error(e, "Error in generate_response")

    async def generate_response_structured(
        self, input_data: Union[ChatInput, Dict], response_format: BaseModel
    ) -> Any:
        """
        Generate a structured chat completion response asynchronously.

        Processes the input data, makes the async API request, and returns a validated response
        that matches the provided Pydantic model schema.

        Args:
            input_data: Either a ChatInput instance or a dictionary matching ChatInput schema.
            response_format: A Pydantic model defining the response structure.

        Returns:
            Any: A validated response object matching the provided schema.

        Raises:
            ValueError: If input validation fails.
            RuntimeError: If API request fails or response processing fails.
            Exception: For any other unexpected errors during execution.
        """
        try:
            chat_input = self._prepare_structured_request(input_data, response_format)
            self._log_request(chat_input)

            try:
                response = await self.client.chat.completions.parse(
                    **chat_input.model_dump(exclude_none=True),
                    response_format=response_format,
                )
                return response.choices[0].message.parsed
            except Exception as e:
                self._handle_error(e, "Failed to generate structured chat completion")
        except Exception as e:
            self._handle_error(e, "Error in generate_response_structured")
