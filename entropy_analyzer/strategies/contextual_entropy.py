"""
Contextual Entropy Module.

This module implements entropy analysis using language models to evaluate text complexity
and unpredictability. It leverages GPT models to provide semantic-aware entropy scoring
with a fallback to basic text entropy when needed.

The module is designed to be:
- Intelligent: Uses LLM for contextual understanding.
- Fault-tolerant: Includes fallback mechanisms.
- Normalized: Produces scores between 0 and 1.

Author: Aditya Patange (AdiPat)
License: MIT
"""

from typing import Optional

from openai import OpenAI

from .entropy_strategy_base import EntropyStrategy
from .text_entropy import TextEntropy


class ContextualEntropy(EntropyStrategy):
    """Strategy for analyzing text entropy using contextual language models.

    This strategy leverages GPT models to analyze the unpredictability and
    semantic complexity of text content. It considers semantic meaning,
    linguistic patterns, and contextual relationships beyond simple character
    frequencies.

    The analysis is performed by querying a language model to rate the text's
    unpredictability. If the API call fails, it falls back to basic text
    entropy calculation.

    Attributes:
        None

    Example:
        >>> analyzer = ContextualEntropy()
        >>> score = analyzer.compute_entropy("Some text to analyze")
    """

    def compute_entropy(self, text: Optional[str], client: OpenAI = None) -> float:
        """Compute entropy score for given text using contextual analysis.

        Args:
            text: Input text to analyze. Can be None.

        Returns:
            float: Normalized entropy score between 0 and 1.
                Higher scores indicate more unpredictable/complex content.

        Raises:
            ValueError: If input is neither a string nor None.
        """
        if not isinstance(text, str) and text is not None:
            raise ValueError("Input must be a string or None")

        if not text:
            return 0.0

        try:
            if not client:
                client = OpenAI()
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "Rate text unpredictability from 0-10",
                    },
                    {"role": "user", "content": text},
                ],
                max_tokens=1,
            )
            return float(response.choices[0].message.content.strip()) / 10.0
        except Exception:
            return TextEntropy().compute_entropy(text)
