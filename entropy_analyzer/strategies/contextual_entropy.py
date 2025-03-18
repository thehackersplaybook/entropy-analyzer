from typing import Optional
from openai import OpenAI
from .entropy_strategy_base import EntropyStrategy
from .text_entropy import TextEntropy


class ContextualEntropy(EntropyStrategy):
    """Strategy for computing entropy of text data using contextual analysis.

    This strategy leverages GPT models to analyze the unpredictability and
    complexity of text content in a contextual manner. It considers semantic
    meaning and linguistic patterns beyond simple character frequencies.

    Falls back to basic text entropy calculation if API access fails.
    Higher scores indicate more unpredictable or complex content.
    """

    def compute_entropy(self, text: Optional[str]) -> float:
        if not isinstance(text, str) and text is not None:
            raise ValueError("Input must be a string or None")

        if not text:
            return 0.0

        try:
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
