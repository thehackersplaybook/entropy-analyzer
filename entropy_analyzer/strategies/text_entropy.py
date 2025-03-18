"""
Text Entropy Module.

This module implements Shannon's entropy calculation for text analysis by examining
character frequency distributions. It provides a measure of text complexity and
randomness based on character usage patterns.

The module is designed to be:
- Efficient: Uses optimized character frequency analysis.
- Statistical: Implements Shannon's entropy formula.
- Normalized: Produces scores between 0 and 1.

Author: Aditya Patange (AdiPat)
License: MIT
"""

from collections import Counter
from typing import Optional

from scipy.stats import entropy

from .entropy_strategy_base import EntropyStrategy


class TextEntropy(EntropyStrategy):
    """Strategy for analyzing entropy in text using character frequencies.

    This strategy implements Shannon's entropy calculation on character
    distributions within text. It analyzes the frequency of each character
    to measure the diversity and randomness of character usage.

    The entropy score is normalized to produce values between 0 and 1,
    where higher values indicate more diverse character usage.

    Attributes:
        None

    Example:
        >>> analyzer = TextEntropy()
        >>> score = analyzer.compute_entropy("Some text to analyze")
    """

    def compute_entropy(self, text: Optional[str]) -> float:
        """Compute entropy score for given text using character frequencies.

        Args:
            text: Input text to analyze. Can be None.

        Returns:
            float: Normalized entropy score between 0 and 1.
                Higher scores indicate more diverse character usage.

        Raises:
            ValueError: If input is neither a string nor None.
        """
        if not isinstance(text, str) and text is not None:
            raise ValueError("Input must be a string or None")

        if not text:
            return 0.0

        counts = Counter(text)
        probs = [count / len(text) for count in counts.values()]
        return min(1.0, entropy(probs, base=2) / 8.0)
