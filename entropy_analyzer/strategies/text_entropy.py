from typing import Optional
from collections import Counter
from scipy.stats import entropy
from .entropy_strategy_base import EntropyStrategy


class TextEntropy(EntropyStrategy):
    """Strategy for computing entropy of text data using character frequency.

    This strategy analyzes the distribution of characters in text to compute
    an entropy score. It uses Shannon's entropy formula on character frequencies,
    normalized to produce values between 0 and 1.

    Higher scores indicate more diverse character usage and potentially
    more complex or random text.
    """

    def compute_entropy(self, text: Optional[str]) -> float:
        if not isinstance(text, str) and text is not None:
            raise ValueError("Input must be a string or None")

        if not text:
            return 0.0

        counts = Counter(text)
        probs = [count / len(text) for count in counts.values()]
        return min(1.0, entropy(probs, base=2) / 8.0)
