from typing import List, Union, Optional
import numpy as np
from scipy.stats import entropy
from .entropy_strategy_base import EntropyStrategy


class NumericalEntropy(EntropyStrategy):
    """Strategy for computing entropy of numerical data using histogram analysis.

    This strategy creates a histogram of numerical values and computes entropy
    based on the distribution of values across bins. The approach helps identify
    patterns and randomness in numerical sequences.

    The strategy is particularly useful for analyzing distributions, detecting
    clusters, and measuring the diversity of numerical datasets.
    """

    def compute_entropy(self, numbers: Optional[List[Union[int, float]]]) -> float:
        if not isinstance(numbers, list) and numbers is not None:
            raise ValueError("Input must be a list of numbers or None")

        if not numbers:
            return 0.0

        if not all(isinstance(n, (int, float)) for n in numbers):
            raise ValueError("All elements must be numbers")

        numbers = [float(n) for n in numbers]
        if not all(np.isfinite(n) for n in numbers):
            raise ValueError("Input contains non-finite values")
        if len(numbers) < 2:
            return 0.0

        hist, _ = np.histogram(np.array(numbers), bins="auto", density=True)
        hist = hist + 1e-10
        return float(min(1.0, entropy(hist) / 8.0))
