"""
Numerical Entropy Module.

This module provides entropy analysis for numerical sequences using histogram-based
distribution analysis. It quantifies the randomness and patterns in numerical data
through statistical methods.

The module is designed to be:
- Robust: Handles both integer and floating-point numbers.
- Statistical: Uses histogram analysis for distribution patterns.
- Normalized: Produces scores between 0 and 1.

Author: Aditya Patange (AdiPat)
License: MIT
"""

from typing import List, Optional, Union

import numpy as np
from scipy.stats import entropy

from .entropy_strategy_base import EntropyStrategy


class NumericalEntropy(EntropyStrategy):
    """Strategy for analyzing entropy in numerical sequences.

    This strategy creates a histogram of numerical values and computes entropy
    based on the distribution of values across bins. The approach helps identify
    patterns, randomness, and clustering in numerical sequences.

    The entropy calculation is normalized to produce values between 0 and 1,
    where higher values indicate more diverse or random distributions.

    Attributes:
        None

    Example:
        >>> analyzer = NumericalEntropy()
        >>> score = analyzer.compute_entropy([1.0, 2.5, 3.7, 1.2])
    """

    def compute_entropy(self, numbers: Optional[List[Union[int, float]]]) -> float:
        """Compute entropy score for a sequence of numbers.

        Args:
            numbers: List of numerical values to analyze. Can be None.
                    Accepts both integers and floating-point numbers.

        Returns:
            float: Normalized entropy score between 0 and 1.
                Higher scores indicate more diverse distributions.

        Raises:
            ValueError: If input is not a list of numbers, contains non-finite
                      values, or if elements are not numerical types.
        """
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
