"""
Time Entropy Module.

This module implements entropy analysis for temporal data sequences. It analyzes
the distribution of time intervals between events to identify patterns and
irregularities in temporal sequences.

The module is designed to be:
- Temporal-aware: Specialized for timestamp analysis.
- ISO-compliant: Works with standard timestamp formats.
- Normalized: Produces scores between 0 and 1.

Author: Aditya Patange (AdiPat)
License: MIT
"""

from datetime import datetime
from typing import List, Optional

import numpy as np
from scipy.stats import entropy

from .entropy_strategy_base import EntropyStrategy


class TimeEntropy(EntropyStrategy):
    """Strategy for analyzing entropy in time series data.

    This strategy analyzes the distribution of time intervals between timestamps
    to measure temporal patterns and irregularities. It converts ISO format
    timestamps to numerical intervals and computes entropy on their distribution.

    The entropy calculation is normalized and considers the relative spacing
    of events rather than absolute times.

    Attributes:
        None

    Example:
        >>> analyzer = TimeEntropy()
        >>> score = analyzer.compute_entropy(["2023-01-01T12:00:00",
                                           "2023-01-01T12:30:00"])
    """

    def compute_entropy(self, timestamps: Optional[List[str]]) -> float:
        """Compute entropy score for a sequence of timestamps.

        Args:
            timestamps: List of ISO format timestamp strings to analyze. Can be None.

        Returns:
            float: Normalized entropy score between 0 and 1.
                Higher scores indicate more irregular timing patterns.

        Raises:
            ValueError: If input is not a list of valid ISO format timestamp strings.
        """
        if not isinstance(timestamps, list) and timestamps is not None:
            raise ValueError("Input must be a list of timestamp strings or None")

        if not timestamps:
            return 0.0

        times = []
        for ts in timestamps:
            if not isinstance(ts, str):
                raise ValueError("All elements must be timestamp strings")
            times.append(datetime.fromisoformat(ts).timestamp())

        if len(times) < 2:
            return 0.0

        intervals = np.diff(times)
        interval_range = max(intervals) - min(intervals)
        if interval_range == 0:
            return 0.0

        normalized_intervals = intervals / interval_range
        return float(min(1.0, entropy(normalized_intervals + 1e-10) / 8.0))
