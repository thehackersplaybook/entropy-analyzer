from typing import Optional, List
import numpy as np
from datetime import datetime
from scipy.stats import entropy
from .entropy_strategy_base import EntropyStrategy


class TimeEntropy(EntropyStrategy):
    """Strategy for computing entropy of time series data using interval analysis.

    This strategy analyzes the distribution of time intervals between timestamps
    to measure temporal patterns and irregularities. It converts timestamps to
    numerical intervals and computes entropy on their distribution.

    Higher scores indicate more irregular or random timing patterns, while
    lower scores suggest regular or predictable intervals.
    """

    def compute_entropy(self, timestamps: Optional[List[str]]) -> float:
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
