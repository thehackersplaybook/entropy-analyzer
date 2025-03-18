"""

Module to define the EntropyStrategy interface and its implementations.
Covers the different types of entropy scores the system needs to compute.

Author: Aditya Patange (AdiPat)
License: MIT

"""

from abc import ABC, abstractmethod
from typing import List, Union, Optional, Dict, Any
import numpy as np
from scipy.stats import entropy
from collections import Counter
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from datetime import datetime


class EntropyStrategy(ABC):
    """Base abstract class for entropy calculation strategies.

    This class defines the interface for all entropy calculation implementations.
    Each implementation must normalize its output between 0 and 1.
    """

    @abstractmethod
    def compute_entropy(self, data: Union[str, List, None]) -> float:
        """Compute normalized entropy value for the given data.

        Args:
            data: Input data for entropy calculation. Can be string, list, or None.

        Returns:
            float: Normalized entropy value between 0 and 1.

        Raises:
            ValueError: If data format is invalid for the specific strategy.
        """
        pass


class TextEntropy(EntropyStrategy):
    """Strategy for computing entropy of text data using character frequency."""

    def compute_entropy(self, text: Optional[str]) -> float:
        """Compute entropy of text based on character frequency distribution.

        Args:
            text: Input text string to analyze.

        Returns:
            float: Normalized entropy value between 0 and 1.

        Raises:
            ValueError: If input is not a string.
        """
        if not isinstance(text, str) and text is not None:
            raise ValueError("Input must be a string or None")

        if not text:
            return 0.0

        try:
            counts = Counter(text)
            probs = [count / len(text) for count in counts.values()]
            return min(1.0, entropy(probs, base=2) / 8.0)
        except Exception as e:
            raise ValueError(f"Error computing text entropy: {str(e)}")


class NumericalEntropy(EntropyStrategy):
    """Strategy for computing entropy of numerical data using histogram analysis."""

    def compute_entropy(self, numbers: Optional[List[Union[int, float]]]) -> float:
        """Compute entropy of numerical data using histogram distribution.

        Args:
            numbers: List of numerical values.

        Returns:
            float: Normalized entropy value between 0 and 1.

        Raises:
            ValueError: If input is not a list of numbers.
        """
        if not isinstance(numbers, list) and numbers is not None:
            raise ValueError("Input must be a list of numbers or None")

        if not numbers:
            return 0.0

        try:
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
        except (ValueError, TypeError) as e:
            raise ValueError(str(e))


class SearchEngineEntropy(EntropyStrategy):
    """Strategy for computing entropy of search engine results using TF-IDF."""

    def compute_entropy(self, results: Optional[List[str]]) -> float:
        """Compute entropy of search engine results using TF-IDF diversity score.

        Args:
            results: List of search engine result strings.

        Returns:
            float: Normalized entropy value between 0 and 1.

        Raises:
            ValueError: If input is not a list of strings.
        """
        if not isinstance(results, list) and results is not None:
            raise ValueError("Input must be a list of strings or None")

        if not results:
            return 0.0

        if not all(isinstance(s, str) for s in results):
            raise ValueError("All elements must be strings")

        try:
            if len(set(results)) == 1:
                return 0.0

            vectorizer = TfidfVectorizer(
                min_df=1, analyzer="char_wb", ngram_range=(2, 3)
            )
            tfidf_matrix = vectorizer.fit_transform(results)
            scores = tfidf_matrix.toarray().std(axis=0)
            return float(min(1.0, np.mean(scores) if scores.size > 0 else 0.0))
        except Exception as e:
            raise ValueError(f"Error computing search entropy: {str(e)}")


class ContextualEntropy(EntropyStrategy):
    """Strategy for computing entropy of text data using contextual analysis."""

    def compute_entropy(self, text: Optional[str]) -> float:
        """Compute entropy of text based on contextual unpredictability.

        Args:
            text: Input text string to analyze.

        Returns:
            float: Normalized entropy value between 0 and 1.

        Raises:
            ValueError: If input is not a string.
        """
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
        except Exception as e:
            return TextEntropy().compute_entropy(text)


class TimeEntropy(EntropyStrategy):
    """Strategy for computing entropy of time series data using interval analysis."""

    def compute_entropy(self, timestamps: Optional[List[str]]) -> float:
        """Compute entropy of time series data based on interval distribution.

        Args:
            timestamps: List of ISO format timestamp strings.

        Returns:
            float: Normalized entropy value between 0 and 1.

        Raises:
            ValueError: If input is not a list of timestamp strings.
        """
        if not isinstance(timestamps, list) and timestamps is not None:
            raise ValueError("Input must be a list of timestamp strings or None")

        if not timestamps:
            return 0.0

        try:
            times = []
            for ts in timestamps:
                if not isinstance(ts, str):
                    raise ValueError("All elements must be timestamp strings")
                try:
                    times.append(datetime.fromisoformat(ts).timestamp())
                except ValueError:
                    raise ValueError(f"Invalid timestamp format: {ts}")

            if len(times) < 2:
                return 0.0

            intervals = np.diff(times)
            interval_range = max(intervals) - min(intervals)
            if interval_range == 0:
                return 0.0

            normalized_intervals = intervals / interval_range
            return float(min(1.0, entropy(normalized_intervals + 1e-10) / 8.0))
        except Exception as e:
            raise ValueError(f"Error computing time entropy: {str(e)}")


class EntropyFactory:
    """Factory class for creating entropy calculator instances."""

    @staticmethod
    def get_entropy_calculator(strategy_type: Optional[Any]) -> EntropyStrategy:
        """Create and return an entropy calculator instance based on strategy type.

        Args:
            strategy_type: String identifier for the desired strategy or None.

        Returns:
            EntropyStrategy: Instance of TextEntropy for invalid or None types,
            otherwise returns the requested strategy instance.
        """

        if not isinstance(strategy_type, str) or strategy_type is None:
            raise ValueError("Strategy type must be a string")

        valid_types = ["text", "numerical", "search", "contextual", "time"]

        if strategy_type not in valid_types:
            raise ValueError(f"Invalid Strategy Type: {strategy_type}")

        strategies: Dict[str, EntropyStrategy] = {
            "text": TextEntropy(),
            "numerical": NumericalEntropy(),
            "search": SearchEngineEntropy(),
            "contextual": ContextualEntropy(),
            "time": TimeEntropy(),
        }
        return strategies.get(strategy_type, TextEntropy())
