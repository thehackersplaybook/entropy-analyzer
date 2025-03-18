"""
Search Engine Entropy Module.

This module implements entropy analysis for search engine results using TF-IDF
vectorization. It measures result diversity and semantic uniqueness through
character-level n-gram analysis.

The module is designed to be:
- Semantic-aware: Uses TF-IDF for content comparison.
- Scalable: Handles variable result sizes.
- Normalized: Produces scores between 0 and 1.

Author: Aditya Patange (AdiPat)
License: MIT
"""

from typing import List, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from .entropy_strategy_base import EntropyStrategy


class SearchEngineEntropy(EntropyStrategy):
    """Strategy for analyzing diversity in search engine results.

    This strategy measures the uniqueness and diversity of search results
    using TF-IDF (Term Frequency-Inverse Document Frequency) analysis.
    It captures the semantic differences between search results by analyzing
    character-level n-grams.

    The entropy calculation considers the standard deviation of TF-IDF scores
    across results to quantify result diversity.

    Attributes:
        None

    Example:
        >>> analyzer = SearchEngineEntropy()
        >>> score = analyzer.compute_entropy(["result1", "result2", "result3"])
    """

    def compute_entropy(self, results: Optional[List[str]]) -> float:
        """Compute entropy score for a list of search results.

        Args:
            results: List of search result strings to analyze. Can be None.

        Returns:
            float: Normalized entropy score between 0 and 1.
                Higher scores indicate more diverse search results.

        Raises:
            ValueError: If input is not a list of strings.
        """
        if not isinstance(results, list) and results is not None:
            raise ValueError("Input must be a list of strings or None")

        if not results:
            return 0.0

        if not all(isinstance(s, str) for s in results):
            raise ValueError("All elements must be strings")

        if len(set(results)) == 1:
            return 0.0

        vectorizer = TfidfVectorizer(min_df=1, analyzer="char_wb", ngram_range=(2, 3))
        tfidf_matrix = vectorizer.fit_transform(results)
        scores = tfidf_matrix.toarray().std(axis=0)
        return float(min(1.0, np.mean(scores) if scores.size > 0 else 0.0))
