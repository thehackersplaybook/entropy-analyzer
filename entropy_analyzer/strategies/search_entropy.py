from typing import Optional, List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from .entropy_strategy_base import EntropyStrategy


class SearchEngineEntropy(EntropyStrategy):
    """Strategy for computing entropy of search engine results using TF-IDF.

    This strategy measures the diversity and uniqueness of search results
    using Term Frequency-Inverse Document Frequency (TF-IDF) analysis.
    It captures how different or similar the search results are from each other.

    Higher scores indicate more diverse and unique search results, while
    lower scores suggest more repetitive or similar results.
    """

    def compute_entropy(self, results: Optional[List[str]]) -> float:
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
