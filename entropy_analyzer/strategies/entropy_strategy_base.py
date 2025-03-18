"""
Entropy Strategy Base Module.

This module defines the base abstract class for all entropy calculation strategies.
It establishes a common interface and contract that all entropy analyzers must follow,
enabling polymorphic usage of different entropy calculation methods.

The module is designed to be:
- Abstract: Provides interface definition only.
- Flexible: Supports various input data types.
- Consistent: Enforces normalized output range.

Author: Aditya Patange (AdiPat)
License: MIT
"""

from abc import ABC, abstractmethod
from typing import List, Union


class EntropyStrategy(ABC):
    """Base abstract class for entropy calculation strategies.

    This class serves as the foundation for all entropy calculation implementations
    in the system. Each concrete implementation must provide a way to compute
    entropy values normalized between 0 and 1.

    The strategy pattern allows for different entropy calculation methods
    to be interchangeable while maintaining a consistent interface.
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
