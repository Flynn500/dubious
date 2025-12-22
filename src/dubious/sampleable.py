import numpy as np
from typing import Optional
from abc import abstractmethod

class Sampleable():
    @abstractmethod
    def sample(self, n: int, *, rng: np.random.Generator) -> np.ndarray:
        """
        Sample points from a distribution
        Args:
            n (int): Number of samples.
            rng (np.random.Generator): Numpy random generator.
        Returns:
            np.ndarray: Array of sampled points.
        """
        raise NotImplementedError
    
    @abstractmethod
    def mean(self) -> float:
        """
        Get the mean of a distribution
        Returns:
            float: mean
        """
        raise NotImplementedError
    
    @abstractmethod
    def var(self) -> float:
        """
        Get the variance of a distribution
        Returns:
            float: variance
        """
        raise NotImplementedError
    
    @abstractmethod
    def quantile(self, q: float, n: int = 50000, *, rng: Optional[np.random.Generator] = None, method: str = "linear", seed: int = 0) -> float:
        raise NotImplementedError
    