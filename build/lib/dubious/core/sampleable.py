import numpy as np
from abc import abstractmethod
from typing import Union, Optional, cast, Literal


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
    def quantile(self, q: float, n: int = 50000, *, rng: Optional[np.random.Generator] = None, method: str = "linear", seed: Union[int, None] = None) -> float:
        raise NotImplementedError

class Distribution(Sampleable):
    @abstractmethod
    def sample(self, n: int, *, rng: Optional[np.random.Generator] = None, seed: Union[int, None] = None) -> np.ndarray:
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

    def quantile(self, q: Union[float, np.ndarray], n: int = 50000, *, method: str = "linear", rng: Optional[np.random.Generator] = None, seed: Union[int, None] = None) -> Union[float, np.ndarray]:
        """
        Compute an approximation of the q-th quantile of data. 
        Args:
            q (float): Probabilty of quantiles to compute.
            n (int): Number of samples.
            rng (np.random.Generator): Numpy random generator.
        Returns:
            float: quantile
        """
        q = np.asarray(q)

        if np.any((q < 0.0) | (q > 1.0)):
            raise ValueError("q must be between 0 and 1")
        
        if rng is None:
            rng = np.random.default_rng(seed)
        s = self.sample(n, rng=rng, seed=seed)

        #cast to avoid numpy getting mad
        method_lit = cast(
            Literal[
                "inverted_cdf", "averaged_inverted_cdf",
                "closest_observation", "interpolated_inverted_cdf",
                "hazen", "weibull", "linear", "median_unbiased",
                "normal_unbiased"
            ],
            method,
        )
        result = np.quantile(s, q, method=method_lit)
        return result.item() if result.ndim == 0 else result
    
    def cdf(self, x: float, n: int = 200_000, *, rng=None, seed: Union[int, None] = None) -> float:
        """
        Compute an approximation of the cumulative density function. 
        Args:
            x (float): Value.
            n (int): Number of samples.
            rng (np.random.Generator): Numpy random generator.
        Returns:
            float: quantile
        """
        if rng is None:
            rng = np.random.default_rng(seed)
        
        s = self.sample(n, rng=rng)
        return float(np.mean(s <= x))