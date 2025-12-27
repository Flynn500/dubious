import numpy as np
from abc import abstractmethod
from typing import Union, Optional, cast, Literal
from math import sqrt

from .sampler import Sampler

class Sampleable():
    @abstractmethod
    def sample(self, n: int, *, sampler: Optional[Sampler] = None) -> np.ndarray:
        """
        Sample points from a distribution.

        :param n: Number of samples.
        :type n: int
        :param rng: NumPy random number generator.
        :type rng: np.random.Generator
        :return: Array of sampled points.
        :rtype: np.ndarray
        """
        raise NotImplementedError
    
    @abstractmethod
    def mean(self) -> float:
        """
        Get the mean of a distribution.

        :return: Mean.
        :rtype: float
        """
        raise NotImplementedError
    
    @abstractmethod
    def var(self) -> float:
        """
        Get the variance of a distribution.

        :return: Variance.
        :rtype: float
        """
        raise NotImplementedError
    
    @abstractmethod
    def quantile(self, q: float, n: int = 50000, *, sampler: Optional[Sampler] = None, method: str = "linear") -> float:
        """
        Compute an approximation of the q-th quantile of data. Defaults to monte carlo simulation if other 
        dsitributions are used as parameters otherwise analytic methods are used.

        :param q: Probability of the quantile to compute.
        :type q: float
        :param n: Number of samples.
        :type n: int
        :param rng: NumPy random number generator.
        :type rng: np.random.Generator
        :return: Estimated quantile value.
        :rtype: float
        """
        raise NotImplementedError
    @abstractmethod
    def cdf(self, x: float, n: int = 200_000, *, sampler: Optional[Sampler] = None) -> float:
        """
        Compute an approximation of the cumulative density function. Defaults to monte carlo simulation if other 
        distributions are used as parameters otherwise analytic methods are used. 
        
        :param x: Value.
        :type q: float
        :param x: Numer of samples.
        :type n: int
        :param rng: NumPy random number generator.
        :type rng: np.random.Generator
        :return: Estimated CDF.
        :rtype: float
        """
        raise NotImplementedError
    
    def summary(self):
        """
        Print a summary of summary of statistics.
        """
        print(f"Summary for type: {self.__class__.__name__}")
        print("---------------------------\n" \
            f"mean      : {self.mean():.3f}\n" \
            f"median    : {self.quantile(0.5):.3f}\n" \
            f"std       : {sqrt(self.var()):.3f}\n" \
            f"variance  : {self.var():.3f}\n" \
            f"q05 / q95 : {self.quantile(0.05):.3f} / {self.quantile(0.95):.3f}\n" \
            f"q25 / q75 : {self.quantile(0.25):.3f} / {self.quantile(0.75):.3f}\n"
        )   

class Distribution(Sampleable):
    @abstractmethod
    def sample(self, n: int, *, sampler: Optional[Sampler] = None, advance: int = 0) -> np.ndarray:
        """
        Sample points from a distribution.

        :param n: Number of samples.
        :type n: int
        :param rng: NumPy random number generator.
        :type rng: np.random.Generator
        :return: Array of sampled points.
        :rtype: np.ndarray
        """
        raise NotImplementedError
    
    @abstractmethod
    def mean(self) -> float:
        """
        Get the mean of a distribution. Defaults to monte carlo simulation if other 
        dsitributions are used as parameters otherwise analytic methods are used.

        :return: Mean.
        :rtype: float
        """
        raise NotImplementedError
    
    @abstractmethod
    def var(self) -> float:
        """
        Get the variance of a distribution. Defaults to monte carlo simulation if other 
        dsitributions are used as parameters otherwise analytic methods are used.

        :return: Variance.
        :rtype: float
        """
        raise NotImplementedError

    def quantile(self, q: Union[float, np.ndarray], n: int = 50000, *, method: str = "linear", sampler: Optional[Sampler] = None) -> Union[float, np.ndarray]:
        """
        Compute an approximation of the q-th quantile of data. Defaults to monte carlo simulation if other 
        distributions are used as parameters otherwise analytic methods are used.

        :param q: Probability of the quantile to compute.
        :type q: float
        :param n: Number of samples.
        :type n: int
        :param rng: NumPy random number generator.
        :type rng: np.random.Generator
        :return: Estimated quantile value.
        :rtype: float
        """
        q = np.asarray(q)

        if np.any((q < 0.0) | (q > 1.0)):
            raise ValueError("q must be between 0 and 1")
        
        s = self.sample(n, sampler=sampler)

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
    
    def cdf(self, x: float, n: int = 200_000, *, sampler: Optional[Sampler] = None) -> float:
        """
        Compute an approximation of the cumulative density function. Defaults to monte carlo simulation if other 
        dsitributions are used as parameters otherwise analytic methods are used. 
        
        :param x: Value.
        :type q: float
        :param x: Numer of samples.
        :type n: int
        :param rng: NumPy random number generator.
        :type rng: np.random.Generator
        :return: Estimated CDF.
        :rtype: float
        """
        
        s = self.sample(n, sampler=sampler)
        return float(np.mean(s <= x))

