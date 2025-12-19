import warnings
import numpy as np
from typing import Union
import numbers
class Distribution:
    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """
        Sample points from a distribution
        Args:
            n (int): Number of samples.
            rng (np.random.Generator): Numpy random generator.
        Returns:
            np.ndarray: Array of sampled points.
        """
        raise NotImplementedError
    
    def mean(self) -> float:
        """
        Mean of a distribution
        Returns:
            float: mean
        """
        raise NotImplementedError
    
    def var(self) -> float:
        """
        Variance of a distribution
        Returns:
            float: variance
        """
        raise NotImplementedError
    
    

def _mean(x: Union[float, Distribution]) -> float:
    return x.mean() if isinstance(x, Distribution) else float(x)

def _var(x: Union[float, Distribution]) -> float:
    return x.var() if isinstance(x, Distribution) else 0.0

def _is_scalar_real(x: object) -> bool:
    return isinstance(x, numbers.Real) and not isinstance(x, np.ndarray)

class Normal(Distribution):
    def __init__(self, mu: Union[float, Distribution] = 0.0, sigma: Union[float, Distribution] = 1.0):
        if isinstance(sigma, numbers.Real) and sigma <= 0:
            raise ValueError("sigma must be positive.")
        self.mu = mu
        self.sigma = sigma

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if isinstance(self.mu, Distribution):
            mu = self.mu.sample(n, rng)
        else:
            mu = self.mu

        if isinstance(self.sigma, Distribution):
            sigma = self.sigma.sample(n, rng) 
        else:
            sigma = self.sigma
        
        if np.any(sigma <= 0):
            warnings.warn("Warning: Sigma <= 0 found, clamped to 1e-6.")
            sigma = np.clip(sigma, a_min=1e-6, a_max=None)
        return rng.normal(loc=mu, scale=sigma, size=n)

    def mean(self) -> float:
        return _mean(self.mu)

    def var(self) -> float:
        mu_var = _var(self.mu)

        if isinstance(self.sigma, Distribution):
            s_mean = self.sigma.mean()
            s_var = self.sigma.var()
            sigma2 = s_var + s_mean**2

        else:
            s = float(self.sigma)
            sigma2 = s**2
        return sigma2 + mu_var


class Uniform(Distribution):
    def __init__(self, low: Union[float, Distribution] = 0.0, high: Union[float, Distribution] = 1.0):
        if _is_scalar_real(high) and _is_scalar_real(low):
            if high <= low: # type: ignore
                raise ValueError("high must be greater than low.")
        self.low = low
        self.high = high

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if isinstance(self.high, Distribution):
            high = self.high.sample(n, rng)
        else:
            high = self.high

        if isinstance(self.low, Distribution):
            low = self.low.sample(n, rng) 
        else:
            low = self.low
        return rng.uniform(low=low, high=high, size=n)

    def mean(self) -> float:
        if isinstance(self.high, Distribution): h = self.high.mean() 
        else: h = self.high

        if isinstance(self.low, Distribution): l = self.low.mean() 
        else: l = self.low

        return 0.5 * (l + h)

    def var(self) -> float:
        low_m, high_m = _mean(self.low), _mean(self.high)
        low_v, high_v = _var(self.low), _var(self.high)

        term1 = (low_v + high_v + (high_m - low_m) ** 2) / 12.0
        term2 = (low_v + high_v) / 4.0
        return term1 + term2
    
class LogNormal(Distribution):
    def __init__(self, mu: float = 0.0, sigma: float = 1.0):
        if sigma <= 0:
            raise ValueError("sigma must be positive.")
        self.mu = mu
        self.sigma = sigma

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.lognormal(mean=self.mu, sigma=self.sigma, size=n)

    def mean(self) -> float:
        return np.exp(self.mu + 0.5 * self.sigma ** 2)
    
    def var(self) -> float:
        return (np.exp(self.sigma ** 2) - 1) * np.exp(2 * self.mu + self.sigma ** 2)

