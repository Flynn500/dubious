import warnings
from matplotlib.pylab import Generator
import numpy as np
from typing import Union, Optional, cast, Literal
import numbers
from abc import abstractmethod

class Distribution:
    @abstractmethod
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
    
    @abstractmethod
    def mean(self) -> float:
        """
        Mean of a distribution
        Returns:
            float: mean
        """
        raise NotImplementedError
    
    @abstractmethod
    def var(self) -> float:
        """
        Variance of a distribution
        Returns:
            float: variance
        """
        raise NotImplementedError
    
    @abstractmethod
    def quantile(self, q: float, n: int = 50_000, rng: Optional[np.random.Generator] = None, method: str = "linear",) -> float:
        if not (0.0 <= q <= 1.0):
            raise ValueError("q must be between 0 and 1.")
        if rng is None:
            rng = np.random.default_rng()
        s = self.sample(n, rng)

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
        return float(np.quantile(s, q, method=method_lit))
    
    

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
    
    def quantile(self, q: float, n: int = 50000, rng: Optional[np.random.Generator] = None, method: str = "linear") -> float:
        return super().quantile(q, n, rng, method)


class Uniform(Distribution):
    def __init__(self, low: Union[float, Distribution] = 0.0, high: Union[float, Distribution] = 1.0):
        if _is_scalar_real(high) and _is_scalar_real(low):
            if high <= low: # type: ignore
                raise ValueError("high must be greater than low.")
        self.low = low
        self.high = high

    def sample(self, n: int, rng: Optional[np.random.Generator]) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        
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
    
    def quantile(self, q: float, n: int = 50000, rng: Optional[np.random.Generator] = None, method: str = "linear") -> float:
        return super().quantile(q, n, rng, method)


class LogNormal(Distribution):
    def __init__(self, mu: Union[float, Distribution] =0.0, sigma: Union[float, Distribution] =1.0):
        if isinstance(sigma, numbers.Real) and sigma <= 0:
            raise ValueError("sigma must be positive.")
        self.mu = mu
        self.sigma = sigma

    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()

        mu = self.mu.sample(n, rng) if isinstance(self.mu, Distribution) else self.mu
        sigma = self.sigma.sample(n, rng) if isinstance(self.sigma, Distribution) else self.sigma

        if np.any(np.asarray(sigma) <= 0):
            warnings.warn("Warning: Sigma <= 0 found, clamped to 1e-6.")
            sigma = np.clip(sigma, a_min=1e-6, a_max=None)

        return rng.lognormal(mean=mu, sigma=sigma, size=n)

    def mean(self, _moment_mc_samples: int = 200_000, rng: Optional[np.random.Generator] = None,  _moment_seed: int = 0) -> float:
        if not isinstance(self.mu, Distribution) and not isinstance(self.sigma, Distribution):
            mu = float(self.mu)
            sigma = float(self.sigma)
            return float(np.exp(mu + 0.5 * sigma**2))

        if rng is None:
            rng = np.random.default_rng(_moment_seed)

        mu_s = self.mu.sample(_moment_mc_samples, rng) if isinstance(self.mu, Distribution) else np.full(_moment_mc_samples, self.mu)
        sg_s = self.sigma.sample(_moment_mc_samples, rng) if isinstance(self.sigma, Distribution) else np.full(_moment_mc_samples, self.sigma)

        sg_s = np.clip(sg_s, 1e-6, None)
        return float(np.mean(np.exp(mu_s + 0.5 * sg_s**2)))
    
    def var(self, _moment_mc_samples: int = 200_000, rng: Optional[np.random.Generator] = None, _moment_seed: int = 0) -> float:
        if not isinstance(self.mu, Distribution) and not isinstance(self.sigma, Distribution):
            mu = float(self.mu)
            sigma = float(self.sigma)
            return float((np.exp(sigma**2) - 1.0) * np.exp(2.0 * mu + sigma**2))

        if rng is None:
            rng = np.random.default_rng(_moment_seed)

        mu_s = self.mu.sample(_moment_mc_samples, rng) if isinstance(self.mu, Distribution) else np.full(_moment_mc_samples, self.mu)
        sg_s = self.sigma.sample(_moment_mc_samples, rng) if isinstance(self.sigma, Distribution) else np.full(_moment_mc_samples, self.sigma)

        sg_s = np.clip(sg_s, 1e-6, None)

        Ey_cond = np.exp(mu_s + 0.5 * sg_s**2)
        Vy_cond = (np.exp(sg_s**2) - 1.0) * np.exp(2.0 * mu_s + sg_s**2)

        return float(np.mean(Vy_cond) + np.var(Ey_cond, ddof=0))
    
    def quantile(self, q: float, n: int = 50000, rng: Optional[np.random.Generator] = None, method: str = "linear") -> float:
        return super().quantile(q, n, rng, method)
