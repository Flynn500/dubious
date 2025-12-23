import warnings
from matplotlib.pylab import Generator
import numpy as np
from typing import Union, Optional, cast, Literal
import numbers
from abc import abstractmethod

from .sampleable import Sampleable

class Distribution(Sampleable):
    @abstractmethod
    def sample(self, n: int, *, rng: np.random.Generator, seed: int = 0) -> np.ndarray:
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
        """
        Compute the q-th quantile of data.
        Args:
            q (float): Probabilty of quantiles to compute.
            n (int): Number of samples.
            rng (np.random.Generator): Numpy random generator.
        Returns:
            float: quantile
        """
        if not (0.0 <= q <= 1.0):
            raise ValueError("q must be between 0 and 1.")
        if rng is None:
            rng = np.random.default_rng(seed)
        s = self.sample(n, rng=rng)

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
    
    

def _mean(x: Union[float, Sampleable]) -> float:
    return x.mean() if isinstance(x, Sampleable) else float(x)

def _var(x: Union[float, Sampleable]) -> float:
    return x.var() if isinstance(x, Sampleable) else 0.0

def _is_scalar_real(x: object) -> bool:
    return isinstance(x, numbers.Real) and not isinstance(x, np.ndarray)

class Normal(Distribution):
    def __init__(self, mu: Union[float, Sampleable] = 0.0, sigma: Union[float, Sampleable] = 1.0):
        if isinstance(sigma, numbers.Real) and sigma <= 0:
            raise ValueError("sigma must be positive.")
        self.mu = mu
        self.sigma = sigma

    def sample(self, n: int, *, rng: Optional[np.random.Generator] = None, seed: int = 0) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng(seed)

        if isinstance(self.mu, Sampleable):
            mu = self.mu.sample(n, rng=rng)
        else:
            mu = self.mu

        if isinstance(self.sigma, Sampleable):
            sigma = self.sigma.sample(n, rng=rng) 
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

        if isinstance(self.sigma, Sampleable):
            s_mean = self.sigma.mean()
            s_var = self.sigma.var()
            sigma2 = s_var + s_mean**2

        else:
            s = float(self.sigma)
            sigma2 = s**2
        return sigma2 + mu_var
    
    def quantile(self, q: float, n: int = 50000, *, method: str = "linear", rng: Optional[np.random.Generator] = None, seed: int = 0) -> float:
        return super().quantile(q, n, rng=rng, method=method, seed=seed)


class Uniform(Distribution):
    def __init__(self, low: Union[float, Sampleable] = 0.0, high: Union[float, Sampleable] = 1.0):
        if _is_scalar_real(high) and _is_scalar_real(low):
            if high <= low: # type: ignore
                raise ValueError("high must be greater than low.")
        self.low = low
        self.high = high

    def sample(self, n: int, *, rng: Optional[np.random.Generator] = None, seed: int = 0) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng(seed)
        
        if isinstance(self.high, Sampleable):
            high = self.high.sample(n, rng=rng)
        else:
            high = self.high

        if isinstance(self.low, Sampleable):
            low = self.low.sample(n, rng=rng) 
        else:
            low = self.low
        return rng.uniform(low=low, high=high, size=n)

    def mean(self) -> float:
        if isinstance(self.high, Sampleable): h = self.high.mean() 
        else: h = self.high

        if isinstance(self.low, Sampleable): l = self.low.mean() 
        else: l = self.low

        return 0.5 * (l + h)

    def var(self) -> float:
        low_m, high_m = _mean(self.low), _mean(self.high)
        low_v, high_v = _var(self.low), _var(self.high)

        term1 = (low_v + high_v + (high_m - low_m) ** 2) / 12.0
        term2 = (low_v + high_v) / 4.0
        return term1 + term2
    
    def quantile(self, q: float, n: int = 50000, *, method: str = "linear", rng: Optional[np.random.Generator] = None, seed: int = 0) -> float:
        return super().quantile(q, n, rng=rng, method=method, seed=seed)


class LogNormal(Distribution):
    def __init__(self, mu: Union[float, Sampleable] =0.0, sigma: Union[float, Sampleable] =1.0):
        if isinstance(sigma, numbers.Real) and sigma <= 0:
            raise ValueError("sigma must be positive.")
        self.mu = mu
        self.sigma = sigma

    def sample(self, n: int, *, rng: Optional[np.random.Generator] = None, seed: int = 0) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng(seed)

        mu = self.mu.sample(n, rng=rng) if isinstance(self.mu, Sampleable) else self.mu
        sigma = self.sigma.sample(n, rng=rng) if isinstance(self.sigma, Sampleable) else self.sigma

        if np.any(np.asarray(sigma) <= 0):
            warnings.warn("Warning: Sigma <= 0 found, clamped to 1e-6.")
            sigma = np.clip(sigma, a_min=1e-6, a_max=None)

        return rng.lognormal(mean=mu, sigma=sigma, size=n)

    def mean(self, n: int = 200_000, *, rng: Optional[np.random.Generator] = None,  seed: int = 0) -> float:
        """
        Get the mean of a distribution
        Args:
            n (int): Only applies when paramaters are distributions. Number of samples used in MC aproximation.
            rng (np.random.Generator): Random generator to use when sampling.
            seed (int): RNG seed.
        Returns:
            float: mean
        """
        if not isinstance(self.mu, Sampleable) and not isinstance(self.sigma, Sampleable):
            mu = float(self.mu)
            sigma = float(self.sigma)
            return float(np.exp(mu + 0.5 * sigma**2))

        if rng is None:
            rng = np.random.default_rng(seed)

        mu_s = self.mu.sample(n, rng=rng) if isinstance(self.mu, Sampleable) else np.full(n, self.mu)
        sg_s = self.sigma.sample(n, rng=rng) if isinstance(self.sigma, Sampleable) else np.full(n, self.sigma)

        sg_s = np.clip(sg_s, 1e-6, None)
        return float(np.mean(np.exp(mu_s + 0.5 * sg_s**2)))
    
    def var(self, n: int = 200_000, *, rng: Optional[np.random.Generator] = None, seed: int = 0) -> float:
        """
        Get the variance of a distribution
        Args:
            n (int): Only applies when paramaters are distributions. Number of samples used in MC aproximation.
            rng (np.random.Generator): Random generator to use when sampling.
            seed (int): RNG seed.
        Returns:
            float: mean
        """
        if not isinstance(self.mu, Sampleable) and not isinstance(self.sigma, Sampleable):
            mu = float(self.mu)
            sigma = float(self.sigma)
            return float((np.exp(sigma**2) - 1.0) * np.exp(2.0 * mu + sigma**2))

        if rng is None:
            rng = np.random.default_rng(seed)

        mu_s = self.mu.sample(n, rng=rng) if isinstance(self.mu, Sampleable) else np.full(n, self.mu)
        sg_s = self.sigma.sample(n, rng=rng) if isinstance(self.sigma, Sampleable) else np.full(n, self.sigma)

        sg_s = np.clip(sg_s, 1e-6, None)

        Ey_cond = np.exp(mu_s + 0.5 * sg_s**2)
        Vy_cond = (np.exp(sg_s**2) - 1.0) * np.exp(2.0 * mu_s + sg_s**2)

        return float(np.mean(Vy_cond) + np.var(Ey_cond, ddof=0))
    
    def quantile(self, q: float, n: int = 50000, *, method: str = "linear", rng: Generator | None = None, seed: int = 0) -> float:
        return super().quantile(q, n, rng=rng, method=method, seed=seed)


class Beta(Distribution):
    def __init__(self, alpha: Union[float, Sampleable] = 1.0, beta: Union[float, Sampleable] = 1.0):
        if isinstance(alpha, numbers.Real) and alpha <= 0:
            raise ValueError("alpha must be positive.")
        if isinstance(beta, numbers.Real) and beta <= 0:
            raise ValueError("beta must be positive.")
        self.alpha = alpha
        self.beta = beta

    def sample(self, n: int, *, rng: Optional[np.random.Generator] = None, seed: int = 0) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng(seed)

        if isinstance(self.alpha, Sampleable):
            a = self.alpha.sample(n, rng=rng)
        else:
            a = self.alpha

        if isinstance(self.beta, Sampleable):
            b = self.beta.sample(n, rng=rng)
        else:
            b = self.beta
        
        a_arr = np.asarray(a)
        b_arr = np.asarray(b)

        bad = (a_arr <= 0) | (b_arr <= 0)
        if np.any(bad):
            warnings.warn("Warning: alpha <= 0 or beta <= 0 found, clamped to 1e-6.")
            a_arr = np.clip(a_arr, a_min=1e-6, a_max=None)
            b_arr = np.clip(b_arr, a_min=1e-6, a_max=None)

        return rng.beta(a_arr, b_arr, size=n)
    
    def mean(self, n=200_000, *, rng: Optional[np.random.Generator] = None, seed: int = 0) -> float:
        """
        Get the mean of a distribution
        Args:
            n (int): Only applies when paramaters are distributions. Number of samples used in MC aproximation.
            rng (np.random.Generator): Random generator to use when sampling.
            seed (int): RNG seed.
        Returns:
            float: mean
        """
        if not isinstance(self.alpha, Sampleable) and not isinstance(self.beta, Sampleable):
            a = float(self.alpha)
            b = float(self.beta)
            return a / (a + b)

        if rng is None:
            rng = np.random.default_rng(seed)
        
        a = self.alpha.sample(n, rng=rng) if isinstance(self.alpha, Sampleable) else float(self.alpha)
        b = self.beta.sample(n, rng=rng) if isinstance(self.beta, Sampleable) else float(self.beta)
        a = np.clip(np.asarray(a), 1e-6, None)
        b = np.clip(np.asarray(b), 1e-6, None)
        return float(np.mean(a / (a + b)))
    
    def var(self, n: int = 200_000, *, rng: Optional[np.random.Generator] = None, seed: int = 0) -> float:
        """
        Get the variance of a distribution
        Args:
            n (int): Only applies when paramaters are distributions. Number of samples used in MC aproximation.
            rng (np.random.Generator): Random generator to use when sampling.
            seed (int): RNG seed.
        Returns:
            float: mean
        """
        if not isinstance(self.alpha, Sampleable) and not isinstance(self.beta, Sampleable):
            a = float(self.alpha)
            b = float(self.beta)
            denom = (a + b) ** 2 * (a + b + 1.0)
            return (a * b) / denom
        
        if rng is None:
            rng = np.random.default_rng(seed)

        a = self.alpha.sample(n, rng=rng) if isinstance(self.alpha, Sampleable) else float(self.alpha)
        b = self.beta.sample(n, rng=rng) if isinstance(self.beta, Sampleable) else float(self.beta)
        a = np.clip(np.asarray(a), 1e-6, None)
        b = np.clip(np.asarray(b), 1e-6, None)

        s = a + b
        m = a / s
        v = (a * b) / (s * s * (s + 1.0))

        return float(np.mean(v) + np.var(m))
    
    def quantile(self, q: float, n: int = 50000, *, method: str = "linear", rng: Generator | None = None, seed: int = 0) -> float:
        return super().quantile(q, n, rng=rng, method=method, seed=seed)






