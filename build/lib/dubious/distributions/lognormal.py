
import warnings
import numpy as np
from typing import Union, Optional
import numbers

from ..core.sampleable import Sampleable, Distribution


class LogNormal(Distribution):
    def __init__(self, mu: Union[float, Sampleable] =0.0, sigma: Union[float, Sampleable] =1.0):
        if isinstance(sigma, numbers.Real) and sigma <= 0:
            raise ValueError("sigma must be positive.")
        self.mu = mu
        self.sigma = sigma

    def sample(self, n: int, *, rng: Optional[np.random.Generator] = None, seed: Union[int, None] = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

        mu = self.mu.sample(n, rng=rng) if isinstance(self.mu, Sampleable) else self.mu
        sigma = self.sigma.sample(n, rng=rng) if isinstance(self.sigma, Sampleable) else self.sigma

        if np.any(np.asarray(sigma) <= 0):
            warnings.warn("Warning: Sigma <= 0 found, clamped to 1e-6.")
            sigma = np.clip(sigma, a_min=1e-6, a_max=None)

        return rng.lognormal(mean=mu, sigma=sigma, size=n)

    def mean(self, n: int = 200_000, *, rng: Optional[np.random.Generator] = None,  seed: Union[int, None] = None) -> float:
        if not isinstance(self.mu, Sampleable) and not isinstance(self.sigma, Sampleable):
            mu = float(self.mu)
            sigma = float(self.sigma)
            return float(np.exp(mu + 0.5 * sigma**2))

        if rng is None:
            rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

        mu_s = self.mu.sample(n, rng=rng) if isinstance(self.mu, Sampleable) else np.full(n, self.mu)
        sg_s = self.sigma.sample(n, rng=rng) if isinstance(self.sigma, Sampleable) else np.full(n, self.sigma)

        sg_s = np.clip(sg_s, 1e-6, None)
        return float(np.mean(np.exp(mu_s + 0.5 * sg_s**2)))
    
    def var(self, n: int = 200_000, *, rng: Optional[np.random.Generator] = None, seed: Union[int, None] = None) -> float:
        if not isinstance(self.mu, Sampleable) and not isinstance(self.sigma, Sampleable):
            mu = float(self.mu)
            sigma = float(self.sigma)
            return float((np.exp(sigma**2) - 1.0) * np.exp(2.0 * mu + sigma**2))

        if rng is None:
            rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

        mu_s = self.mu.sample(n, rng=rng) if isinstance(self.mu, Sampleable) else np.full(n, self.mu)
        sg_s = self.sigma.sample(n, rng=rng) if isinstance(self.sigma, Sampleable) else np.full(n, self.sigma)

        sg_s = np.clip(sg_s, 1e-6, None)

        Ey_cond = np.exp(mu_s + 0.5 * sg_s**2)
        Vy_cond = (np.exp(sg_s**2) - 1.0) * np.exp(2.0 * mu_s + sg_s**2)

        return float(np.mean(Vy_cond) + np.var(Ey_cond, ddof=0))
