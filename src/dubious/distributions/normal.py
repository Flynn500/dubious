import warnings
import numpy as np
from typing import Union, Optional
import numbers

from ..core.sampleable import Sampleable, Distribution
from .dist_helpers import _mean, _var

class Normal(Distribution):
    def __init__(self, mu: Union[float, Sampleable] = 0.0, sigma: Union[float, Sampleable] = 1.0):
        if isinstance(sigma, numbers.Real) and sigma <= 0:
            raise ValueError("sigma must be positive.")
        self.mu = mu
        self.sigma = sigma

    def sample(self, n: int, *, rng: Optional[np.random.Generator] = None, seed: Union[int, None] = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

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