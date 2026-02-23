import warnings
import ironforest as irn
from typing import Union, Optional
import numbers

from ..core.sampleable import Sampleable, Distribution
from ..core.sampler import Sampler
from .dist_helpers import _mean, _var

class Normal(Distribution):
    def __init__(self, mu: Union[float, Sampleable] = 0.0, sigma: Union[float, Sampleable] = 1.0):
        if isinstance(sigma, numbers.Real) and sigma <= 0:
            raise ValueError("sigma must be positive.")
        self.mu = mu
        self.sigma = sigma

    def sample(self, n: int, *, sampler: Optional[Sampler] = None) -> irn.Array:
        if sampler is None:
            sampler = Sampler()

        if isinstance(self.mu, Sampleable):
            mu = self.mu.sample(n, sampler=sampler)
        else:
            mu = self.mu

        if isinstance(self.sigma, Sampleable):
            sigma = self.sigma.sample(n, sampler=sampler)
        else:
            sigma = self.sigma

        # Check for invalid sigma values
        if isinstance(sigma, irn.Array):
            if any(val <= 0 for val in sigma):
                warnings.warn("Warning: Sigma <= 0 found, clamped to 1e-6.")
                sigma = sigma.clip(1e-6, 1e308)
        elif sigma <= 0:
            warnings.warn("Warning: Sigma <= 0 found, clamped to 1e-6.")
            sigma = 1e-6

        return sampler.normal(loc=mu, scale=sigma, size=[n])

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
