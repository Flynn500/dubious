import warnings
import ironforest as irn
import math
from typing import Union, Optional
import numbers

from ..core.sampleable import Sampleable, Distribution
from ..core.sampler import Sampler

class LogNormal(Distribution):
    def __init__(self, mu: Union[float, Sampleable] = 0.0, sigma: Union[float, Sampleable] = 1.0):
        if isinstance(sigma, numbers.Real) and sigma <= 0:
            raise ValueError("sigma must be positive.")
        self.mu = mu
        self.sigma = sigma

    def sample(self, n: int, *, sampler: Optional[Sampler] = None) -> irn.Array:
        if sampler is None:
            sampler = Sampler()

        mu = self.mu.sample(n, sampler=sampler) if isinstance(self.mu, Sampleable) else self.mu
        sigma = self.sigma.sample(n, sampler=sampler) if isinstance(self.sigma, Sampleable) else self.sigma

        # Check for invalid sigma values
        if isinstance(sigma, irn.Array):
            if any(val <= 0 for val in sigma):
                warnings.warn("Warning: Sigma <= 0 found, clamped to 1e-6.")
                sigma = sigma.clip(1e-6, 1e308)
        elif sigma <= 0:
            warnings.warn("Warning: Sigma <= 0 found, clamped to 1e-6.")
            sigma = 1e-6

        return sampler.lognormal(mean=mu, sigma=sigma, size=[n])

    def mean(self, n: int = 200_000, *, sampler: Optional[Sampler] = None) -> float:
        if not isinstance(self.mu, Sampleable) and not isinstance(self.sigma, Sampleable):
            mu = float(self.mu)
            sigma = float(self.sigma)
            return math.exp(mu + 0.5 * sigma**2)

        mu_s = self.mu.sample(n, sampler=sampler) if isinstance(self.mu, Sampleable) else irn.full([n], float(self.mu))
        sg_s = self.sigma.sample(n, sampler=sampler) if isinstance(self.sigma, Sampleable) else irn.full([n], float(self.sigma))

        sg_s = sg_s.clip(1e-6, 1e308)
        result = (mu_s + sg_s * sg_s * 0.5).exp()
        return result.mean()

    def var(self, n: int = 200_000, *, sampler: Optional[Sampler] = None) -> float:
        if not isinstance(self.mu, Sampleable) and not isinstance(self.sigma, Sampleable):
            mu = float(self.mu)
            sigma = float(self.sigma)
            return (math.exp(sigma**2) - 1.0) * math.exp(2.0 * mu + sigma**2)

        mu_s = self.mu.sample(n, sampler=sampler) if isinstance(self.mu, Sampleable) else irn.full([n], float(self.mu))
        sg_s = self.sigma.sample(n, sampler=sampler) if isinstance(self.sigma, Sampleable) else irn.full([n], float(self.sigma))

        sg_s = sg_s.clip(1e-6, 1e308)

        Ey_cond = (mu_s + sg_s * sg_s * 0.5).exp()
        Vy_cond = ((sg_s * sg_s).exp() - 1.0) * (mu_s * 2.0 + sg_s * sg_s).exp()

        return Vy_cond.mean() + Ey_cond.var()
