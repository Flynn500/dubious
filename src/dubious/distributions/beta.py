import warnings
import substratum as ss
from typing import Union, Optional
import numbers

from ..core.sampleable import Sampleable, Distribution
from ..core.sampler import Sampler

class Beta(Distribution):
    def __init__(self, alpha: Union[float, Sampleable] = 1.0, beta: Union[float, Sampleable] = 1.0):
        if isinstance(alpha, numbers.Real) and alpha <= 0:
            raise ValueError("alpha must be positive.")
        if isinstance(beta, numbers.Real) and beta <= 0:
            raise ValueError("beta must be positive.")
        self.alpha = alpha
        self.beta = beta

    def sample(self, n: int, *, sampler: Optional[Sampler] = None) -> ss.Array:
        if sampler is None:
            sampler = Sampler()

        if isinstance(self.alpha, Sampleable):
            a = self.alpha.sample(n, sampler=sampler)
        else:
            a = self.alpha

        if isinstance(self.beta, Sampleable):
            b = self.beta.sample(n, sampler=sampler)
        else:
            b = self.beta

        # Validate and clamp values
        if isinstance(a, ss.Array):
            if any(val <= 0 for val in a):
                warnings.warn("Warning: alpha <= 0 found, clamped to 1e-6.")
                a = a.clip(1e-6, 1e308)
        elif a <= 0:
            warnings.warn("Warning: alpha <= 0 found, clamped to 1e-6.")
            a = 1e-6

        if isinstance(b, ss.Array):
            if any(val <= 0 for val in b):
                warnings.warn("Warning: beta <= 0 found, clamped to 1e-6.")
                b = b.clip(1e-6, 1e308)
        elif b <= 0:
            warnings.warn("Warning: beta <= 0 found, clamped to 1e-6.")
            b = 1e-6

        return sampler.beta(a, b, size=[n])

    def mean(self, n=200_000, *, sampler: Optional[Sampler] = None) -> float:
        if not isinstance(self.alpha, Sampleable) and not isinstance(self.beta, Sampleable):
            a = float(self.alpha)
            b = float(self.beta)
            return a / (a + b)

        a = self.alpha.sample(n, sampler=sampler) if isinstance(self.alpha, Sampleable) else float(self.alpha)
        b = self.beta.sample(n, sampler=sampler) if isinstance(self.beta, Sampleable) else float(self.beta)

        if isinstance(a, ss.Array):
            a = a.clip(1e-6, 1e308)
        if isinstance(b, ss.Array):
            b = b.clip(1e-6, 1e308)

        # Calculate mean of a / (a + b)
        if isinstance(a, ss.Array) and isinstance(b, ss.Array):
            ratio = a / (a + b)
            return ratio.mean()
        elif isinstance(a, ss.Array):
            ratio = a / (a + b)
            return ratio.mean()
        elif isinstance(b, ss.Array):
            ratio = a / (a + b)
            return ratio.mean()
        else:
            return a / (a + b)

    def var(self, n: int = 200_000, *, sampler: Optional[Sampler] = None) -> float:
        if not isinstance(self.alpha, Sampleable) and not isinstance(self.beta, Sampleable):
            a = float(self.alpha)
            b = float(self.beta)
            denom = (a + b) ** 2 * (a + b + 1.0)
            return (a * b) / denom

        a = self.alpha.sample(n, sampler=sampler) if isinstance(self.alpha, Sampleable) else float(self.alpha)
        b = self.beta.sample(n, sampler=sampler) if isinstance(self.beta, Sampleable) else float(self.beta)

        if isinstance(a, ss.Array):
            a = a.clip(1e-6, 1e308)
        if isinstance(b, ss.Array):
            b = b.clip(1e-6, 1e308)

        # s = a + b, m = a / s, v = (a * b) / (s * s * (s + 1))
        if isinstance(a, ss.Array) or isinstance(b, ss.Array):
            s = a + b
            m = a / s
            v = (a * b) / (s * s * (s + 1.0))

            if isinstance(v, ss.Array) and isinstance(m, ss.Array):
                return v.mean() + m.var()
            elif isinstance(v, ss.Array):
                return v.mean() + 0.0
            else:
                return float(v)
        else:
            s = a + b
            return (a * b) / (s * s * (s + 1.0))
