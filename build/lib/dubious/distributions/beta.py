import warnings
import numpy as np
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

    def sample(self, n: int, *, sampler: Optional[Sampler] = None) -> np.ndarray:
        sampler = Sampler()

        if isinstance(self.alpha, Sampleable):
            a = self.alpha.sample(n, sampler=sampler)
        else:
            a = self.alpha

        if isinstance(self.beta, Sampleable):
            b = self.beta.sample(n, sampler=sampler)
        else:
            b = self.beta
        
        a_arr = np.asarray(a)
        b_arr = np.asarray(b)

        bad = (a_arr <= 0) | (b_arr <= 0)
        if np.any(bad):
            warnings.warn("Warning: alpha <= 0 or beta <= 0 found, clamped to 1e-6.")
            a_arr = np.clip(a_arr, a_min=1e-6, a_max=None)
            b_arr = np.clip(b_arr, a_min=1e-6, a_max=None)

        return sampler.beta(a_arr, b_arr, size=n)
    
    def mean(self, n=200_000, *, sampler: Optional[Sampler] = None) -> float:
        if not isinstance(self.alpha, Sampleable) and not isinstance(self.beta, Sampleable):
            a = float(self.alpha)
            b = float(self.beta)
            return a / (a + b)
        
        a = self.alpha.sample(n, sampler=sampler) if isinstance(self.alpha, Sampleable) else float(self.alpha)
        b = self.beta.sample(n, sampler=sampler) if isinstance(self.beta, Sampleable) else float(self.beta)
        a = np.clip(np.asarray(a), 1e-6, None)
        b = np.clip(np.asarray(b), 1e-6, None)
        return float(np.mean(a / (a + b)))
    
    def var(self, n: int = 200_000, *, sampler: Optional[Sampler] = None) -> float:
        if not isinstance(self.alpha, Sampleable) and not isinstance(self.beta, Sampleable):
            a = float(self.alpha)
            b = float(self.beta)
            denom = (a + b) ** 2 * (a + b + 1.0)
            return (a * b) / denom

        a = self.alpha.sample(n, sampler=sampler) if isinstance(self.alpha, Sampleable) else float(self.alpha)
        b = self.beta.sample(n, sampler=sampler) if isinstance(self.beta, Sampleable) else float(self.beta)
        a = np.clip(np.asarray(a), 1e-6, None)
        b = np.clip(np.asarray(b), 1e-6, None)

        s = a + b
        m = a / s
        v = (a * b) / (s * s * (s + 1.0))

        return float(np.mean(v) + np.var(m, ddof=0))






