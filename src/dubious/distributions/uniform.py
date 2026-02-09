import ironforest as irn
from typing import Union, Optional

from ..core.sampleable import Sampleable, Distribution
from ..core.sampler import Sampler
from .dist_helpers import _is_scalar_real, _mean, _var

class Uniform(Distribution):
    def __init__(self, low: Union[float, Sampleable] = 0.0, high: Union[float, Sampleable] = 1.0):
        if _is_scalar_real(high) and _is_scalar_real(low):
            if high <= low:  # type: ignore
                raise ValueError("high must be greater than low.")
        self.low = low
        self.high = high

    def sample(self, n: int, *, sampler: Optional[Sampler] = None) -> irn.Array:
        if sampler is None:
            sampler = Sampler()

        if isinstance(self.high, Sampleable):
            high = self.high.sample(n, sampler=sampler)
        else:
            high = self.high

        if isinstance(self.low, Sampleable):
            low = self.low.sample(n, sampler=sampler)
        else:
            low = self.low

        return sampler.uniform(low=low, high=high, size=[n])

    def mean(self) -> float:
        if isinstance(self.high, Sampleable):
            h = self.high.mean()
        else:
            h = self.high

        if isinstance(self.low, Sampleable):
            l = self.low.mean()
        else:
            l = self.low

        return 0.5 * (l + h)

    def var(self) -> float:
        low_m, high_m = _mean(self.low), _mean(self.high)
        low_v, high_v = _var(self.low), _var(self.high)

        term1 = (low_v + high_v + (high_m - low_m) ** 2) / 12.0
        term2 = (low_v + high_v) / 4.0
        return term1 + term2
