import warnings
import ironforest as irn
from typing import Union, Optional
import numbers

from ..core.sampleable import Sampleable, Distribution
from ..core.sampler import Sampler
from .dist_helpers import _mean, _var

class Gamma(Distribution):
    def __init__(self, shape: Union[float, Sampleable] = 1.0, scale: Union[float, Sampleable] = 1.0):
        if isinstance(shape, numbers.Real) and shape <= 0:
            raise ValueError("shape must be positive.")
        if isinstance(scale, numbers.Real) and scale <= 0:
            raise ValueError("scale must be positive.")
        self.shape = shape
        self.scale = scale

    def sample(self, n: int, *, sampler: Optional[Sampler] = None) -> irn.Array:
        if sampler is None:
            sampler = Sampler()

        if isinstance(self.shape, Sampleable):
            shape = self.shape.sample(n, sampler=sampler)
        else:
            shape = self.shape

        if isinstance(self.scale, Sampleable):
            scale = self.scale.sample(n, sampler=sampler)
        else:
            scale = self.scale

        if isinstance(shape, irn.Array):
            if any(val <= 0 for val in shape):
                warnings.warn("Warning: shape <= 0 found, clamped to 1e-6.")
                shape = shape.clip(1e-6, 1e308)
        elif shape <= 0:
            warnings.warn("Warning: shape <= 0 found, clamped to 1e-6.")
            shape = 1e-6

        if isinstance(scale, irn.Array):
            if any(val <= 0 for val in scale):
                warnings.warn("Warning: scale <= 0 found, clamped to 1e-6.")
                scale = scale.clip(1e-6, 1e308)
        elif scale <= 0:
            warnings.warn("Warning: scale <= 0 found, clamped to 1e-6.")
            scale = 1e-6

        return sampler.gamma(shape, scale, size=[n])

    def mean(self) -> float:
        shape_mean = _mean(self.shape)
        scale_mean = _mean(self.scale)

        if isinstance(self.shape, Sampleable) or isinstance(self.scale, Sampleable):
            return shape_mean * scale_mean
        else:
            return float(self.shape) * float(self.scale)

    def var(self) -> float:
        shape_mean = _mean(self.shape)
        scale_mean = _mean(self.scale)
        shape_var = _var(self.shape)
        scale_var = _var(self.scale)

        if isinstance(self.shape, Sampleable) or isinstance(self.scale, Sampleable):
            direct_var = shape_mean * (scale_var + scale_mean**2)
            e_shape2 = shape_var + shape_mean**2
            e_scale2 = scale_var + scale_mean**2
            var_of_mean = e_shape2 * e_scale2 - (shape_mean * scale_mean)**2

            return direct_var + var_of_mean
        else:
            s_shape = float(self.shape)
            s_scale = float(self.scale)
            return s_shape * s_scale**2
