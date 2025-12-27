import numpy as np
from typing import Union, Optional

class Sampler():
    def __init__(self, rng: Optional[np.random.Generator] = None, seed: Union[int, None] = None):
        if rng is None:
            rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        self.rng: np.random.Generator = rng
    
    def advance(self, n: int = 1):
        self.rng.bit_generator.advance(n) # type: ignore

    #sampling
    def normal(self, loc, scale, size):
        return self.rng.normal(loc=loc, scale=scale, size=size)
    
    def standard_normal(self, size):
        return self.rng.standard_normal(size=size)
    
    def uniform(self, low, high, size):
        return self.rng.uniform(low=low, high=high, size=size)
    
    def lognormal(self, mean, sigma, size):
        return self.rng.lognormal(mean=mean, sigma=sigma, size=size)
    
    def beta(self, a_arr, b_arr, size):
        return self.rng.beta(a_arr, b_arr, size=size)

