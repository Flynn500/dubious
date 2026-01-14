import substratum as ss
from typing import Union, Optional

class Sampler():
    def __init__(self, rng: Optional[ss.random.Generator] = None, seed: Union[int, None] = None):
        if rng is None:
            rng = ss.random.Generator.from_seed(seed) if seed is not None else ss.random.Generator()
        self.rng: ss.random.Generator = rng

    #sampling
    def normal(self, loc, scale, size) -> ss.Array:
        return self.rng.normal(loc, scale, size)

    def standard_normal(self, size) -> ss.Array:
        return self.rng.standard_normal(size)

    def uniform(self, low, high, size) -> ss.Array:
        return self.rng.uniform(low, high, size)

    def lognormal(self, mean, sigma, size) -> ss.Array:
        return self.rng.lognormal(mean, sigma, size)

    def beta(self, a_arr, b_arr, size) -> ss.Array:
        return self.rng.beta(a_arr, b_arr, size)

