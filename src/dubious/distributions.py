import numpy as np

class Distribution:
    def sample(self, n, rng) -> np.ndarray:
        raise NotImplementedError
    
    def mean(self) -> float:
        raise NotImplementedError
    
    def var(self) -> float:
        raise NotImplementedError
    
class Normal(Distribution):
    def __init__(self, mu: float = 0.0, sigma: float = 1.0):
        if sigma <= 0:
            raise ValueError("sigma must be positive.")
        self.mu = mu
        self.sigma = sigma

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(loc=self.mu, scale=self.sigma, size=n)

    def mean(self) -> float:
        return self.mu

    def var(self) -> float:
        return self.sigma ** 2


class Uniform(Distribution):
    def __init__(self, low: float = 0.0, high: float = 1.0):
        if high <= low:
            raise ValueError("high must be greater than low.")
        self.low = low
        self.high = high

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.uniform(low=self.low, high=self.high, size=n)

    def mean(self) -> float:
        return 0.5 * (self.low + self.high)

    def var(self) -> float:
        return (self.high - self.low) ** 2 / 12
    
class LogNormal(Distribution):
    def __init__(self, mu: float = 0.0, sigma: float = 1.0):
        if sigma <= 0:
            raise ValueError("sigma must be positive.")
        self.mu = mu
        self.sigma = sigma

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.lognormal(mean=self.mu, sigma=self.sigma, size=n)

    def mean(self) -> float:
        return np.exp(self.mu + 0.5 * self.sigma ** 2)
    
    def var(self) -> float:
        return (np.exp(self.sigma ** 2) - 1) * np.exp(2 * self.mu + self.sigma ** 2)

