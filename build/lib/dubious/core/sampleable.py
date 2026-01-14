import substratum as ss
from abc import abstractmethod
from typing import Optional
from math import sqrt
from .sampler import Sampler

class Sampleable():
    @abstractmethod
    def sample(self, n: int, *, sampler: Optional[Sampler] = None) -> ss.Array:
        """
        Sample points from a distribution.

        :param n: Number of samples.
        :type n: int
        :param sampler: Dubious Sampler object.
        :type sampler: Sampler
        :return: Array of sampled points.
        :rtype: ss.Array
        """
        raise NotImplementedError

    @abstractmethod
    def mean(self) -> float:
        """
        Get the mean of a distribution.

        :return: Mean.
        :rtype: float
        """
        raise NotImplementedError

    @abstractmethod
    def var(self) -> float:
        """
        Get the variance of a distribution.

        :return: Variance.
        :rtype: float
        """
        raise NotImplementedError

    @abstractmethod
    def quantile(self, q: float, n: int = 50000, *, sampler: Optional[Sampler] = None) -> float:
        """
        Compute an approximation of the q-th quantile of data. Defaults to monte carlo simulation if other
        dsitributions are used as parameters otherwise analytic methods are used.

        :param q: Probability of the quantile to compute.
        :type q: float
        :param n: Number of samples.
        :type n: int
        :param sampler: Dubious Sampler object.
        :type sampler: Sampler
        :return: Estimated quantile value.
        :rtype: float
        """
        raise NotImplementedError

    @abstractmethod
    def cdf(self, x: float, n: int = 200_000, *, sampler: Optional[Sampler] = None) -> float:
        """
        Compute an approximation of the cumulative density function. Defaults to monte carlo simulation if other
        distributions are used as parameters otherwise analytic methods are used.

        :param x: Value.
        :type q: float
        :param x: Numer of samples.
        :type n: int
        :param sampler: Dubious Sampler object.
        :type sampler: Sampler
        :return: Estimated CDF.
        :rtype: float
        """
        raise NotImplementedError

    def summary(self):
        """
        Print a summary of summary of statistics.
        """
        print(f"Summary for type: {self.__class__.__name__}")
        print("---------------------------\n" \
            f"mean      : {self.mean():.3f}\n" \
            f"median    : {self.quantile(0.5):.3f}\n" \
            f"std       : {sqrt(self.var()):.3f}\n" \
            f"variance  : {self.var():.3f}\n" \
            f"q05 / q95 : {self.quantile(0.05):.3f} / {self.quantile(0.95):.3f}\n" \
            f"q25 / q75 : {self.quantile(0.25):.3f} / {self.quantile(0.75):.3f}\n"
        )

class Distribution(Sampleable):
    @abstractmethod
    def sample(self, n: int, *, sampler: Optional[Sampler] = None, advance: int = 0) -> ss.Array:
        """
        Sample points from a distribution.

        :param n: Number of samples.
        :type n: int
        :param sampler: Dubious Sampler object.
        :type sampler: Sampler
        :return: Array of sampled points.
        :rtype: ss.Array
        """
        raise NotImplementedError

    @abstractmethod
    def mean(self) -> float:
        """
        Get the mean of a distribution. Defaults to monte carlo simulation if other
        dsitributions are used as parameters otherwise analytic methods are used.

        :return: Mean.
        :rtype: float
        """
        raise NotImplementedError

    @abstractmethod
    def var(self) -> float:
        """
        Get the variance of a distribution. Defaults to monte carlo simulation if other
        dsitributions are used as parameters otherwise analytic methods are used.

        :return: Variance.
        :rtype: float
        """
        raise NotImplementedError

    def quantile(self, q: float, n: int = 50000, *, sampler: Optional[Sampler] = None) -> float:
        """
        Compute an approximation of the q-th quantile of data. Defaults to monte carlo simulation if other
        distributions are used as parameters otherwise analytic methods are used.

        :param q: Probability of the quantile to compute.
        :type q: float
        :param n: Number of samples.
        :type n: int
        :param sampler: Dubious Sampler object.
        :type sampler: Sampler
        :return: Estimated quantile value.
        :rtype: float
        """
        if q < 0.0 or q > 1.0:
            raise ValueError("q must be between 0 and 1")

        s = self.sample(n, sampler=sampler)
        return s.quantile(q)

    def cdf(self, x: float, n: int = 200_000, *, sampler: Optional[Sampler] = None) -> float:
        """
        Compute an approximation of the cumulative density function. Defaults to monte carlo simulation if other
        distributions are used as parameters otherwise analytic methods are used.

        :param x: Value.
        :type q: float
        :param x: Numer of samples.
        :type n: int
        :param sampler: Dubious Sampler object.
        :type sampler: Sampler
        :return: Estimated CDF.
        :rtype: float
        """
        s = self.sample(n, sampler=sampler)
        # Count how many samples are <= x
        count = sum(1 for val in s if val <= x)
        return count / len(s)
