import math
import numpy as np
from scipy.stats import norm, beta as sp_beta, lognorm as sp_lognorm

from dubious.distributions import Beta, LogNormal
from dubious.core import Uncertain, Context, Sampler

import substratum as ss

#note this file takes a while to run due to list conversions. Once substratum can convert to numpy directly this will be faster


def gaussian_copula_reference_beta_lognorm(a, b, mu, sigma, r, n, rng: ss.random.Generator):
    """
    Reference: Gaussian copula (Normal -> Uniform via CDF) + SciPy PPFs.
    Uses your new RNG/Array for the normal draws, then converts to NumPy
    only for SciPy's cdf/ppf.
    """
    # Draw correlated standard normals using your library
    z1 = rng.standard_normal([n])
    z2 = rng.standard_normal([n])
    z2 = z1 * r + z2 * math.sqrt(1.0 - r * r)

    # Convert to NumPy for SciPy transforms
    z1_np = np.asarray(z1.tolist(), dtype=np.float64)
    z2_np = np.asarray(z2.tolist(), dtype=np.float64)

    u1 = norm.cdf(z1_np)
    u2 = norm.cdf(z2_np)

    x = sp_beta.ppf(u1, a, b)
    y = sp_lognorm.ppf(u2, s=sigma, scale=np.exp(mu))
    return x, y


def run():
    r = 0.7
    Ns = [10_000, 50_000, 100_000]
    seeds = [0, 1, 2]

    beta = Beta(20, 80)
    lnorm = LogNormal(8.0, 0.4)

    for N in Ns:
        errs = []
        for seed in seeds:
            rng1 = ss.random.Generator.from_seed(seed)
            rng2 = ss.random.Generator.from_seed(seed)

            ctx = Context()
            x = Uncertain(beta, ctx=ctx)
            y = Uncertain(lnorm, ctx=ctx)
            x.corr(y, r)

            sales = (x * y).sample(N, sampler=Sampler(rng=rng1))

            xr, yr = gaussian_copula_reference_beta_lognorm(20, 80, 8.0, 0.4, r, N, rng2)
            ref = xr * yr

            sales_q95 = sales.quantile(0.95)
            ref_q95 = float(np.quantile(ref, 0.95))

            q95_err = abs(sales_q95 - ref_q95) / ref_q95
            errs.append(q95_err)

        print(
            f"N={N}: median err={np.median(errs):.3%}, max err={max(errs):.3%}"
        )


if __name__ == "__main__":
    run()


"""
Correlated uncertainty propagation matches a Gaussian-copula reference to within
~0.1% relative error on tail quantiles.

N=10000: median err=0.089%, max err=0.102%
N=50000: median err=0.010%, max err=0.088%
N=100000: median err=0.023%, max err=0.045% 
"""
