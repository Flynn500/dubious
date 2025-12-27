from dubious.distributions import Beta, LogNormal
from dubious.core import Uncertain, Context, Sampler

import numpy as np
from scipy.stats import norm, beta as sp_beta, lognorm as sp_lognorm

def gaussian_copula_reference_beta_lognorm(a, b, mu, sigma, r, n, rng):
    z1 = rng.standard_normal(n)
    z2 = rng.standard_normal(n)
    z2 = r * z1 + np.sqrt(1 - r*r) * z2

    u1 = norm.cdf(z1)
    u2 = norm.cdf(z2)

    x = sp_beta.ppf(u1, a, b)
    y = sp_lognorm.ppf(u2, s=sigma, scale=np.exp(mu))
    return x, y


def test_sales_matches_reference_within_error_band():
    seed = 999
    rng1 = np.random.default_rng(seed)
    rng2 = np.random.default_rng(seed)
    sampler1 = Sampler(rng=rng1)

    N = 80_000
    r = 0.7

    beta = Beta(20, 80)
    lnorm = LogNormal(8.0, 0.4)

    ctx = Context()
    conv = Uncertain(beta, ctx=ctx)
    traffic = Uncertain(lnorm, ctx=ctx)
    conv.corr(traffic, r)
    sales = conv * traffic
    sales_lib = sales.sample(N, sampler=sampler1)

    x_ref, y_ref = gaussian_copula_reference_beta_lognorm(
        a=20, b=80, mu=8.0, sigma=0.4, r=r, n=N, rng=rng2
    )
    sales_ref = x_ref * y_ref

    for p in (0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99):
        q_lib = np.quantile(sales_lib, p)
        q_ref = np.quantile(sales_ref, p)
        rel_err = abs(q_lib - q_ref) / max(1e-9, abs(q_ref))
        assert rel_err < 0.01

