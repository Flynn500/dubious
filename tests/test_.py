import math
import numpy as np
import pytest

from dubious import Normal, Uniform, Beta, Uncertain, Context, sin, sample_uncertain

def test_normal_moments():
    d = Normal(5, 4)  # if your Normal is (mean, std)
    assert abs(d.mean() - 5) < 1e-12
    assert abs(d.var() - 16) < 1e-12

def test_uncertain_addition_matches_analytic():
    ctx = Context()
    x = Uncertain(Normal(5, 4), ctx=ctx)
    y = Uncertain(Normal(10, 2), ctx=ctx)
    z = x + y

    n = 20000
    seed = 123

    m = z.mean(n=n) if "n" in z.mean.__code__.co_varnames else z.mean()
    v = z.var(seed=seed, n=n) if "n" in z.var.__code__.co_varnames else z.var(seed=seed)

    assert abs(m - 15) < 0.2
    assert abs(v - (16 + 4)) < 1.0

def test_same_uncertain_reuse_is_same_variable():
    ctx = Context()
    x = Uncertain(Normal(0, 1), ctx=ctx)
    y = x - x
    v = y.var(seed=1, n=20000) if "n" in y.var.__code__.co_varnames else y.var(seed=1, n=20000)
    assert v < 1e-3

def test_different_uncertain_nodes_are_independent():
    ctx = Context()
    x = Uncertain(Normal(0, 1), ctx=ctx)
    y = Uncertain(Normal(0, 1), ctx=ctx)
    z = x - y
    v = z.var(seed=1, n=20000) if "n" in z.var.__code__.co_varnames else z.var(seed=1, n=20000)
    assert abs(v - 2.0) < 0.2

def test_umath_sin_matches_numpy_on_samples():
    ctx = Context()
    x = Uncertain(Normal(0, 1), ctx=ctx)
    n = 5000
    seed = 42

    s1 = x.sample(n, seed=seed)
    s2 = sin(x).sample(n, seed=seed)

    assert np.allclose(s2, np.sin(s1), rtol=0, atol=1e-12)
