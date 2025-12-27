import math
import numpy as np
import pytest

from dubious.distributions import Normal
from dubious.core import Uncertain, Context, Sampler
from dubious.umath import sin

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
    sampler = Sampler(seed=1)

    m = z.mean(n=n) if "n" in z.mean.__code__.co_varnames else z.mean()
    v = z.var(sampler=sampler, n=n) if "n" in z.var.__code__.co_varnames else z.var(sampler=sampler)

    assert abs(m - 15) < 0.2
    assert abs(v - (16 + 4)) < 1.0

def test_same_uncertain_reuse_is_same_variable():
    ctx = Context()
    sampler = Sampler(seed=1)

    x = Uncertain(Normal(0, 1), ctx=ctx)
    y = x - x
    v = y.var(sampler=sampler, n=20000) if "n" in y.var.__code__.co_varnames else y.var(sampler=sampler, n=20000)
    assert v < 1e-3

def test_different_uncertain_nodes_are_independent():
    ctx = Context()
    sampler = Sampler(seed=1)

    x = Uncertain(Normal(0, 1), ctx=ctx)
    y = Uncertain(Normal(0, 1), ctx=ctx)
    z = x - y
    v = z.var(sampler=sampler, n=20000) if "n" in z.var.__code__.co_varnames else z.var(sampler=sampler, n=20000)
    assert abs(v - 2.0) < 0.2

def test_umath_sin_matches_numpy_on_samples():
    ctx = Context()
    x = Uncertain(Normal(0, 1), ctx=ctx)
    n = 5000
    seed = 42
    sampler = Sampler(seed=seed)
    sampler2 = Sampler(seed=seed)
    s1 = x.sample(n, sampler=sampler)
    s2 = sin(x).sample(n, sampler=sampler2)

    assert np.allclose(s2, np.sin(s1), rtol=0, atol=1e-12)
