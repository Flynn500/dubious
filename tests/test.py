import numpy as np
from dubious import Normal, LogNormal, Uncertain, sample_uncertain, Context


normal = Normal(1, 0.1)
log_normal = Normal(10, 0.1)

rng = np.random.Generator(np.random.PCG64())
ctx = Context()
x = Uncertain(normal, ctx=ctx)
y = Uncertain(log_normal,ctx=ctx)

result = x + y

print(sample_uncertain(result, 10, rng))


