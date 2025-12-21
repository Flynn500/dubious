import numpy as np
from dubious import Normal, LogNormal, Uncertain, sample_uncertain, Context


normal = Normal(1, 1)
normal2 = Normal(5, 1)

rng = np.random.Generator(np.random.PCG64())
ctx = Context()

x = Uncertain(normal, ctx=ctx)
y = Uncertain(normal2,ctx=ctx)

normal3 = Normal(y+2, 3)

x = x*y

print(normal3.mean())

