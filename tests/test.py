import numpy as np
from dubious import Normal, LogNormal, Uncertain, sample_uncertain


normal = Normal(1, 0.1)
log_normal = Normal(10, 0.1)

rng = np.random.Generator(np.random.PCG64())

x = Uncertain(normal)
y = Uncertain(log_normal)

result = x + y

print(sample_uncertain(result, 10, rng))


