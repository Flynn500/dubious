# Dubious
Dubious is a Python library for propagating uncertainty through numerical computations. Instead of collapsing uncertain values into single numbers early, Dubious lets you represent values as probability distributions, combine them with normal arithmetic operations, and only evaluate the resulting uncertainty when you explicitly ask for it.

```python
from dubious import Normal, Beta, Uncertain, Context

normal = Normal(5, 4)
normal2 = Normal(10,2)

x = Uncertain(normal) + Uncertain(normal2)

print(f"variance: {x.var()} mean: {x.mean()} q(0.05): {x.quantile(0.05)}")
```
Rounded output: variance: 19.9, mean: 15, q(0.05): 7.7

---
The key idea behind Dubious is lazy uncertainty propagation. We don't calculate approximations and lose information at each step, instead we build a graph of operations applied to uncertain values that we traverse upon sampling. You can construct complex expressions from uncertain inputs in a simple and readable manner, and evaluate the result using Monte Carlo simulations.

By default, distributions are assumed to be independent. We can correlate two uncertain objects using `a.corr(b,rho)`, implemented via Gaussian copula (see notes for details). 

After applying any numerical operations to `Uncertain` objects, sampling and evaluation only occur when calling a function like `mean()`, `quantile()` or `sample()` is called. 

### Installation
With python v3.9+ `pip install dubious`

### Notes for the user

If several instances of the same `Uncertain` object are involved in an operation these are assumed to represent the same variable so the samples used to calculate these values for each will be identical.

Correlation between Uncertain objects is currently implemented using Gaussian Copula. This rank based correlation and the rho value used to correlate different objects is NOT the same as the pearson coefficient. 

Numpy RNG objects do not need to be provided by default, but they can be optionally provided for any function that generates its result using MC sampling methods, you can alternatively just provide a seed.


### Classes
`Distribution()`:
Currently supporting Normal, LogNormal, Beta and Uniform distributions. Distribution objects also support using other distribution objects for their parameters, although this may lead to unexpected behaviour in cases where parameters can become negative. For each you distribution can get `mean()`, `var()`, `quantile` and `sample()`.

`Uncertaint()`:
Uncertaint objects are the wrapper for distributions that allow them to be used like numeric values. Alongside being able to perform numeric operations on these uncertaint objects, they support the same properties as standard distributions (mean, variance, sampling and quantile). You can apply the exact same operations on these objects you might apply to real data, and easily calculate the propagated uncertainty that comes from using several unreliable input values.

`Context()`:
Context objects own the graph through which we manage the uncertainty propagation. You can add uncertaint objects from different contexts, although this is slightly less performant than first creating a context object, and then creating all new uncertaint objects with ctx =  _Your context object_.

---
### Some examples:

```python
from dubious.distributions import Normal
from dubious.core import Uncertain, Context

#Create a shared context.
ctx = Context()

# Define our Length distribution  (about 10 ± 1)
length_dist = Normal(10, 1)
length = Uncertain(length_dist, ctx=ctx)

# Define Width as 5 ± 0.5
width_dist = Normal(5, 0.5)
width = Uncertain(width_dist, ctx=ctx)

#Compute area using normal arithmetic
area = length * width

#Inspect the uncertainty
print("Mean area:", area.mean())
print("Variance:", area.var())
print("Some samples:", area.sample(5))
```

We can also use distribution and uncertain objects as parameters.

```python
from dubious.distributions import Normal, Beta
from dubious.core import Uncertain, Context

ctx = Context()

#We can define distribution parameters with other distributions.
normal = Normal(10, 1)
beta = Beta(3,normal)

x = Uncertain(normal, ctx=ctx)
y = Uncertain(beta, ctx=ctx)

#Apply some arithmetic.
x = x*y

print(x.sample(5))

#We can also use uncertain distributions to define parameters.
normal3 = Normal(y+2, 3)
print(normal3.mean())
```

An example of correlation:
```python
#two non-Gaussian marginals correlated using copula
from dubious.distributions import Beta, LogNormal
from dubious.core import Uncertain, Context

ctx = Context()

conv = Uncertain(Beta(20,80), ctx=ctx)
traffic = Uncertain(LogNormal(8.0,0.4), ctx=ctx)

conv.corr(traffic, 0.7)

sales = conv * traffic

print(f"Mean: {sales.mean()}")
print(f"p10, p90: {sales.quantile(0.1)}, {sales.quantile(0.9)}")
````
Mean: 685.3054124550143
p10, p90: 282.88651636845674, 1185.6903692334781

Correlated uncertainty propagation currently matches a Gaussian-copula reference to within 
~0.25% relative error on tail quantiles.

