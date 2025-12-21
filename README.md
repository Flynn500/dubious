# Dubious
<p>Dubious is a Python library for propagating uncertainty through numerical computations. Instead of collapsing uncertain values into single numbers early, Dubious lets you represent values as probability distributions, combine them with normal arithmetic operations, and only evaluate the resulting uncertainty when you explicitly ask for it.</p>

<h3>Distribution Objects</h3>
<p>Currently only supporting Normal, LogNormal, Beta and Uniform distributions. For each you can get mean, variance and samples. Distribution objects also support using other distribution objects for their parameters, although this may lead to unexpected behaviour in cases where parameters can become negative. </p>

<h3>Uncertainty Objects</h3>
<p></p>Uncertainty objects are the wrapper for distributions that allow them to be used in as though they were numeric values. To create an uncertainty object you must first provide a distribution and a context object. Uncertain objects that belong to different contexts cannot be combined currently, this will change in the future. You can perform numeric operations on these uncertainty objects and sample from them via the sample_uncertain() function or .sample(). Mean variance and quantile functions are also available for these distributions.
<br>
<br>
Some examples:</p>

```python
from dubious import Normal, Context, Uncertain

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

<p>We can also use distribution and uncertainty objects as parameters.</p>

```python
from dubious import Normal, Beta, Uncertain, Context

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

