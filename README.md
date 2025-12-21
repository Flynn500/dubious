# Dubious
<p>Dubious is a Python library for propagating uncertainty through numerical computations. Instead of collapsing uncertain values into single numbers early, Dubious lets you represent values as probability distributions, combine them with normal arithmetic operations, and only evaluate the resulting uncertainty when you explicitly ask for it.</p>

<h3>Distribution Objects</h3>
<p>Currently only supporting Normal, LogNormal, Beta and Uniform distributions. For each you can get mean, variance and samples. Distribution objects also support using other distribution objects for their parameters, although this may lead to unexpected behaviour in cases where parameters can become negative. </p>

<h3>Uncertainty Objects</h3>
Uncertainty objects are the wrapper for distributions that allow them to be used in as though they were numeric values. To create an uncertainty object you must first provide a distribution and a context object. Uncertain objects that belong to different contexts cannot be combined currently, this will change in the future. You can perform numeric operations on these uncertainty objects and sample from them via the sample_uncertain() function or .sample(). Mean variance and quantile functions are also available for these distributions.
