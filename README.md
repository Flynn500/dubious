# Dubious
<p>Dubious is a Python library for propagating uncertainty through numerical computations. Instead of collapsing uncertain values into single numbers early, Dubious lets you represent values as probability distributions, combine them with normal arithmetic operations, and only evaluate the resulting uncertainty when you explicitly ask for it.</p>

<h3>Distribution Objects</h3>
<p>Currenlty only supporting Normal, LogNormal and Uniform distributions. For each you can get mean, variance and sample from them. Distribution objects also support using other distribution objects for their parameters</p>

<h3>Uncertainty Objects</h3>
To create an uncertainty object you must first provide a distribution and preferably a context object. Uncertain objects that belong to different contexts cannot be combined currently, this will change in the future. You can perform numeric operations on these uncertainty objects and sample from them via the sample_uncertain() function or .sample().
