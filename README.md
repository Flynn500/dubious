# Dubious
<p>A python library for handling uncertain data. In cases where you need to apply numeric operations to several variables, each with varying degrees of uncertainty, you can instead apply those numeric operations to their distributions. This information is held in a graph, and only when wish to sample or get metrics is their combined distribution calculated</p>

<h3>Distribution Objects</h3>
<p>Currenlty only supporting Normal, LogNormal and Uniform distributions. For each you can get mean, variance and sample from them. Distribution objects also support using other distribution objects for their parameters</p>

<h3>Uncertainty Object</h3>
To create an uncertainty object you must first provide a distribution. You can perform numeric operations on these distributions and sample from them via the sample_uncertain() function.
