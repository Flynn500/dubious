## 0.4
### Added
- Local sensitivity function shows the sensitivty at a specifed region within the sample space.
- Draw and __float__ methods. Uncertain objects can now be used like floats, calling redraw on the context will cycle to the next value else the same random value will be used.
- Median method for uncertain objects.
- Gamma Distribution
### Changed
- When frozen, Monte Carlo methods will default to the value of n the node / context was frozen at. Previously the different defaults for mean var etc. would error unless n was specified manually to be the same value of n frozen with.
- Changed from numpy to ironforest as the libraries back end.
## 0.3
### Added
- Added correlation via Gaussian Copula
- Added `freeze()` and `unfreeze()` functions to uncertain and context objects. They will ensure the same set of samples is used for all function calls while frozen. Context freezing is recommended in most cases as it freezes every random node in the graph as well as constants. Freezing uncertain objects individually will lead to semi-random behaviour that isn't as useful in most cases.
- Benchmarks and additional testing
- Documentation now on https://dubious.readthedocs.io/

### Changed
- Changed import structure. Instead of everything living in the main name space, we have core, distributions and umath.
- Seeds defaulted to 0 which meant that everything was deterministic by default. We now default as random and only when a seed or rng object is provided are outputs deterministic.
  
### Fixed

## 0.2
### Added
- Context objects now handle graph ownership and can be merged, e.g. Uncertain objects from different contexts can be used together.
- Uncertain objects and Distributions now inherit from Sampleable and can both be used as input parameters
- Added log, sin, cos, tan, asin, acos and atan operations for Uncertain objects in umath. Umath functions also support normal numbers.
### Fixed
- Some functions had inconsistent requirements regarding numpy generators. Now all do not require one but give the option of either providing one or a seed.
## 0.1.1
### Fixed
- 0.1 release had a major bug making most uncertain methods unusable on other machines... oops

## 0.1
- First release
### Added
Distribution objects
- Normal,
- LogNormal
- Uniform
- Beta
- Support for using distribution objects as params
- Uncertain objects
- Standard arithmetic through dunder methods
