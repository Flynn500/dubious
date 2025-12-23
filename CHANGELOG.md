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
Uncertainty objects
- Standard arithmetic through dunder methods
