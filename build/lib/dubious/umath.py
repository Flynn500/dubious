from __future__ import annotations
from typing import Union, Optional, overload
import math
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .uncertain import Uncertain, Number

@overload
def log(x: Uncertain) -> Uncertain: ...
@overload
def log(x: Number) -> float: ...
def log(x: Union[Uncertain, Number], base: float | None = None) -> Uncertain | float:
    if isinstance(x, Uncertain):
        return x.log(base=base)
    return math.log(x)

@overload
def sin(x: Uncertain) -> Uncertain: ...
@overload
def sin(x: Number) -> float: ...
def sin(x: Union[Uncertain, Number]) -> Uncertain | float:
    if isinstance(x, Uncertain):
        return x.sin()
    return math.sin(x)

@overload
def cos(x: Uncertain) -> Uncertain: ...
@overload
def cos(x: Number) -> float: ...
def cos(x: Union[Uncertain, Number]) -> Uncertain | float:
    if isinstance(x, Uncertain):
        return x.cos()
    return math.cos(x)

@overload
def tan(x: Uncertain) -> Uncertain: ...
@overload
def tan(x: Number) -> float: ...
def tan(x: Union[Uncertain, Number]) -> Uncertain | float:
    if isinstance(x, Uncertain):
        return x.tan()
    return math.tan(x)

@overload
def asin(x: Uncertain) -> Uncertain: ...
@overload
def asin(x: Number) -> float: ...
def asin(x: Union[Uncertain, Number]) -> Uncertain | float:
    if isinstance(x, Uncertain):
        return x.asin()
    return math.asin(x)

@overload
def acos(x: Uncertain) -> Uncertain: ...
@overload
def acos(x: Number) -> float: ...
def acos(x: Union[Uncertain, Number]) -> Uncertain | float:
    if isinstance(x, Uncertain):
        return x.acos()
    return math.acos(x)

@overload
def atan(x: Uncertain) -> Uncertain: ...
@overload
def atan(x: Number) -> float: ...
def atan(x: Union[Uncertain, Number]) -> Uncertain | float:
    if isinstance(x, Uncertain):
        return x.atan()
    return math.atan(x)


def erf(x: ArrayLike) -> NDArray[np.float64]:
    x_f: NDArray[np.float64] = np.asarray(x, dtype=np.float64)
    p  = 0.3275911
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429

    sign: NDArray[np.float64] = np.sign(x_f).astype(np.float64, copy=False)
    ax: NDArray[np.float64] = np.abs(x_f) 

    t: NDArray[np.float64] = 1.0 / (1.0 + p * ax)

    poly: NDArray[np.float64] = (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t)

    # Write as (-ax * ax) instead of -(ax * ax) to placate some type stubs
    y: NDArray[np.float64] = 1.0 - poly * np.exp((-ax) * ax)

    return sign * y

