from __future__ import annotations
from typing import Union, Optional, overload
import math

from ..core.uncertain import Uncertain, Number

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




