import ironforest as irn
from typing import Union
import numbers
from ..core.sampleable import Sampleable


def _mean(x: Union[float, Sampleable]) -> float:
    return x.mean() if isinstance(x, Sampleable) else float(x)

def _var(x: Union[float, Sampleable]) -> float:
    return x.var() if isinstance(x, Sampleable) else 0.0

def _is_scalar_real(x: object) -> bool:
    return isinstance(x, numbers.Real) and not isinstance(x, irn.Array)
