import ironforest as ss
from typing import Union, Sequence

"""
This file should not import any core modules. Stats modules should be strictly dealing with Arrays
conversions should happen in the core modules if needed.
"""

def erf(x: Union[ss.Array, Sequence[float]]) -> ss.Array:
    if not isinstance(x, ss.Array):
        x_f = ss.ndutils.asarray(list(x))
    else:
        x_f = x

    p  = 0.3275911
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429

    sign = x_f.sign()
    ax = x_f.abs()

    t = 1.0 / (1.0 + p * ax)

    poly = (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t)

    y = 1.0 - poly * ((-ax) * ax).exp()

    return sign * y