import numpy as np
from numpy.typing import ArrayLike, NDArray

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