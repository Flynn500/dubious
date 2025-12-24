from dataclasses import dataclass, field
from typing import Dict, Optional, Callable, Any
import numpy as np

@dataclass
class SampleSession():
    n: int
    rng: np.random.Generator
    cache: Dict[int, np.ndarray] = field(default_factory=dict)

    group_samplers: Dict[Any, Callable[["SampleSession"], None]] = field(default_factory=dict)
    leaf_to_group: Dict[int, Any] = field(default_factory=dict)

    correlation_prepared: bool = False