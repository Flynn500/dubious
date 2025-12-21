from __future__ import annotations
import numpy as np
from typing import Any, Optional, Tuple, Union, Literal, cast
from enum import Enum, auto
from dataclasses import dataclass
import itertools

class Op(Enum):
    LEAF = auto()
    CONST = auto()
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    NEG = auto()
    POW = auto()
    LOG = auto()

@dataclass(frozen=True)
class Node:
    id: int
    op: Op
    parents: Tuple[int, ...]
    payload: Optional[Any] = None #distrubtion, constant number etc

    # def __post_init__(self):
    #     _node_registry[self.id] = self