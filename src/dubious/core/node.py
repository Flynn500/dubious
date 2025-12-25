from __future__ import annotations
from typing import Any, Optional, Tuple
from enum import Enum, auto
from dataclasses import dataclass

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
    SIN = auto()
    COS = auto()
    TAN = auto()
    ASIN = auto()
    ACOS = auto()
    ATAN = auto()

@dataclass(frozen=True)
class Node:
    id: int
    op: Op
    parents: Tuple[int, ...]
    payload: Optional[Any] = None #distrubtion, constant number etc
