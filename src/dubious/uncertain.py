from __future__ import annotations
import numpy as np
from typing import Any, Optional, Tuple, Union
from enum import Enum, auto
from dataclasses import dataclass
import itertools


from dubious import Distribution

Number = Union[int, float, np.number]

_node_ids = itertools.count(1)
_node_registry: dict[int, Node] = {}

class Op(Enum):
    LEAF = auto()
    CONST = auto()
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    NEG = auto()
    POW = auto()

@dataclass(frozen=True)
class Node:
    id: int
    op: Op
    parents: Tuple[int, ...]
    payload: Optional[Any] = None #distrubtion, constant number etc

    def __post_init__(self):
        _node_registry[self.id] = self

class Uncertain():
    def __init__(self, dist: Optional[Distribution] = None, _node: Optional[Node] = None,):
        if _node is not None:
            self._node = _node
        else:
            if dist is None:
                raise ValueError("Distribution required.")
            self._node = Node(
                id=next(_node_ids),
                op=Op.LEAF,
                parents=(),
                payload=dist,
            )

    @property
    def node_id(self): return self._node.id

    @property
    def node(self): return self._node

    #create uncertain objects from numbers
    @staticmethod
    def const(x: Number) -> Uncertain:
        return Uncertain(_node=Node(
            id=next(_node_ids),
            op=Op.CONST,
            parents=(),
            payload=float(x),
        ))

    #create an Uncertain object if we recieve a number, else it already is an uncertain so return
    @staticmethod
    def _coerce(other: Union[Uncertain, Number]) -> Uncertain:
        return other if isinstance(other, Uncertain) else Uncertain.const(other)

    #creates operation nodes within our tree, + - * / etc.
    @staticmethod
    def _make(op: Op, *parents: Uncertain, payload=None) -> Uncertain:
        return Uncertain(_node=Node(
            id=next(_node_ids),
            op=op,
            parents=tuple(p.node_id for p in parents),
            payload=payload,
        ))

    #our arithmatic operations
    def __add__(self, other: Union[Uncertain, Number]) -> Uncertain:
        o = Uncertain._coerce(other)
        return Uncertain._make(Op.ADD, self, o)

    def __radd__(self, other: Number) -> Uncertain:
        return Uncertain._coerce(other).__add__(self)

    def __sub__(self, other: Union[Uncertain, Number]) -> Uncertain:
        o = Uncertain._coerce(other)
        return Uncertain._make(Op.SUB, self, o)

    def __rsub__(self, other: Number) -> Uncertain:
        return Uncertain._coerce(other).__sub__(self)

    def __mul__(self, other: Union[Uncertain, Number]) -> Uncertain:
        o = Uncertain._coerce(other)
        return Uncertain._make(Op.MUL, self, o)

    def __rmul__(self, other: Number) -> Uncertain:
        return Uncertain._coerce(other).__mul__(self)

    def __truediv__(self, other: Union[Uncertain, Number]) -> Uncertain:
        o = Uncertain._coerce(other)
        return Uncertain._make(Op.DIV, self, o)

    def __rtruediv__(self, other: Number) -> Uncertain:
        return Uncertain._coerce(other).__truediv__(self)

    def __neg__(self) -> Uncertain:
        return Uncertain._make(Op.NEG, self)

    def __pow__(self, power: Union[Uncertain, Number]) -> Uncertain:
        p = Uncertain._coerce(power)
        return Uncertain._make(Op.POW, self, p)
    

def sample_uncertain(u: Uncertain,n: int,rng: np.random.Generator,) -> np.ndarray:
    cache: dict[int, np.ndarray] = {}

    def eval_node(node_id: int) -> np.ndarray:
        if node_id in cache:
            return cache[node_id]

        node = _node_registry[node_id]

        if node.op == Op.LEAF:
            if node.payload is None:
                raise RuntimeError("LEAF node has no payload.")
            result = node.payload.sample(n, rng)

        elif node.op == Op.CONST:
            result = np.full(n, node.payload)

        else:
            parents = [eval_node(pid) for pid in node.parents]

            if node.op == Op.ADD:
                result = parents[0] + parents[1]
            elif node.op == Op.SUB:
                result = parents[0] - parents[1]
            elif node.op == Op.MUL:
                result = parents[0] * parents[1]
            elif node.op == Op.DIV:
                result = parents[0] / parents[1]
            elif node.op == Op.NEG:
                result = -parents[0]
            elif node.op == Op.POW:
                result = parents[0] ** parents[1]
            else:
                raise ValueError(f"Unsupported op {node.op}")

        cache[node_id] = result
        return result

    return eval_node(u.node_id)
