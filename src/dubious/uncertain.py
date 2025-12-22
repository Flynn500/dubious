from __future__ import annotations
import numpy as np
from typing import Any, Optional, Tuple, Union, Literal, cast
from .node import Node, Op

from .context import Context
from .distributions import Distribution
from .sampleable import Sampleable
Number = Union[int, float, np.number]


class Uncertain(Sampleable):
    def __init__(self, dist: Optional[Distribution] = None, *, ctx: Optional[Context] = None,_node: Optional[Node] = None,):
        if ctx is None:
            self._ctx = Context()
        else: self._ctx = ctx

        if _node is not None:
            if ctx is None:
                raise ValueError("ctx must be provided when constructing from an existing _node.")
            
            self._node = _node

        else:
            if dist is None:
                raise ValueError("Distribution required.")
            
            self._node = self._ctx.add_node(Op.LEAF, parents=(), payload=dist)
    
    @property
    def node_id(self): return self._node.id

    @property
    def node(self): return self._node

    @property
    def ctx(self) -> Context:
        return self._ctx

    #create uncertain objects from numbers
    @staticmethod
    def const(x: Number, ctx: Context) -> Uncertain:
        node = ctx.add_node(Op.CONST, parents=(), payload=float(x))
        return Uncertain(ctx=ctx, _node=node)

    #create an Uncertain object if we recieve a number, else it already is an uncertain so return
    @staticmethod
    def _coerce(other: Union[Uncertain, Number], ctx: Context) -> Uncertain:
        return other if isinstance(other, Uncertain) else Uncertain.const(other, ctx=ctx)

    
    @staticmethod
    def _ensure_same_ctx(a: "Uncertain", b: "Uncertain"):
        """
        Legacy function, context merging is possible so uneeded, keeping because merging can be expensive.
        May provide some way to force the same context to be used in cases where performance is an issue.
        """
        if a.ctx is not b.ctx:
            raise ValueError(
                "Cannot combine Uncertain values from different contexts. "
                "Create them with the same ctx=..."
            )

    @staticmethod
    def _align_contexts(a: "Uncertain", b: "Uncertain") -> tuple["Uncertain", "Uncertain"]:
        if a.ctx is b.ctx:
            return a, b

        merged = Context()

        a_new_id = merged.copy_subgraph_from(a.ctx, a.node_id)
        b_new_id = merged.copy_subgraph_from(b.ctx, b.node_id)

        return (
            Uncertain(ctx=merged, _node=merged.get(a_new_id)),
            Uncertain(ctx=merged, _node=merged.get(b_new_id)),
        )


    #statistical methods
    def sample(self, n: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """
        Sample points from a distribution
        Args:
            n (int): Number of samples.
            rng (np.random.Generator): Numpy random generator.
        Returns:
            np.ndarray: Array of sampled points.
        """
        if rng is None:
            rng = np.random.default_rng()
        return sample_uncertain(self, n, rng=rng)

    def mean(self, n: int = 20_000, rng: Optional[np.random.Generator] = None) -> float:
        """
        Get the mean of a distribution
        Returns:
            float: mean
        """
        if rng is None:
            rng = np.random.default_rng()
        s = self.sample(n, rng)
        return float(np.mean(s))
    
    def var(self, n: int = 20_000, rng: Optional[np.random.Generator] = None, ddof: int = 1):
        """
        Get the variance of a distribution
        Returns:
            float: variance
        """
        if rng is None:
            rng = np.random.default_rng()
        s = self.sample(n, rng)
        return float(np.var(s, ddof=ddof))
    
    def quantile(self, q: float, n: int = 50_000, rng: Optional[np.random.Generator] = None, method: str = "linear",) -> float:
        """
        Compute the q-th quantile of data.
        Args:
            q (float): Probabilty of quantiles to compute.
            n (int): Number of samples.
            rng (np.random.Generator): Numpy random generator.
        Returns:
            float: quantile
        """
        if not (0.0 <= q <= 1.0):
            raise ValueError("q must be between 0 and 1.")
        if rng is None:
            rng = np.random.default_rng()
        s = self.sample(n, rng)

        #cast to avoid numpy getting mad
        method_lit = cast(
            Literal[
                "inverted_cdf", "averaged_inverted_cdf",
                "closest_observation", "interpolated_inverted_cdf",
                "hazen", "weibull", "linear", "median_unbiased",
                "normal_unbiased"
            ],
            method,
        )
        return float(np.quantile(s, q, method=method_lit))


    #our arithmatic operations
    def __add__(self, other: Union["Uncertain", Number]) -> "Uncertain":
        if not isinstance(other, Uncertain):
            o = Uncertain._coerce(other, ctx=self.ctx)
            node = self.ctx.add_node(Op.ADD, parents=(self.node_id, o.node_id))
            return Uncertain(ctx=self.ctx, _node=node)
        
        a, b = Uncertain._align_contexts(self, other)
        node = a.ctx.add_node(Op.ADD, parents=(a.node_id, b.node_id))
        return Uncertain(ctx=a.ctx, _node=node)

    def __radd__(self, other: Union["Uncertain", Number]) -> "Uncertain":
        return self.__add__(other)

    def __sub__(self, other: Union["Uncertain", Number]) -> "Uncertain":
        if not isinstance(other, Uncertain):
            o = Uncertain._coerce(other, ctx=self.ctx)
            node = self.ctx.add_node(Op.SUB, parents=(self.node_id, o.node_id))
            return Uncertain(ctx=self.ctx, _node=node)
        
        a, b = Uncertain._align_contexts(self, other)
        node = a.ctx.add_node(Op.SUB, parents=(a.node_id, b.node_id))
        return Uncertain(ctx=a.ctx, _node=node)

    def __rsub__(self, other: Union["Uncertain", Number]) -> "Uncertain":
        if not isinstance(other, Uncertain):
            o = Uncertain._coerce(other, ctx=self.ctx)
            node = self.ctx.add_node(Op.SUB, parents=(o.node_id, self.node_id))
            return Uncertain(ctx=self.ctx, _node=node)

        a, b = Uncertain._align_contexts(other, self)
        node = a.ctx.add_node(Op.SUB, parents=(a.node_id, b.node_id))
        return Uncertain(ctx=a.ctx, _node=node)

    def __mul__(self, other: Union["Uncertain", Number]) -> "Uncertain":
        if not isinstance(other, Uncertain):
            o = Uncertain._coerce(other, ctx=self.ctx)
            node = self.ctx.add_node(Op.MUL, parents=(self.node_id, o.node_id))
            return Uncertain(ctx=self.ctx, _node=node)
        
        a, b = Uncertain._align_contexts(self, other)
        node = a.ctx.add_node(Op.MUL, parents=(a.node_id, b.node_id))
        return Uncertain(ctx=a.ctx, _node=node)
    
    def __rmul__(self, other: Union["Uncertain", Number]) -> "Uncertain":
        return self.__mul__(other)

    def __truediv__(self, other: Union["Uncertain", Number]) -> "Uncertain":
        if not isinstance(other, Uncertain):
            o = Uncertain._coerce(other, ctx=self.ctx)
            node = self.ctx.add_node(Op.DIV, parents=(self.node_id, o.node_id))
            return Uncertain(ctx=self.ctx, _node=node)
        
        a, b = Uncertain._align_contexts(self, other)
        node = a.ctx.add_node(Op.DIV, parents=(a.node_id, b.node_id))
        return Uncertain(ctx=a.ctx, _node=node)
    
    def __rtruediv__(self, other: Union["Uncertain", Number]) -> "Uncertain":
        if not isinstance(other, Uncertain):
            o = Uncertain._coerce(other, ctx=self.ctx)
            node = self.ctx.add_node(Op.DIV, parents=(o.node_id, self.node_id))
            return Uncertain(ctx=self.ctx, _node=node)

        a, b = Uncertain._align_contexts(other, self)
        node = a.ctx.add_node(Op.DIV, parents=(a.node_id, b.node_id))
        return Uncertain(ctx=a.ctx, _node=node)
    
    def __neg__(self) -> "Uncertain":
        node = self.ctx.add_node(Op.NEG, parents=(self.node_id,))
        return Uncertain(ctx=self.ctx, _node=node)

    def __pow__(self, power: Union["Uncertain", Number]) -> "Uncertain":
        if not isinstance(power, Uncertain):
            p = Uncertain._coerce(power, ctx=self.ctx)
            node = self.ctx.add_node(Op.POW, parents=(self.node_id, p.node_id))
            return Uncertain(ctx=self.ctx, _node=node)
        
        a, b = Uncertain._align_contexts(self, power)
        node = a.ctx.add_node(Op.POW, parents=(a.node_id, b.node_id))
        return Uncertain(ctx=a.ctx, _node=node)

    def __rpow__(self, other: Union["Uncertain", Number]) -> "Uncertain":
        if not isinstance(other, Uncertain):
            o = Uncertain._coerce(other, ctx=self.ctx)
            node = self.ctx.add_node(Op.POW, parents=(o.node_id, self.node_id))
            return Uncertain(ctx=self.ctx, _node=node)
        
        a, b = Uncertain._align_contexts(other, self)
        node = a.ctx.add_node(Op.POW, parents=(a.node_id, b.node_id))
        return Uncertain(ctx=a.ctx, _node=node)

    

def sample_uncertain(u: Uncertain,n: int, *, rng: Optional[np.random.Generator] = None, seed: int = 0) -> np.ndarray:
    """
    Sample points from a composite distribution.
        Args:
            u (Uncertain) Uncertain object to sample.
            n (int): Number of samples.
            rng (np.random.Generator): Numpy random generator.
    Returns:
        np.ndarray: Array of sampled points.
    """
    cache: dict[int, np.ndarray] = {}
    ctx = u.ctx

    if rng is None:
        rng = np.random.default_rng(seed)
    
    def eval_node(node_id: int) -> np.ndarray:
        if node_id in cache:
            return cache[node_id]

        node = ctx.get(node_id)

        if node.op == Op.LEAF:
            if node.payload is None:
                raise RuntimeError("LEAF node has no payload.")
            result = node.payload.sample(n, rng=rng)

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
