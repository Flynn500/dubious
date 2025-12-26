from __future__ import annotations
import numpy as np
from typing import Any, Optional, Union, Literal, cast, TYPE_CHECKING
import warnings

from .node import Node, Op
from .context import Context
from .sampleable import Sampleable, Distribution
if TYPE_CHECKING:
    from .sample_session import SampleSession

Number = Union[int, float, np.number]


class Uncertain(Sampleable):
    """
    A wrapper for distribution objects that allow them to be used as though they were numeric values from said distribution.
    """
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
    def sample(self, n: int, rng: Optional[np.random.Generator] = None, seed: Union[int, None] = None) -> np.ndarray:
        """
        Sample points from a distribution
        Args:
            n (int): Number of samples.
            rng (np.random.Generator): Numpy random generator.
        Returns:
            np.ndarray: Array of sampled points.
        """
        if rng is None:
            rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        
        from .sample_session import SampleSession #hacky avoiding
        session = SampleSession(n, rng)
        return sample_uncertain(self, session)

    def mean(self, n: int = 20_000, rng: Optional[np.random.Generator] = None, seed: Union[int, None] = None) -> float:
        """
        Get the mean of a distribution
        Returns:
            float: mean
        """
        s = self.sample(n, rng, seed=seed)
        return float(np.mean(s))
    
    def var(self, n: int = 20_000, rng: Optional[np.random.Generator] = None, seed: Union[int, None] = None):
        """
        Get the variance of a distribution
        Returns:
            float: variance
        """
        s = self.sample(n, rng, seed=seed)
        return float(np.var(s, ddof=0))
    
    def quantile(self, q: Union[float, np.ndarray], n: int = 50_000, rng: Optional[np.random.Generator] = None, seed: Union[int, None] = None, method: str = "linear",) -> Union[float,np.ndarray]:
        """
        Compute an approximation of the q-th quantile of data.
        Args:
            q (float): Probabilty of quantiles to compute.
            n (int): Number of samples.
            rng (np.random.Generator): Numpy random generator.
        Returns:
            float: quantile
        """
        q = np.asarray(q)

        if np.any((q < 0.0) | (q > 1.0)):
            raise ValueError("q must be between 0 and 1")
        
        s = self.sample(n, rng, seed=seed)

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
        result = np.quantile(s, q, method=method_lit)
        return result.item() if result.ndim == 0 else result

    
    def cdf(self, x: float, n: int = 200_000, *, rng=None, seed: Union[int, None] = None) -> float:
        """
        Compute an approximation of the cumulative density function. 
        Args:
            x (float): Value.
            n (int): Number of samples.
            rng (np.random.Generator): Numpy random generator.
        Returns:
            float: quantile
        """
        if rng is None:
            rng = np.random.default_rng(seed)
        
        s = self.sample(n, rng=rng)
        return float(np.mean(s <= x))
    
    #correlation
    def corr(self, u: "Uncertain", rho: float):
        """
        Correlate this Uncertain object with another using Gaussian Copular. 
        Both objects must be a leaf nodes, meaning they have not yet had any numerical 
        operations applied to them. 
        Args:
            u (Uncertain): The object with which to correlate.
            rho (int): Gaussian copula correlation parameter.
        Returns:
            None
        """
        Uncertain._align_contexts(self, u)
        self._ctx.set_corr(self.node_id, u.node_id, rho)



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
    
    #custom numerical operations
    def log(self, base: float | None = None) -> "Uncertain":
        payload = None if base is None else float(base)
        node = self.ctx.add_node(Op.LOG, parents=(self.node_id,), payload=payload)
        return Uncertain(ctx=self.ctx, _node=node)
    
    def sin(self) -> "Uncertain":
        node = self.ctx.add_node(Op.SIN, parents=(self.node_id,))
        return Uncertain(ctx=self.ctx, _node=node)

    def cos(self) -> "Uncertain":
        node = self.ctx.add_node(Op.COS, parents=(self.node_id,))
        return Uncertain(ctx=self.ctx, _node=node)

    def tan(self) -> "Uncertain":
        node = self.ctx.add_node(Op.TAN, parents=(self.node_id,))
        return Uncertain(ctx=self.ctx, _node=node)
    
    def asin(self) -> "Uncertain":
        node = self.ctx.add_node(Op.ASIN, parents=(self.node_id,))
        return Uncertain(ctx=self.ctx, _node=node)
    
    def acos(self) -> "Uncertain":
        node = self.ctx.add_node(Op.ACOS, parents=(self.node_id,))
        return Uncertain(ctx=self.ctx, _node=node)

    def atan(self) -> "Uncertain":
        node = self.ctx.add_node(Op.ATAN, parents=(self.node_id,))
        return Uncertain(ctx=self.ctx, _node=node)
    

def sample_uncertain(u: Uncertain, session: SampleSession) -> np.ndarray:
    """
    Sample points from a composite distribution.
        Args:
            u (Uncertain) Uncertain object to sample.
            n (int): Number of samples.
            rng (np.random.Generator): Numpy random generator.
    Returns:
        np.ndarray: Array of sampled points.
    """

    ctx = u.ctx
    
    def eval_node(node_id: int) -> np.ndarray:
        if node_id in session.cache:
            return session.cache[node_id]

        node = ctx.get(node_id)

        if node.op == Op.LEAF:
            if node.payload is None:
                raise RuntimeError("LEAF node has no payload.")
            
            if not session.correlation_prepared:
                session.prepare_correlation(ctx)

            group_id = session.leaf_to_group.get(node_id)
            if group_id is not None:
                session.group_samplers[group_id](session)
                return session.cache[node_id]

            result = node.payload.sample(session.n, rng=session.rng)

        elif node.op == Op.CONST:
            result = np.full(session.n, node.payload)

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
            elif node.op == Op.LOG:
                x = parents[0]
                if np.any(x <= 0):
                    warnings.warn("Warning: Log domain <= 0 found, clamped to 1e-6.")
                    x = np.clip(x, a_min=1e-6, a_max=None)

                if node.payload is None:
                    result = np.log(x)
                else:
                    base = float(node.payload)
                    if base <= 0 or base == 1.0:
                        raise ValueError("log() base must be > 0 and != 1.")
                    result = np.log(x) / np.log(base)
            elif node.op == Op.SIN:
                result = np.sin(parents[0])
            elif node.op == Op.COS:
                result = np.cos(parents[0])
            elif node.op == Op.TAN:
                result = np.tan(parents[0])
            elif node.op == Op.ASIN:
                result = np.arcsin(parents[0])
            elif node.op == Op.ACOS:
                result = np.arccos(parents[0])
            elif node.op == Op.ATAN:
                result = np.arctan(parents[0])
            else:
                raise ValueError(f"Unsupported op {node.op}")

        session.cache[node_id] = result
        return result

    return eval_node(u.node_id)
