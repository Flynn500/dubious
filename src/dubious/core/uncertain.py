from __future__ import annotations
import numpy as np
from typing import Any, Optional, Union, Literal, cast, TYPE_CHECKING
import warnings
import copy

from .node import Node, Op
from .context import Context
from .sampleable import Sampleable, Distribution
from .sampler import Sampler
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
            
            self._node = self._ctx._add_node(Op.LEAF, parents=(), payload=dist)
        
        self._frozen = False
        self._frozen_n: int | None = None
        self._frozen_samples: np.ndarray | None = None
    
    @property
    def node_id(self): return self._node.id

    @property
    def node(self): return self._node

    @property
    def ctx(self) -> Context:
        return self._ctx
    
    @property
    def frozen(self) -> bool:
        return self._frozen

    @property
    def frozen_n(self) -> int | None:
        return self._frozen_n

    #create uncertain objects from numbers
    @staticmethod
    def const(x: Number, ctx: Context) -> Uncertain:
        node = ctx._add_node(Op.CONST, parents=(), payload=float(x))
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

        memo_a: dict[int, int] = {}
        memo_b: dict[int, int] = {}

        a_new_id = merged._copy_subgraph_from(a.ctx, a.node_id, memo=memo_a)
        b_new_id = merged._copy_subgraph_from(b.ctx, b.node_id, memo=memo_b)

        merged._copy_corr_from(a.ctx, memo_a)
        merged._copy_corr_from(b.ctx, memo_b)

        if a.ctx.frozen or b.ctx.frozen:
            if a.ctx.frozen and b.ctx.frozen and (a.ctx.frozen_n != b.ctx.frozen_n):
                raise ValueError(f"Frozen sample length mismatch. Cannot merge two frozen contexts with different sample sizes.")
            else:
                n = a.ctx.frozen_n if a.ctx.frozen else b.ctx.frozen_n

                merged._frozen = True
                merged._frozen_n = n

                if a.ctx.frozen:
                    merged._copy_frozen_samples_from(a.ctx, memo_a)
                if b.ctx.frozen:
                    merged._copy_frozen_samples_from(b.ctx, memo_b)

        return (
            Uncertain(ctx=merged, _node=merged.get(a_new_id)),
            Uncertain(ctx=merged, _node=merged.get(b_new_id)),
        )


    #statistical methods
    def sample(self, n: int, *, sampler: Optional[Sampler] = None) -> np.ndarray:
        if self._frozen_samples is not None:
            if n != self._frozen_samples.shape[0]:
                raise ValueError(f"Frozen sample length mismatch. To change n, first unfreeze the uncertain object.")
            return self._frozen_samples
  
        if sampler is None:
            sampler = Sampler()
        
        from .sample_session import SampleSession
        session = SampleSession(n, sampler=sampler)

        return sample_uncertain(self, session)

    def mean(self, n: int = 20_000, *, sampler: Optional[Sampler] = None) -> float:
        if self.frozen and self.frozen_n:
            n = self.frozen_n
        if self.ctx.frozen and self.ctx.frozen_n:
            n = self.ctx.frozen_n

        s = self.sample(n, sampler=sampler)
        return float(np.mean(s))
    
    def var(self, n: int = 20_000, sampler: Optional[Sampler] = None):
        if self.frozen and self.frozen_n:
            n = self.frozen_n
        if self.ctx.frozen and self.ctx.frozen_n:
            n = self.ctx.frozen_n

        s = self.sample(n, sampler=sampler)
        return float(np.var(s, ddof=0))
    
    def quantile(self, q: Union[float, np.ndarray], n: int = 50_000, *, sampler: Optional[Sampler] = None, method: str = "linear",) -> Union[float,np.ndarray]:
        if self.frozen and self.frozen_n:
            n = self.frozen_n
        if self.ctx.frozen and self.ctx.frozen_n:
            n = self.ctx.frozen_n
        
        q = np.asarray(q)

        if np.any((q < 0.0) | (q > 1.0)):
            raise ValueError("q must be between 0 and 1")
        
        s = self.sample(n, sampler=sampler)

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
    
    def cdf(self, x: float, n: int = 200_000, *, sampler: Optional[Sampler] = None) -> float:
        if self.frozen and self.frozen_n:
            n = self.frozen_n
        if self.ctx.frozen and self.ctx.frozen_n:
            n = self.ctx.frozen_n
        
        s = self.sample(n, sampler=sampler)
        return float(np.mean(s <= x))
    
    def draw(self, *, sampler: Optional[Sampler] = None) -> float:
        """
        Draw a random value from an uncertain distribution. If the context is frozen,
        cycle through the values in the current frozen cache, else random values are
        drawn. The same draw value will be returned, until redraw is called on the context,
        this uncertain object, or any other uncertain object within the same context.

        calling `float()` on an uncertain object is the same as calling `draw()`. This 
        can be used to artificially run monte carlo simulations on external functions
        that aren't supported by dubious. By repeatedly passing in the uncertain object, and 
        calling redraw you will get random results from the distribution. Alternitvely, 
        if your function supports vectorized inputs, call `sample()` and pass in the result.

        :param sampler: Dubious Sampler object.
        :type sampler: Sampler
        :return: Randomly drawn float.
        :rtype: float
        """
        if sampler is None:
            sampler = Sampler()
        
        from .sample_session import SampleSession
        session = SampleSession(1, sampler=sampler)
        idx = None
        if self.frozen:
            assert self.frozen_n is not None
            idx = self.ctx.draw_idx % self.frozen_n
            if self._frozen_samples == None or self._frozen_samples.size == 0:
                self.sample(self.frozen_n)
        if self.ctx.frozen:
            assert self.ctx.frozen_n is not None
            idx = self.ctx.draw_idx % self.ctx.frozen_n
            if self.ctx._frozen_samples == None or len(self.ctx._frozen_samples) == 0:
                self.sample(self.ctx.frozen_n)

        return draw_uncertain(self, session=session, draw_idx=idx).item()
        
    def redraw(self, *, sampler: Optional[Sampler] = None):
        """
        Redraw for this node and any others within its context.

        :param sampler: Dubious Sampler object.
        :type sampler: Sampler
        :return: Randomly drawn float.
        :rtype: float
        """
        self.ctx.redraw()
        return self.draw(sampler=sampler)

    
    def freeze(self, n: int, *, sampler: Optional[Sampler] = None):
        """
        Freeze an uncertain object. Sample once and cache the result for all future 
        operations until unfreeze() or freeze() is called with a different value of n.

        This will only freeze a single uncertain object within the context. Context.Freeze()
        is recommended in most cases.

        :param n: Number of samples.
        :type n: int
        :param sampler: Dubious Sampler object.
        :type sampler: Sampler
        :return: Array of sampled points.
        :rtype: np.ndarray
        """
        if self.frozen and self.frozen_n == n: #if they call freeze with same n just return
            return
        samples = self.sample(n, sampler=sampler)
        samples.setflags(write=False)
        self._frozen_samples = samples
        self._frozen = True
        self._frozen_n = n
    
    def unfreeze(self):
        """
        Unfreeze an uncertain object, clearing it's cache.
        """
        self._frozen = False
        self._frozen_n = None
        self._frozen_samples = None

        
    #correlation
    def corr(self, u: "Uncertain", rho: float):
        """
        Correlate this Uncertain object with another using Gaussian Copular. 
        Both objects must be a leaf nodes, meaning they have not yet had any numerical 
        operations applied to them. 

        Dubious uses Gaussian copula (rank/latent-normal dependence). rho is not Pearson 
        correlation and the realized linear correlation can differ. Validate by 
        sampling if a specific dependence measure matters.

        :param u: The object with which to correlate.
        :type n: Uncertain
        :param rho: Gaussian copula correlation parameter.
        :type n: float
        """
        Uncertain._align_contexts(self, u)
        self._ctx.set_corr(self.node_id, u.node_id, rho)

    #float conversion
    def __float__(self):
        return self.draw()


    #our arithmatic operations
    def __add__(self, other: Union["Uncertain", Number]) -> "Uncertain":
        if not isinstance(other, Uncertain):
            o = Uncertain._coerce(other, ctx=self.ctx)
            node = self.ctx._add_node(Op.ADD, parents=(self.node_id, o.node_id))
            return Uncertain(ctx=self.ctx, _node=node)
        
        a, b = Uncertain._align_contexts(self, other)
        node = a.ctx._add_node(Op.ADD, parents=(a.node_id, b.node_id))
        return Uncertain(ctx=a.ctx, _node=node)

    def __radd__(self, other: Union["Uncertain", Number]) -> "Uncertain":
        return self.__add__(other)

    def __sub__(self, other: Union["Uncertain", Number]) -> "Uncertain":
        if not isinstance(other, Uncertain):
            o = Uncertain._coerce(other, ctx=self.ctx)
            node = self.ctx._add_node(Op.SUB, parents=(self.node_id, o.node_id))
            return Uncertain(ctx=self.ctx, _node=node)
        
        a, b = Uncertain._align_contexts(self, other)
        node = a.ctx._add_node(Op.SUB, parents=(a.node_id, b.node_id))
        return Uncertain(ctx=a.ctx, _node=node)

    def __rsub__(self, other: Union["Uncertain", Number]) -> "Uncertain":
        if not isinstance(other, Uncertain):
            o = Uncertain._coerce(other, ctx=self.ctx)
            node = self.ctx._add_node(Op.SUB, parents=(o.node_id, self.node_id))
            return Uncertain(ctx=self.ctx, _node=node)

        a, b = Uncertain._align_contexts(other, self)
        node = a.ctx._add_node(Op.SUB, parents=(a.node_id, b.node_id))
        return Uncertain(ctx=a.ctx, _node=node)

    def __mul__(self, other: Union["Uncertain", Number]) -> "Uncertain":
        if not isinstance(other, Uncertain):
            o = Uncertain._coerce(other, ctx=self.ctx)
            node = self.ctx._add_node(Op.MUL, parents=(self.node_id, o.node_id))
            return Uncertain(ctx=self.ctx, _node=node)
        
        a, b = Uncertain._align_contexts(self, other)
        node = a.ctx._add_node(Op.MUL, parents=(a.node_id, b.node_id))
        return Uncertain(ctx=a.ctx, _node=node)
    
    def __rmul__(self, other: Union["Uncertain", Number]) -> "Uncertain":
        return self.__mul__(other)

    def __truediv__(self, other: Union["Uncertain", Number]) -> "Uncertain":
        if not isinstance(other, Uncertain):
            o = Uncertain._coerce(other, ctx=self.ctx)
            node = self.ctx._add_node(Op.DIV, parents=(self.node_id, o.node_id))
            return Uncertain(ctx=self.ctx, _node=node)
        
        a, b = Uncertain._align_contexts(self, other)
        node = a.ctx._add_node(Op.DIV, parents=(a.node_id, b.node_id))
        return Uncertain(ctx=a.ctx, _node=node)
    
    def __rtruediv__(self, other: Union["Uncertain", Number]) -> "Uncertain":
        if not isinstance(other, Uncertain):
            o = Uncertain._coerce(other, ctx=self.ctx)
            node = self.ctx._add_node(Op.DIV, parents=(o.node_id, self.node_id))
            return Uncertain(ctx=self.ctx, _node=node)

        a, b = Uncertain._align_contexts(other, self)
        node = a.ctx._add_node(Op.DIV, parents=(a.node_id, b.node_id))
        return Uncertain(ctx=a.ctx, _node=node)
    
    def __neg__(self) -> "Uncertain":
        node = self.ctx._add_node(Op.NEG, parents=(self.node_id,))
        return Uncertain(ctx=self.ctx, _node=node)

    def __pow__(self, power: Union["Uncertain", Number]) -> "Uncertain":
        if not isinstance(power, Uncertain):
            p = Uncertain._coerce(power, ctx=self.ctx)
            node = self.ctx._add_node(Op.POW, parents=(self.node_id, p.node_id))
            return Uncertain(ctx=self.ctx, _node=node)
        
        a, b = Uncertain._align_contexts(self, power)
        node = a.ctx._add_node(Op.POW, parents=(a.node_id, b.node_id))
        return Uncertain(ctx=a.ctx, _node=node)

    def __rpow__(self, other: Union["Uncertain", Number]) -> "Uncertain":
        if not isinstance(other, Uncertain):
            o = Uncertain._coerce(other, ctx=self.ctx)
            node = self.ctx._add_node(Op.POW, parents=(o.node_id, self.node_id))
            return Uncertain(ctx=self.ctx, _node=node)
        
        a, b = Uncertain._align_contexts(other, self)
        node = a.ctx._add_node(Op.POW, parents=(a.node_id, b.node_id))
        return Uncertain(ctx=a.ctx, _node=node)
    
    #custom numerical operations
    def log(self, base: float | None = None) -> "Uncertain":
        payload = None if base is None else float(base)
        node = self.ctx._add_node(Op.LOG, parents=(self.node_id,), payload=payload)
        return Uncertain(ctx=self.ctx, _node=node)
    
    def sin(self) -> "Uncertain":
        node = self.ctx._add_node(Op.SIN, parents=(self.node_id,))
        return Uncertain(ctx=self.ctx, _node=node)

    def cos(self) -> "Uncertain":
        node = self.ctx._add_node(Op.COS, parents=(self.node_id,))
        return Uncertain(ctx=self.ctx, _node=node)

    def tan(self) -> "Uncertain":
        node = self.ctx._add_node(Op.TAN, parents=(self.node_id,))
        return Uncertain(ctx=self.ctx, _node=node)
    
    def asin(self) -> "Uncertain":
        node = self.ctx._add_node(Op.ASIN, parents=(self.node_id,))
        return Uncertain(ctx=self.ctx, _node=node)
    
    def acos(self) -> "Uncertain":
        node = self.ctx._add_node(Op.ACOS, parents=(self.node_id,))
        return Uncertain(ctx=self.ctx, _node=node)

    def atan(self) -> "Uncertain":
        node = self.ctx._add_node(Op.ATAN, parents=(self.node_id,))
        return Uncertain(ctx=self.ctx, _node=node)
    

def eval_op(node, a, b=None):
    if node.op == Op.ADD:
        result = a + b
    elif node.op == Op.SUB:
        result = a - b
    elif node.op == Op.MUL:
        result = a * b
    elif node.op == Op.DIV:
        result = a / b
    elif node.op == Op.NEG:
        result = -a
    elif node.op == Op.POW:
        result = a ** b
    elif node.op == Op.LOG:
        x = a
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
        result = np.sin(a)
    elif node.op == Op.COS:
        result = np.cos(a)
    elif node.op == Op.TAN:
        result = np.tan(a)
    elif node.op == Op.ASIN:
        result = np.arcsin(a)
    elif node.op == Op.ACOS:
        result = np.arccos(a)
    elif node.op == Op.ATAN:
        result = np.arctan(a)
    else:
        raise ValueError(f"Unsupported op {node.op}")
    return result

def sample_uncertain(u: Uncertain, session: SampleSession) -> np.ndarray:
    ctx = u.ctx
    ctx_frozen = ctx.frozen

    def _ctx_frozen_get(node_id: int) -> np.ndarray | None:
        if not ctx_frozen:
            return None
        if ctx.frozen_n is None:
            raise RuntimeError("Context is marked frozen but frozen_n is None.")
        if session.n != ctx.frozen_n:
            raise ValueError(
                f"Frozen sample length mismatch for Context. "
                f"Context frozen_n={ctx.frozen_n}, requested n={session.n}. "
                f"Call ctx.unfreeze() or ctx.freeze(n) with the new n."
            )
        return ctx._frozen_samples.get(node_id)

    def _ctx_frozen_put(node_id: int, samples: np.ndarray) -> None:
        samples.setflags(write=False)
        ctx._frozen_samples[node_id] = samples
    
    #recursive eval function
    def eval_node(node_id: int) -> np.ndarray:
        if node_id in session.cache:
            return session.cache[node_id]
        
        frozen_hit = _ctx_frozen_get(node_id)
        if frozen_hit is not None:
            session.cache[node_id] = frozen_hit
            return frozen_hit

        node = ctx.get(node_id)

        if node.op == Op.LEAF:                
            if node.payload is None:
                raise RuntimeError("LEAF node has no payload.")
            
            if not session.correlation_prepared:
                session.prepare_correlation(ctx)

            group_id = session.leaf_to_group.get(node_id)

            if group_id is not None:
                if ctx_frozen and group_id in ctx._frozen_groups_done:
                    frozen_hit = _ctx_frozen_get(node_id)

                    if frozen_hit is None:
                        raise RuntimeError("Frozen correlated group marked done, but node not found in frozen cache.")
                    
                    session.cache[node_id] = frozen_hit
                    return frozen_hit
        
                session.group_samplers[group_id](session)

                if ctx_frozen:
                    for leaf_id, gid in session.leaf_to_group.items():
                        if gid == group_id and leaf_id in session.cache:
                            _ctx_frozen_put(leaf_id, session.cache[leaf_id])
                    ctx._frozen_groups_done.add(group_id)
                     
                return session.cache[node_id]

            result = node.payload.sample(session.n, sampler=session.sampler)

            if ctx_frozen:
                _ctx_frozen_put(node_id, result)

        elif node.op == Op.CONST:
            result = np.full(session.n, node.payload)

            if ctx_frozen:
                _ctx_frozen_put(node_id, result)

        else:
            parents = [eval_node(pid) for pid in node.parents]
            if len(parents) == 1:
                result = eval_op(node, parents[0])
            elif len(parents) == 2:
                result = eval_op(node, parents[0], parents[1])

        session.cache[node_id] = result
        return result

    return eval_node(u.node_id)


def draw_uncertain(u: Uncertain, session: SampleSession, draw_idx: int | None = None) -> np.ndarray:
    ctx = u.ctx
    ctx_frozen = ctx.frozen

    if ctx_frozen:
        if ctx.frozen_n is None:
            raise RuntimeError("Context is marked frozen but frozen_n is None.")
        if draw_idx is None:
            raise ValueError("draw_idx must be provided when Context is frozen.")
        if not (0 <= draw_idx < ctx.frozen_n):
            raise IndexError(f"draw_idx={draw_idx} out of range for frozen_n={ctx.frozen_n}.")

    def _ctx_frozen_draw(node_id: int) -> np.ndarray | None:
        if not ctx_frozen:
            return None
        arr = ctx._frozen_samples.get(node_id)
        if arr is None:
            return None
        return np.asarray(arr[draw_idx])

    def eval_node(node_id: int) -> np.ndarray:
        if node_id in session.cache:
            return session.cache[node_id]

        frozen_hit = _ctx_frozen_draw(node_id)
        if frozen_hit is not None:
            session.cache[node_id] = frozen_hit
            return frozen_hit

        node = ctx.get(node_id)

        if node.op == Op.LEAF:
            if node.payload is None:
                raise RuntimeError("LEAF node has no payload.")

            if not session.correlation_prepared:
                session.prepare_correlation(ctx)

            group_id = session.leaf_to_group.get(node_id)

            if group_id is not None:
                session.group_samplers[group_id](session)

                if node_id not in session.cache:
                    raise RuntimeError(
                        "Correlated group sampler did not populate this leaf in session cache."
                    )

                result = np.asarray(session.cache[node_id])
                session.cache[node_id] = result
                return result

            arr = node.payload.sample(session.n, sampler=session.sampler)
            result = np.asarray(arr[0])
            session.cache[node_id] = result
            return result

        elif node.op == Op.CONST:
            result = np.asarray(node.payload)
            session.cache[node_id] = result
            return result

        else:
            parents = [eval_node(pid) for pid in node.parents]
            if len(parents) == 1:
                result = np.asarray(eval_op(node, parents[0]))
            elif len(parents) == 2:
                result = np.asarray(eval_op(node, parents[0], parents[1]))
            else:
                raise RuntimeError(f"Unsupported arity: {len(parents)} parents for node {node_id}")

            session.cache[node_id] = result
            return result

    return eval_node(u.node_id)
