from __future__ import annotations
import substratum as ss
import math
from typing import Optional, Union, TYPE_CHECKING
import warnings

from .node import Node, Op
from .context import Context
from .sampleable import Sampleable, Distribution
from .sampler import Sampler
if TYPE_CHECKING:
    from .sample_session import SampleSession

Number = Union[int, float]


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
        self._frozen_samples: ss.Array | None = None

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
    def sample(self, n: int, *, sampler: Optional[Sampler] = None) -> ss.Array:
        if self._frozen_samples is not None:
            if n != len(self._frozen_samples):
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
        return s.mean()

    def var(self, n: int = 20_000, sampler: Optional[Sampler] = None):
        if self.frozen and self.frozen_n:
            n = self.frozen_n
        if self.ctx.frozen and self.ctx.frozen_n:
            n = self.ctx.frozen_n

        s = self.sample(n, sampler=sampler)
        return s.var()

    def quantile(self, q: float, n: int = 50_000, *, sampler: Optional[Sampler] = None) -> float:
        if self.frozen and self.frozen_n:
            n = self.frozen_n
        if self.ctx.frozen and self.ctx.frozen_n:
            n = self.ctx.frozen_n

        if q < 0.0 or q > 1.0:
            raise ValueError("q must be between 0 and 1")

        s = self.sample(n, sampler=sampler)
        return s.quantile(q)

    def cdf(self, x: float, n: int = 200_000, *, sampler: Optional[Sampler] = None) -> float:
        if self.frozen and self.frozen_n:
            n = self.frozen_n
        if self.ctx.frozen and self.ctx.frozen_n:
            n = self.ctx.frozen_n

        s = self.sample(n, sampler=sampler)
        count = sum(1 for val in s if val <= x)
        return count / len(s)

    def draw(self, *, sampler: Optional[Sampler] = None) -> float:
        """
        Draw a random value from an uncertain distribution. If the context is frozen,
        cycle through the values in the current frozen cache, else random values are
        drawn. The same draw value will be returned, until redraw is called on the context,
        this uncertain object, or any other uncertain object within the same context.

        calling `float()` on an uncertain object is the same as calling `draw()`. This
        can be used to artificially run monte carlo simulations on external functions
        that aren't supported by dubious. By repeatedly passing in the uncertain object, and
        calling redraw you will get random results from the distribution. Alternatively,
        if your function supports vectorized inputs, call `sample()` and pass in the result.

        :param sampler: Dubious Sampler object.
        :type sampler: Sampler
        :return: Randomly drawn float.
        :rtype: float
        """
        if sampler is None:
            sampler = Sampler()

        from .sample_session import SampleSession
        session = SampleSession(n=1, sampler=sampler)
        idx = None
        if self.frozen:
            assert self.frozen_n is not None
            idx = self.ctx.draw_idx % self.frozen_n
            if self._frozen_samples is None or len(self._frozen_samples) == 0:
                self.sample(self.frozen_n)
        if self.ctx.frozen:
            assert self.ctx.frozen_n is not None
            idx = self.ctx.draw_idx % self.ctx.frozen_n
            if self.ctx._frozen_samples is None or len(self.ctx._frozen_samples) == 0:
                self.sample(self.ctx.frozen_n)

        return float(draw_uncertain(self, session=session, draw_idx=idx))

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
        """
        if self.frozen and self.frozen_n == n: #if they call freeze with same n just return
            return
        samples = self.sample(n, sampler=sampler)
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


    def _collect_leaf_ids(self) -> set[int]:
        leaves = set()
        visited = set()
        stack = [self.node_id]
        
        while stack:
            nid = stack.pop()
            if nid in visited:
                continue
            visited.add(nid)
            
            node = self.ctx.get(nid)
            if node.op == Op.LEAF:
                leaves.add(nid)
            else:
                stack.extend(node.parents)
        
        return leaves
    
    def sensitivity(self, n: int = 20_000, method: str = "pearson", *, sampler: Optional[Sampler] = None) -> dict[int, float]:
        """
        Compute correlation-based sensitivity of this output to each input leaf.
        
        :param n: Number of samples.
        :param method: "pearson" or "spearman"
        :param sampler: Dubious Sampler object.
        :return: Dict mapping leaf node_id to correlation with output.
        """
        if method not in ("pearson", "spearman"):
            raise ValueError(f"method must be 'pearson' or 'spearman', got {method!r}")
        
        leaf_ids = self._collect_leaf_ids()
        if not leaf_ids:
            return {}
        
        from .sample_session import SampleSession
        if sampler is None:
            sampler = Sampler()
        session = SampleSession(n, sampler=sampler)
        
        output_samples = sample_uncertain(self, session)
        
        result = {}
        for lid in leaf_ids:
            leaf_samples = session.cache[lid]
            
            if method == "pearson":
                corr = output_samples.pearson(leaf_samples)
            else:
                corr = output_samples.spearman(leaf_samples)
            
            result[lid] = corr
        
        return result


def eval_op(node: Node, a: ss.Array, b: Optional[ss.Array] = None) -> ss.Array:
    if node.op == Op.ADD:
        result = a + b # type: ignore
    elif node.op == Op.SUB:
        result = a - b # type: ignore
    elif node.op == Op.MUL:
        result = a * b # type: ignore
    elif node.op == Op.DIV:
        result = a / b # type: ignore
    elif node.op == Op.NEG:
        result = -a
    elif node.op == Op.POW:
        result = a ** b # type: ignore
    elif node.op == Op.LOG:
        x = a
        # Check if any values are <= 0
        if any(val <= 0 for val in x):
            warnings.warn("Warning: Log domain <= 0 found, clamped to 1e-6.")
            x = x.clip(1e-6, 1e308)

        if node.payload is None:
            result = x.log()
        else:
            base = float(node.payload)
            if base <= 0 or base == 1.0:
                raise ValueError("log() base must be > 0 and != 1.")
            result = x.log() / math.log(base)
    elif node.op == Op.SIN:
        result = a.sin()
    elif node.op == Op.COS:
        result = a.cos()
    elif node.op == Op.TAN:
        result = a.tan()
    elif node.op == Op.ASIN:
        result = a.arcsin()
    elif node.op == Op.ACOS:
        result = a.arccos()
    elif node.op == Op.ATAN:
        result = a.arctan()
    else:
        raise ValueError(f"Unsupported op {node.op}")
    return result


def sample_uncertain(u: Uncertain, session: "SampleSession") -> ss.Array:
    ctx = u.ctx
    ctx_frozen = ctx.frozen

    def _ctx_frozen_get(node_id: int) -> ss.Array | None:
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

    def _ctx_frozen_put(node_id: int, samples: ss.Array) -> None:
        ctx._frozen_samples[node_id] = samples

    #recursive eval function
    def eval_node(node_id: int) -> ss.Array:
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
            result = ss.full([session.n], float(node.payload)) # type: ignore

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


def draw_uncertain(u: Uncertain, session: "SampleSession", draw_idx: int | None = None) -> float:
    ctx = u.ctx
    ctx_frozen = ctx.frozen

    if ctx_frozen:
        if ctx.frozen_n is None:
            raise RuntimeError("Context is marked frozen but frozen_n is None.")
        if draw_idx is None:
            raise ValueError("draw_idx must be provided when Context is frozen.")
        if not (0 <= draw_idx < ctx.frozen_n):
            raise IndexError(f"draw_idx={draw_idx} out of range for frozen_n={ctx.frozen_n}.")

    def _ctx_frozen_draw(node_id: int) -> float | None:
        if not ctx_frozen:
            return None
        arr = ctx._frozen_samples.get(node_id)
        if arr is None:
            return None
        return float(arr[draw_idx]) # type: ignore

    def eval_node(node_id: int) -> float:
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

                result = session.cache[node_id]
                return result

            arr = node.payload.sample(session.n, sampler=session.sampler)
            result = float(arr[0])
            session.cache[node_id] = result
            return result

        elif node.op == Op.CONST:
            result = float(node.payload) # type: ignore
            session.cache[node_id] = result
            return result

        else:
            parents = [eval_node(pid) for pid in node.parents]
            if len(parents) == 1:
                # For single value operations, create a 1-element array
                a_arr = ss.asarray([parents[0]])
                result_arr = eval_op(node, a_arr)
                result = float(result_arr[0]) # type: ignore
            elif len(parents) == 2:
                a_arr = ss.asarray([parents[0]])
                b_arr = ss.asarray([parents[1]])
                result_arr = eval_op(node, a_arr, b_arr)
                result = float(result_arr[0]) # type: ignore
            else:
                raise RuntimeError(f"Unsupported arity: {len(parents)} parents for node {node_id}")

            session.cache[node_id] = result
            return result

    return eval_node(u.node_id)
