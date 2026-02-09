import ironforest as irn
from dataclasses import dataclass, field
from typing import Dict, Any, Callable, Optional, List, Tuple, Set
from .sampler import Sampler
import math

from.context import Context
from ..umath.stats import erf
@dataclass
class SampleSession:
    n: int
    sampler: Sampler
    cache: Dict[int, Any] = field(default_factory=dict)

    group_samplers: Dict[Any, Callable[["SampleSession"], None]] = field(default_factory=dict)
    leaf_to_group: Dict[int, Any] = field(default_factory=dict)

    correlation_prepared: bool = False

    def prepare_correlation(self, ctx: "Context"):
        if self.correlation_prepared:
            return

        involved: Set[int] = set()
        adj: Dict[int, Set[int]] = {}

        for (a, b), rho in ctx._corr.items():
            involved.add(a); involved.add(b)
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)

        visited: Set[int] = set()
        groups: List[List[int]] = []

        for start in involved:
            if start in visited:
                continue
            stack = [start]
            comp = []
            visited.add(start)
            while stack:
                u = stack.pop()
                comp.append(u)
                for v in adj.get(u, ()):
                    if v not in visited:
                        visited.add(v)
                        stack.append(v)
            if len(comp) >= 2:
                groups.append(comp)

        for gi, leaf_ids in enumerate(groups):
            group_id = ("gaussian_copula", gi)
            for lid in leaf_ids:
                self.leaf_to_group[lid] = group_id

            self.group_samplers[group_id] = self._make_gaussian_copula_group_sampler(ctx, leaf_ids)

        self.correlation_prepared = True

    def _make_gaussian_copula_group_sampler(self, ctx: "Context", leaf_ids: List[int]):
        leaf_ids = list(leaf_ids)
        k = len(leaf_ids)
    
        #build correlation matrix
        C = irn.eye(k)
        for i in range(k):
            for j in range(i + 1, k):
                rho = ctx.get_corr(leaf_ids[i], leaf_ids[j])
                C[i, j] = C[j, i] = rho

        #coerce user input to valid matrix
        w, V = irn.linalg.eig(C)
        w_clipped = w.clip(1e-12, float('inf'))

        C_psd = (V * w_clipped) @ (V.transpose())

        d = (C_psd.diagonal()).sqrt()
        C_psd = C_psd / irn.linalg.outer(d, d)

        #try cholesky, adding jitter if failing
        jitter = 0.0
        cholesky_succeded = False
        for _ in range(5):
            try:
                L = irn.linalg.cholesky(C_psd + jitter * irn.eye(C_psd.shape[0]))
                cholesky_succeded = True
                break
            except ValueError:
                jitter = 1e-12 if jitter == 0.0 else jitter * 10
        if not cholesky_succeded:
            w, V = irn.linalg.eig(C_psd)
            w = w.clip(0.0, float('inf'))
            L = V @ (w.sqrt()).diagonal()

        def sampler(session: "SampleSession"):
            if leaf_ids[0] in session.cache:
                return

            eps = session.sampler.standard_normal(size=(k, session.n))
            Z = L @ eps # type: ignore

            inv_sqrt2 = 1.0 / math.sqrt(2.0)
            U = 0.5 * (1.0 + erf(Z * inv_sqrt2))

            #clamp to avoid inf
            U.clip(1e-15, 1 - 1e-15)
            for i, leaf_id in enumerate(leaf_ids):
                node = ctx.get(leaf_id)
                dist = node.payload

                if dist is not None:
                    Xi = dist.quantile(U[i])
                    # Xi_list = [dist.quantile(float(u)) for u in U]
                    # Xi = ss.asarray(Xi_list)
                else: 
                    raise ValueError("Leaf node has no distribution.")
                session.cache[leaf_id] = Xi
        return sampler
    