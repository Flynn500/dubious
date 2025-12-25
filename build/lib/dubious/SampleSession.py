import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Callable, Optional, List, Tuple, Set

from.context import Context

@dataclass
class SampleSession:
    n: int
    rng: np.random.Generator
    cache: Dict[int, np.ndarray] = field(default_factory=dict)

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
        C = np.eye(k, dtype=float)
        for i in range(k):
            for j in range(i + 1, k):
                rho = ctx.get_corr(leaf_ids[i], leaf_ids[j])
                C[i, j] = C[j, i] = rho

        #coerce user input to valid matrix
        w, V = np.linalg.eigh(C)
        w_clipped = np.clip(w, 1e-12, None)
        C_psd = (V * w_clipped) @ V.T

        d = np.sqrt(np.diag(C_psd))
        C_psd = C_psd / np.outer(d, d)


        #try cholesky, adding jitter if failing
        jitter = 0.0
        cholesky_succeded = False
        for _ in range(5):
            try:
                L = np.linalg.cholesky(C_psd + jitter * np.eye(C_psd.shape[0]))
                cholesky_succeded = True
                break
            except np.linalg.LinAlgError:
                jitter = 1e-12 if jitter == 0.0 else jitter * 10
        if not cholesky_succeded:
            w, V = np.linalg.eigh(C_psd)
            w = np.clip(w, 0.0, None)
            L = V @ np.diag(np.sqrt(w))

        def sampler(session: "SampleSession"):
            from .umath import erf #avoid circular import, maybe I should move this function elsewhere
            if leaf_ids[0] in session.cache:
                return

            eps = session.rng.standard_normal(size=(k, session.n))
            Z = L @ eps

            inv_sqrt2 = 1.0 / np.sqrt(2.0)
            U = 0.5 * (1.0 + erf(Z * inv_sqrt2))

            for i, leaf_id in enumerate(leaf_ids):
                node = ctx.get(leaf_id)
                dist = node.payload
                #clamp to avoid inf
                Ui = np.clip(U[i], 1e-15, 1 - 1e-15)
                if dist is not None:
                    Xi = dist.quantile(Ui)
                else: 
                    raise ValueError("Leaf node has no distribution.")
                session.cache[leaf_id] = Xi
        return sampler
