import substratum as ss
import math
from dataclasses import dataclass, field
from typing import Dict, Any, Callable, List, Set

from .sampler import Sampler
from .context import Context
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

        # Build correlation matrix
        C = ss.eye(k)
        C_data = C.tolist()
        for i in range(k):
            for j in range(i + 1, k):
                rho = ctx.get_corr(leaf_ids[i], leaf_ids[j])
                # Set symmetric entries
                C_data[i * k + j] = rho
                C_data[j * k + i] = rho
        C = ss.asarray(C_data, [k, k])

        # Coerce user input to valid matrix via eigendecomposition
        w, V = C.eig()
        # Clip eigenvalues to ensure positive semi-definite
        w_list = [max(val, 1e-12) for val in w]
        w_clipped = ss.asarray(w_list)

        # Reconstruct C_psd = V @ diag(w_clipped) @ V.T
        C_psd = V @ ss.diag(w_clipped.tolist()) @ V.t()

        # Normalize to correlation matrix
        d_list = [math.sqrt(C_psd.get([i, i])) for i in range(k)]
        d = ss.asarray(d_list)
        d_outer = ss.outer(d.tolist(), d.tolist())
        C_psd = C_psd / d_outer

        # Try cholesky, adding jitter if failing
        jitter = 0.0
        L = None
        for _ in range(5):
            try:
                jitter_mat = ss.eye(k) * jitter if jitter > 0 else ss.zeros([k, k])
                L = (C_psd + jitter_mat).cholesky()
                break
            except ValueError:
                jitter = 1e-12 if jitter == 0.0 else jitter * 10

        if L is None:
            # Fallback: use eigendecomposition
            w2, V2 = C_psd.eig()
            w2_list = [max(val, 0.0) for val in w2]
            w2_sqrt = ss.asarray([math.sqrt(v) for v in w2_list])
            L = V2 @ ss.diag(w2_sqrt.tolist())

        def sampler(session: "SampleSession"):
            if leaf_ids[0] in session.cache:
                return

            eps = session.sampler.standard_normal([k, session.n])
            Z = L @ eps

            inv_sqrt2 = 1.0 / math.sqrt(2.0)
            U = (erf(Z * inv_sqrt2) + 1.0) * 0.5

            for i, leaf_id in enumerate(leaf_ids):
                node = ctx.get(leaf_id)
                dist = node.payload
                # Get row i from U and clamp to avoid inf
                Ui_row = U[i]
                # Ui_row is an Array (row of the 2D matrix)
                if isinstance(Ui_row, ss.Array):
                    Ui_clamped = Ui_row.clip(1e-15, 1 - 1e-15)
                else:
                    # Single value case
                    Ui_clamped = ss.asarray([max(1e-15, min(1 - 1e-15, float(Ui_row)))])
                if dist is not None:
                    # Apply inverse CDF (quantile) element-wise
                    Xi_list = [dist.quantile(float(u)) for u in Ui_clamped]
                    Xi = ss.asarray(Xi_list)
                else:
                    raise ValueError("Leaf node has no distribution.")
                session.cache[leaf_id] = Xi

        return sampler
