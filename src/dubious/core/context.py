from __future__ import annotations
import itertools
from typing import Dict, Optional, Union, Any
import ironforest as irn

from .node import Node, Op
from .sampler import Sampler


class Context:
    """
    Context objects own the graph that stores the operations applied to uncertain distributions.
    Creating a context is not required to add Uncertain objects together, but it is preffered as
    merging seperate contexts can be expensive.
    """
    def __init__(self):
        self._ids = itertools.count(1)
        self._nodes: Dict[int, Node] = {}
        self._corr: Dict[tuple[int, int], float] = {}

        self._frozen = False
        self._frozen_n = None
        self._frozen_samples: dict[int, irn.Array] = {}
        self._frozen_groups_done: set[int] = set()
        self.draw_idx = 0
    
    @property
    def nodes(self) -> Dict[int, Node]:
        return self._nodes
    
    @property
    def frozen(self) -> bool:
        return self._frozen

    @property
    def frozen_n(self) -> int | None:
        return self._frozen_n

    def _add_node(self, op: Op, parents: tuple[int, ...] = (), payload=None) -> Node:
        node = Node(id=next(self._ids), op=op, parents=parents, payload=payload)
        self._nodes[node.id] = node
        return node

    def get(self, node_id: int) -> Node:
        return self._nodes[node_id]
    
    def redraw(self):
        self.draw_idx += 1
    
    def freeze(self, n, *, sampler: Optional[Sampler] = None):
        """
        Freeze the context. Sample once and cache the result for all future
        operations until unfreeze() or freeze() is called with a different value of n.

        :param n: Number of samples.
        :type n: int
        :param sampler: Dubious Sampler object.
        :type sampler: Sampler
        """
        if self.frozen and self.frozen_n == n:
            return

        if self._frozen and self._frozen_n != n:
            self._frozen_samples.clear()

        if sampler is None:
            sampler = Sampler()

        self._frozen = True
        self._frozen_n = n

        for node_id, node in self._nodes.items():
            if node.op == Op.LEAF:
                if node.payload is None:
                    raise RuntimeError("LEAF node has no payload.")
                samples = node.payload.sample(n, sampler=sampler)
                self._frozen_samples[node_id] = samples
            elif node.op == Op.CONST:
                if node.payload is None:
                    raise RuntimeError("CONST node has no payload.")
                samples = irn.full([n], float(node.payload))
                self._frozen_samples[node_id] = samples

    def unfreeze(self):
        """
        Unfreeze the context, clearing the cache of all uncertain objects.
        """
        self._frozen_samples.clear()
        self._frozen = False
        self._frozen_n = None

    def _copy_subgraph_from(self, src: "Context", root_id: int, *, memo: Optional[Dict[int, int]] = None,) -> int:
        if memo is None:
            memo = {}

        if root_id in memo:
            return memo[root_id]

        src_node = src.get(root_id)
        new_parents = tuple(self._copy_subgraph_from(src, pid, memo=memo) for pid in src_node.parents)
        new_node = self._add_node(src_node.op, parents=new_parents, payload=src_node.payload)

        memo[root_id] = new_node.id
        return new_node.id

    def _copy_corr_from(self, src: "Context", memo: dict[int, int]) -> None:
        """
        Copy correlation entries for any node-pairs that were copied into this context.
        """
        for (i, j), rho in src._corr.items():
            ni = memo.get(i)
            nj = memo.get(j)
            if ni is None or nj is None:
                continue
            key = (ni, nj) if ni < nj else (nj, ni)
            self._corr[key] = rho

    def _copy_frozen_samples_from(self, src: "Context", memo: dict[int, int]) -> None:
        """
        Copy frozen samples for copied nodes. Assumes n matches.
        """
        for old_id, new_id in memo.items():
            samples = src._frozen_samples.get(old_id)
            if samples is None:
                continue

            self._frozen_samples[new_id] = samples
        
        for old_id in src._frozen_groups_done:
            new_id = memo.get(old_id)
            if new_id is not None:
                self._frozen_groups_done.add(new_id)


    def set_corr(self, a: Union[int, Any], b: Union[int, Any], rho: float):
        """
        Sets correlation between to uncertain leaf nodes using gaussian copular.

        Dubious uses Gaussian copula (rank/latent-normal dependence). rho is not Pearson 
        correlation and the realized linear correlation can differ. Validate by 
        sampling if a specific dependence measure matters.
        """
        a = getattr(a, "node_id", a)
        b = getattr(b, "node_id", b)

        if a == b:
            return
        key = (a, b) if a < b else (b, a)
        self._corr[key] = float(rho)

    def get_corr(self, a: Union[int, Any], b: Union[int, Any]) -> float:
        a = getattr(a, "node_id", a)
        b = getattr(b, "node_id", b)
        
        if a == b:
            return 1.0
        key = (a, b) if a < b else (b, a)
        return self._corr.get(key, 0.0)
