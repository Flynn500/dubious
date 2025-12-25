from __future__ import annotations
import itertools
from typing import Dict, Optional, TYPE_CHECKING, Union, Any
from .node import Node, Op

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
    
    @property
    def nodes(self) -> Dict[int, Node]:
        return self._nodes

    def add_node(self, op: Op, parents: tuple[int, ...] = (), payload=None) -> Node:
        node = Node(id=next(self._ids), op=op, parents=parents, payload=payload)
        self._nodes[node.id] = node
        return node

    def get(self, node_id: int) -> Node:
        return self._nodes[node_id]

    def copy_subgraph_from(self, src: "Context", root_id: int, *, memo: Optional[Dict[int, int]] = None,) -> int:
        if memo is None:
            memo = {}

        if root_id in memo:
            return memo[root_id]

        src_node = src.get(root_id)
        new_parents = tuple(self.copy_subgraph_from(src, pid, memo=memo) for pid in src_node.parents)
        new_node = self.add_node(src_node.op, parents=new_parents, payload=src_node.payload)

        memo[root_id] = new_node.id
        return new_node.id

    def set_corr(self, a: Union[int, Any], b: Union[int, Any], rho: float):
        """
        Sets correlation between to uncertain leaf nodes using gaussian copular.
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
