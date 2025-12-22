from __future__ import annotations
import itertools
from typing import Dict, Optional
from .node import Node, Op

class Context:
    def __init__(self):
        self._ids = itertools.count(1)
        self._nodes: Dict[int, Node] = {}

    def add_node(self, op: Op, parents: tuple[int, ...] = (), payload=None) -> Node:
        node = Node(id=next(self._ids), op=op, parents=parents, payload=payload)
        self._nodes[node.id] = node
        return node

    def get(self, node_id: int) -> Node:
        return self._nodes[node_id]

    @property
    def nodes(self) -> Dict[int, Node]:
        return self._nodes

    def copy_subgraph_from(self, src: "Context", root_id: int, *, memo: Optional[Dict[int, int]] = None,
    ) -> int:
        if memo is None:
            memo = {}

        if root_id in memo:
            return memo[root_id]

        src_node = src.get(root_id)

        # Copy parents first so we can rewrite parent links.
        new_parents = tuple(self.copy_subgraph_from(src, pid, memo=memo) for pid in src_node.parents)

        # Payload is carried over as-is (Distribution objects, floats, etc.).
        new_node = self.add_node(src_node.op, parents=new_parents, payload=src_node.payload)

        memo[root_id] = new_node.id
        return new_node.id