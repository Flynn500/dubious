from __future__ import annotations
import itertools
from typing import Dict
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