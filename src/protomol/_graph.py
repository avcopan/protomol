"""The graph engine: a NetworkX graph with validated node and edge data."""

import networkx as nx
from pydantic import BaseModel


class Node(BaseModel):
    """Base class for node data. Subclass to add validated fields."""


class Edge(BaseModel):
    """Base class for edge data. Subclass to add validated fields."""


class Graph[NodeT: Node, EdgeT: Edge](nx.Graph):
    """Generic graph with typed node and edge data."""

    node_type: type[NodeT]
    edge_type: type[EdgeT]

    def validate(self) -> None:
        """Validate node and edge data against the node and edge classes."""
        for *_, data in self.nodes(data=True):
            self.node_type.model_validate(data)

        for *_, data in self.edges(data=True):
            self.edge_type.model_validate(data)
