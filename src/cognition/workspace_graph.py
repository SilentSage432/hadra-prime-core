# prime-core/cognition/workspace_graph.py

"""
A175 — Temporal Workspace Graph Formation
------------------------------------------
Creates a directed weighted graph of conscious workspace states.
Each node stores tokenized thought segments.
Edges represent temporal progression and conceptual similarity.
"""

import uuid
from collections import defaultdict, deque

try:
    import torch
except ImportError:
    torch = None

from ..neural.torch_utils import safe_tensor, safe_cosine_similarity, TORCH_AVAILABLE


class WorkspaceGraph:
    """
    A175 — Temporal Workspace Graph Formation
    Creates a directed weighted graph of conscious workspace states.
    Each node stores tokenized thought segments.
    Edges represent temporal progression and conceptual similarity.
    """

    def __init__(self, max_nodes=60):
        self.max_nodes = max_nodes
        self.nodes = {}        # node_id → {tokens, raw_vec, metadata}
        self.edges = defaultdict(list)   # node_id → list of (target_node_id, weight)
        self.order = deque()   # maintain temporal order
        self.last_node = None

    def add_node(self, tokens, raw_vec, metadata=None):
        """
        Add a new node to the graph with tokens and raw vector.
        Automatically connects to previous node if available.
        """
        if tokens is None:
            return None

        node_id = str(uuid.uuid4())
        self.nodes[node_id] = {
            "tokens": tokens,
            "vector": raw_vec,
            "meta": metadata or {}
        }

        self.order.append(node_id)

        # prune old nodes
        while len(self.order) > self.max_nodes:
            old = self.order.popleft()
            self.nodes.pop(old, None)
            self.edges.pop(old, None)
            # Also remove edges pointing to this node
            for source_id in list(self.edges.keys()):
                self.edges[source_id] = [
                    (target_id, weight) for target_id, weight in self.edges[source_id]
                    if target_id != old
                ]

        # connect to previous node
        if self.last_node is not None:
            try:
                weight = self.compute_similarity(raw_vec, self.nodes[self.last_node]["vector"])
                self.edges[self.last_node].append((node_id, float(weight)))
            except Exception:
                # If similarity computation fails, still add edge with default weight
                self.edges[self.last_node].append((node_id, 0.5))

        self.last_node = node_id
        return node_id

    def compute_similarity(self, a, b):
        """Cosine similarity for edge weighting."""
        if a is None or b is None:
            return 0.0
        
        try:
            a_tensor = safe_tensor(a)
            b_tensor = safe_tensor(b)
            if a_tensor is not None and b_tensor is not None:
                similarity = safe_cosine_similarity(a_tensor, b_tensor)
                return float(similarity) if similarity is not None else 0.0
        except Exception:
            pass
        
        return 0.0

    def get_recent_subgraph(self, n=10):
        """Return last N nodes & edges."""
        recent = list(self.order)[-n:]
        sub_nodes = {nid: self.nodes[nid] for nid in recent if nid in self.nodes}
        sub_edges = {nid: self.edges[nid] for nid in recent if nid in self.edges}
        return sub_nodes, sub_edges

    def summary(self):
        """Human-readable summary for debug."""
        edge_count = sum(len(v) for v in self.edges.values())
        return {
            "node_count": len(self.nodes),
            "edge_count": edge_count,
            "latest_node": self.last_node,
        }

