# prime-core/cognition/workspace_loop_detector.py

"""
A176 — Recurrent Loop Detection & Cognitive Re-entry
----------------------------------------------------
Identifies recurring cognitive patterns in the workspace graph and
determines if PRIME should re-enter a previous thought loop.
"""

try:
    import torch
except ImportError:
    torch = None

from ..neural.torch_utils import safe_tensor, safe_cosine_similarity, TORCH_AVAILABLE


class WorkspaceLoopDetector:
    """
    A176 — Recurrent Loop Detection & Cognitive Re-entry
    Identifies recurring cognitive patterns in the workspace graph and
    determines if PRIME should re-enter a previous thought loop.
    """

    def __init__(self, similarity_threshold=0.92, max_history=40):
        self.threshold = similarity_threshold
        self.max_history = max_history
        self.last_detected_loop = None

    def detect_loop(self, graph):
        """
        Checks latest node against prior nodes for similarity.
        Returns: (loop_detected, target_node_id, similarity)
        """
        if graph.last_node is None:
            return (False, None, 0.0)

        latest = graph.nodes.get(graph.last_node)
        if latest is None:
            return (False, None, 0.0)

        latest_vec = latest.get("vector")
        if latest_vec is None:
            return (False, None, 0.0)

        latest_tensor = safe_tensor(latest_vec)
        if latest_tensor is None:
            return (False, None, 0.0)

        checked = 0
        best_match = (None, 0.0)

        # compare to recent history (reversed to check most recent first)
        for nid in reversed(graph.order):
            if nid == graph.last_node:
                continue

            prev = graph.nodes.get(nid)
            if prev is None:
                continue

            vec = prev.get("vector")
            if vec is None:
                continue

            prev_tensor = safe_tensor(vec)
            if prev_tensor is None:
                continue

            try:
                sim = safe_cosine_similarity(latest_tensor, prev_tensor)
                if sim is not None:
                    sim_score = float(sim)
                    if sim_score > best_match[1]:
                        best_match = (nid, sim_score)
                else:
                    sim_score = 0.0
            except Exception:
                sim_score = 0.0

            checked += 1
            if checked >= self.max_history:
                break

        # check similarity threshold
        target_id, sim_score = best_match
        if target_id and sim_score >= self.threshold:
            self.last_detected_loop = {
                "target": target_id,
                "similarity": sim_score
            }
            return (True, target_id, sim_score)

        return (False, None, sim_score)

    def cognitive_reentry(self, graph, target_id):
        """
        Pulls the previous node into the active conscious workspace,
        guiding PRIME to continue a previous reasoning chain.
        """
        node = graph.nodes.get(target_id)
        if not node:
            return None

        tokens = node.get("tokens")
        vec = node.get("vector")

        return {
            "reentered_tokens": tokens,
            "reentered_vector": vec,
            "source_node": target_id,
        }

