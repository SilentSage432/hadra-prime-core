# prime-core/cognition/workspace_priority_engine.py

# ============================================
# A178 — Cognitive Workspace Prioritization Hierarchy
# ============================================
# This module implements hierarchical prioritization for cognitive workspace items.
# PRIME now ranks cognitive signals by importance, giving identity, drift, and tasks
# precedence over novelty and exploration.

try:
    import torch
except ImportError:
    torch = None

from ..neural.torch_utils import safe_tensor, safe_cosine_similarity, TORCH_AVAILABLE


class WorkspacePriorityEngine:
    """
    A178 — Cognitive Workspace Prioritization Hierarchy

    Computes priority scores for:
    - identity anchors
    - drift correction signals
    - tasks
    - memory recalls
    - reflections
    - novelty-driven vectors

    Produces a sorted list of workspace elements ordered by cognitive importance.
    """

    def __init__(self):
        # Tunable weight schema
        self.weights = {
            "identity": 4.0,
            "drift": 3.5,
            "task": 3.0,
            "reflection": 2.0,
            "memory": 1.8,
            "novelty": 1.5,
            "exploration": 1.0,
        }

    def compute_priority(self, item_type, vector, state, tasks):
        """
        Compute a numerical priority score.

        Args:
            item_type: Type of cognitive item (identity, drift, task, etc.)
            vector: Tensor or list representing the cognitive vector
            state: Cognitive state object
            tasks: Task queue object with peek() method

        Returns:
            float: Priority score (higher = more important)
        """
        w = self.weights.get(item_type, 1.0)

        score = w

        # If identity-related, boost based on similarity to identity vector
        if item_type == "identity" and hasattr(state, 'timescales') and state.timescales is not None:
            identity_vec = getattr(state.timescales, 'identity_vector', None)
            if identity_vec is not None:
                try:
                    vec_tensor = safe_tensor(vector)
                    id_tensor = safe_tensor(identity_vec)
                    if vec_tensor is not None and id_tensor is not None:
                        similarity = safe_cosine_similarity(vec_tensor, id_tensor)
                        if similarity is not None:
                            score *= max(0.1, float(similarity))
                except Exception:
                    pass

        # Task priority multiplier
        if item_type == "task" and tasks is not None:
            try:
                if hasattr(tasks, 'peek'):
                    t = tasks.peek()
                    if t and isinstance(t, dict) and "priority" in t:
                        score *= max(1.0, t["priority"] / 5.0)
            except Exception:
                pass

        # Drift multiplier
        if item_type == "drift" and hasattr(state, 'drift'):
            try:
                drift_obj = state.drift
                if hasattr(drift_obj, 'avg_drift'):
                    d = drift_obj.avg_drift
                    if d is not None:
                        score *= (1.0 + min(float(d) * 10, 2.0))
            except Exception:
                pass

        # Novelty multiplier
        if item_type == "novelty":
            try:
                vec_tensor = safe_tensor(vector)
                if vec_tensor is not None:
                    if TORCH_AVAILABLE and isinstance(vec_tensor, torch.Tensor):
                        novelty = torch.norm(vec_tensor).item()
                    else:
                        # Fallback for list/array
                        import math
                        novelty = math.sqrt(sum(x * x for x in vec_tensor))
                    score *= max(1.0, novelty / 2.0)
            except Exception:
                pass

        return score

    def rank(self, workspace_items, state, tasks):
        """
        Rank workspace items by priority.

        Args:
            workspace_items: List of dicts with keys:
                - "type": str (identity, drift, task, reflection, memory, novelty, exploration)
                - "vector": tensor or list
                - "raw": original item data
            state: Cognitive state object
            tasks: Task queue object

        Returns:
            List of tuples: [(score, item_dict), ...] sorted highest to lowest
        """
        scored = []

        for item in workspace_items:
            try:
                item_type = item.get("type", "exploration")
                vector = item.get("vector")
                
                if vector is None:
                    continue
                
                score = self.compute_priority(
                    item_type,
                    vector,
                    state,
                    tasks
                )
                scored.append((score, item))
            except Exception:
                # Skip items that fail to score
                continue

        # Sort highest → lowest
        scored.sort(key=lambda x: x[0], reverse=True)

        return scored

