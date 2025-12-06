# prime-core/cognition/loop_purpose_classifier.py

"""
A177 — Loop Purpose Classification & Intent-Aware Re-entry
----------------------------------------------------------
Determines WHY a loop occurred and HOW PRIME should respond.
"""

try:
    import torch
except ImportError:
    torch = None

from ..neural.torch_utils import safe_tensor, safe_cosine_similarity, TORCH_AVAILABLE


class LoopPurposeClassifier:
    """
    A177 — Loop Purpose Classification & Intent-Aware Re-entry
    Determines WHY a loop occurred and HOW PRIME should respond.
    """

    def __init__(self):
        pass

    def classify(self, reentry_vector, graph, state, tasks):
        """
        Returns a dict describing loop purpose.
        """
        scores = {}

        reentry_tensor = safe_tensor(reentry_vector)
        if reentry_tensor is None:
            return {
                "purpose": "exploratory_reentry",
                "scores": {}
            }

        # 1. Identity reinforcement
        try:
            identity_vec = None
            if hasattr(state, 'timescales') and hasattr(state.timescales, 'identity_vector'):
                identity_vec = state.timescales.identity_vector
            
            if identity_vec is not None:
                identity_tensor = safe_tensor(identity_vec)
                if identity_tensor is not None:
                    s = safe_cosine_similarity(reentry_tensor, identity_tensor)
                    scores["identity_reinforcement"] = float(s) if s is not None else 0.0
                else:
                    scores["identity_reinforcement"] = 0.0
            else:
                scores["identity_reinforcement"] = 0.0
        except Exception:
            scores["identity_reinforcement"] = 0.0

        # 2. Task continuity
        try:
            last_task = None
            if tasks and hasattr(tasks, 'peek'):
                last_task = tasks.peek()
            
            if last_task:
                # Try to get task text
                task_text = None
                if isinstance(last_task, dict):
                    task_text = last_task.get("text") or last_task.get("content")
                elif hasattr(last_task, 'text'):
                    task_text = last_task.text
                
                if task_text and hasattr(state, 'hooks') and hasattr(state.hooks, 'on_perception'):
                    task_vec = state.hooks.on_perception(task_text)
                    task_tensor = safe_tensor(task_vec)
                    if task_tensor is not None:
                        s = safe_cosine_similarity(reentry_tensor, task_tensor)
                        scores["task_continuity"] = float(s) if s is not None else 0.0
                    else:
                        scores["task_continuity"] = 0.0
                else:
                    scores["task_continuity"] = 0.0
            else:
                scores["task_continuity"] = 0.0
        except Exception:
            scores["task_continuity"] = 0.0

        # 3. Reflection or consolidation
        # Check similarity to last few nodes as a "reflection echo"
        reflection_score = 0.0
        try:
            if graph and hasattr(graph, 'order') and hasattr(graph, 'nodes'):
                recent_ids = list(graph.order)[-5:] if len(graph.order) > 0 else []
                count = 0
                for nid in recent_ids:
                    node = graph.nodes.get(nid)
                    if node and node.get("vector") is not None:
                        v = node.get("vector")
                        v_tensor = safe_tensor(v)
                        if v_tensor is not None:
                            s = safe_cosine_similarity(reentry_tensor, v_tensor)
                            if s is not None:
                                reflection_score += float(s)
                                count += 1
                
                if count > 0:
                    reflection_score /= count
        except Exception:
            pass
        
        scores["reflection_continuity"] = reflection_score

        # 4. Detect anomaly review (low coherence edges)
        try:
            coherence = 1.0  # Default
            if hasattr(state, 'workspace_coherence'):
                coherence_data = state.workspace_coherence
                if isinstance(coherence_data, dict):
                    # Try to get coherence from identity or evolution alignment
                    coherence = coherence_data.get("identity_alignment", 1.0)
                    if coherence is None:
                        coherence = coherence_data.get("evolution_alignment", 1.0) or 1.0
            elif hasattr(state, 'coherence') and hasattr(state.coherence, 'last_value'):
                coherence = state.coherence.last_value
            
            scores["anomaly_review"] = 1.0 - float(coherence) if coherence is not None else 0.0
        except Exception:
            scores["anomaly_review"] = 0.0

        # Determine purpose (highest score wins)
        if scores:
            purpose = max(scores, key=scores.get)
        else:
            purpose = "exploratory_reentry"

        return {
            "purpose": purpose,
            "scores": scores
        }

    def choose_reentry_mode(self, purpose):
        """
        Map purpose → behavior.
        """
        mapping = {
            "identity_reinforcement": "deep_reentry",
            "reflection_continuity": "shallow_reentry",
            "task_continuity": "task_reentry",
            "anomaly_review": "corrective_reentry",
        }

        return mapping.get(purpose, "exploratory_reentry")

