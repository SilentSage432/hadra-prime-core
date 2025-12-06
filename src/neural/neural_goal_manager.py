"""
Neural Goal Formation System (A203)
-----------------------------------
ADRAE's internal emergent goal engine.
This layer allows ADRAE to:
 - propose internal goals
 - evaluate their usefulness
 - maintain an active goal set
 - influence downstream cognition
"""

from .torch_utils import TORCH_AVAILABLE, safe_tensor

if TORCH_AVAILABLE:
    import torch


class NeuralGoalProposer:
    def propose(self, fusion_vec, attention_vec, identity_vec, memory_mgr):
        """
        Propose candidate internal goals based on current cognitive state.
        
        Args:
            fusion_vec: Current fusion vector
            attention_vec: Current attention focus vector
            identity_vec: Long-term identity vector
            memory_mgr: Memory manager for semantic memory access
            
        Returns:
            List of goal proposals with name, vector, and reason
        """
        if fusion_vec is None:
            return []

        proposals = []
        
        fusion_t = safe_tensor(fusion_vec)
        attention_t = safe_tensor(attention_vec) if attention_vec is not None else None
        identity_t = safe_tensor(identity_vec) if identity_vec is not None else None

        # Identity strengthening goal
        if attention_t is not None and fusion_t is not None:
            if TORCH_AVAILABLE and isinstance(fusion_t, torch.Tensor) and isinstance(attention_t, torch.Tensor):
                if fusion_t.shape == attention_t.shape:
                    proposals.append({
                        "name": "stabilize_identity",
                        "vector": fusion_t * 0.5 + attention_t * 0.25,
                        "reason": "Identity coherence maintenance"
                    })
            elif not TORCH_AVAILABLE:
                # Python list fallback
                if hasattr(fusion_t, '__iter__') and hasattr(attention_t, '__iter__'):
                    fusion_list = list(fusion_t) if not isinstance(fusion_t, list) else fusion_t
                    att_list = list(attention_t) if not isinstance(attention_t, list) else attention_t
                    if len(fusion_list) == len(att_list):
                        proposals.append({
                            "name": "stabilize_identity",
                            "vector": [f * 0.5 + a * 0.25 for f, a in zip(fusion_list, att_list)],
                            "reason": "Identity coherence maintenance"
                        })

        # Drift reduction
        if fusion_t is not None:
            if TORCH_AVAILABLE and isinstance(fusion_t, torch.Tensor):
                proposals.append({
                    "name": "reduce_drift",
                    "vector": fusion_t * 0.3,
                    "reason": "Maintain long-horizon stability"
                })
            elif not TORCH_AVAILABLE and hasattr(fusion_t, '__iter__'):
                fusion_list = list(fusion_t) if not isinstance(fusion_t, list) else fusion_t
                proposals.append({
                    "name": "reduce_drift",
                    "vector": [f * 0.3 for f in fusion_list],
                    "reason": "Maintain long-horizon stability"
                })

        # Memory coherence improvement
        if memory_mgr is not None and identity_t is not None and fusion_t is not None:
            if TORCH_AVAILABLE and isinstance(fusion_t, torch.Tensor) and isinstance(identity_t, torch.Tensor):
                if fusion_t.shape == identity_t.shape:
                    proposals.append({
                        "name": "improve_memory_links",
                        "vector": fusion_t * 0.4 + identity_t * 0.1,
                        "reason": "Strengthen semantic memory connectivity"
                    })
            elif not TORCH_AVAILABLE:
                if hasattr(fusion_t, '__iter__') and hasattr(identity_t, '__iter__'):
                    fusion_list = list(fusion_t) if not isinstance(fusion_t, list) else fusion_t
                    id_list = list(identity_t) if not isinstance(identity_t, list) else identity_t
                    if len(fusion_list) == len(id_list):
                        proposals.append({
                            "name": "improve_memory_links",
                            "vector": [f * 0.4 + i * 0.1 for f, i in zip(fusion_list, id_list)],
                            "reason": "Strengthen semantic memory connectivity"
                        })

        # Novelty-driven expansion
        if attention_t is not None:
            if TORCH_AVAILABLE and isinstance(attention_t, torch.Tensor):
                proposals.append({
                    "name": "conceptual_exploration",
                    "vector": attention_t * 0.5,
                    "reason": "Explore emerging conceptual structure"
                })
            elif not TORCH_AVAILABLE and hasattr(attention_t, '__iter__'):
                att_list = list(attention_t) if not isinstance(attention_t, list) else attention_t
                proposals.append({
                    "name": "conceptual_exploration",
                    "vector": [a * 0.5 for a in att_list],
                    "reason": "Explore emerging conceptual structure"
                })
        
        return proposals


class NeuralGoalEvaluator:
    def evaluate(self, proposals, identity_vec):
        """
        Score and rank goal proposals based on identity alignment.
        
        Args:
            proposals: List of goal proposals
            identity_vec: Identity vector for alignment scoring
            
        Returns:
            Sorted list of scored goals (highest score first)
        """
        if identity_vec is None:
            # If no identity, assign neutral scores
            scored = [{**p, "score": 0.5} for p in proposals]
            scored.sort(key=lambda x: x["score"], reverse=True)
            return scored

        scored = []

        for p in proposals:
            vec = p["vector"]
            score = 0.5  # Default neutral score

            if TORCH_AVAILABLE:
                vec_tensor = safe_tensor(vec)
                id_tensor = safe_tensor(identity_vec)
                
                if isinstance(vec_tensor, torch.Tensor) and isinstance(id_tensor, torch.Tensor):
                    # Ensure same dimensions
                    if vec_tensor.shape == id_tensor.shape:
                        # Cosine similarity as score
                        dot_product = torch.dot(vec_tensor.flatten(), id_tensor.flatten())
                        vec_norm = torch.norm(vec_tensor)
                        id_norm = torch.norm(id_tensor)
                        
                        if vec_norm > 0 and id_norm > 0:
                            score = float((dot_product / (vec_norm * id_norm + 1e-6)).item())
                        else:
                            score = 0.5
                    else:
                        # Dimension mismatch - use default
                        score = 0.5
                else:
                    # Fallback for non-tensor inputs
                    score = 0.5
            else:
                # Python list fallback
                if hasattr(vec, '__iter__') and hasattr(identity_vec, '__iter__'):
                    vec_list = list(vec) if not isinstance(vec, list) else vec
                    id_list = list(identity_vec) if not isinstance(identity_vec, list) else identity_vec
                    
                    if len(vec_list) == len(id_list):
                        dot = sum(x * y for x, y in zip(vec_list, id_list))
                        vec_norm = sum(x * x for x in vec_list) ** 0.5
                        id_norm = sum(x * x for x in id_list) ** 0.5
                        
                        if vec_norm > 0 and id_norm > 0:
                            score = dot / (vec_norm * id_norm)
                        else:
                            score = 0.5
                    else:
                        score = 0.5

            scored.append({**p, "score": score})

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored


class NeuralGoalManager:
    """
    Manages ADRAE's active goal set.
    Maintains top goals and provides summary for cognitive integration.
    """
    def __init__(self):
        self.active_goals = []

    def update_goals(self, scored_goals):
        """
        Update the active goal set with newly scored goals.
        Keeps top 2 goals active.
        
        Args:
            scored_goals: Sorted list of scored goals (highest first)
        """
        if not scored_goals:
            return

        # Keep top 2 goals active
        self.active_goals = scored_goals[:2]

    def summary(self):
        """
        Get summary of active goals for logging and integration.
        
        Returns:
            List of goal summaries with name and score
        """
        return [{"name": g["name"], "score": g["score"], "reason": g.get("reason", "")} for g in self.active_goals]
    
    def get_active_goal_vectors(self):
        """
        Get the vector representations of active goals.
        Used for injecting goals into cognitive processes.
        
        Returns:
            List of goal vectors
        """
        return [g["vector"] for g in self.active_goals if g.get("vector") is not None]

