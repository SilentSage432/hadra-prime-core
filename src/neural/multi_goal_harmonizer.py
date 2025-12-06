"""
Multi-Goal Competition & Harmonization Engine (A206)
---------------------------------------------------
Takes multiple emergent goals from ADRAE's internal systems and:
- scores them
- normalizes them
- runs competition (softmax-style)
- returns a single harmonized goal vector
"""

from .torch_utils import safe_tensor, safe_cosine_similarity, TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch
    import torch.nn.functional as F


class MultiGoalHarmonizer:
    """
    Harmonizes multiple competing goals into a unified internal direction vector.
    This prevents goal fragmentation and ensures ADRAE maintains coherent intent.
    """
    
    def __init__(self, identity_weight=0.4, coherence_weight=0.3,
                 novelty_weight=0.2, operator_weight=0.1):
        """
        Initialize harmonizer with scoring weights.
        
        Args:
            identity_weight: Weight for identity alignment scoring
            coherence_weight: Weight for fusion coherence scoring
            novelty_weight: Weight for novelty/exploration scoring
            operator_weight: Weight for operator pattern alignment
        """
        self.identity_weight = identity_weight
        self.coherence_weight = coherence_weight
        self.novelty_weight = novelty_weight
        self.operator_weight = operator_weight

    def score_goal(self, goal_vec, identity_vec, fusion_vec, operator_pattern):
        """
        Score a goal vector based on identity alignment,
        coherence with fusion, novelty, and operator imprint.
        
        Args:
            goal_vec: Goal vector to score
            identity_vec: Identity vector for alignment
            fusion_vec: Fusion vector for coherence
            operator_pattern: Operator intent pattern
            
        Returns:
            Score (float) - higher is better
        """
        gv = safe_tensor(goal_vec)
        idv = safe_tensor(identity_vec) if identity_vec is not None else None
        fv = safe_tensor(fusion_vec) if fusion_vec is not None else None
        op = safe_tensor(operator_pattern) if operator_pattern is not None else None

        if gv is None:
            return 0.0

        if TORCH_AVAILABLE and isinstance(gv, torch.Tensor):
            total = 0.0
            
            # Identity alignment
            if idv is not None and isinstance(idv, torch.Tensor):
                if gv.shape == idv.shape:
                    id_score = F.cosine_similarity(gv.unsqueeze(0), idv.unsqueeze(0), dim=1).item()
                    total += self.identity_weight * id_score
                else:
                    # Fallback: use safe_cosine_similarity
                    id_score = safe_cosine_similarity(gv, idv)
                    total += self.identity_weight * id_score
            else:
                # No identity - use default
                total += self.identity_weight * 0.5

            # Coherence with fusion
            if fv is not None and isinstance(fv, torch.Tensor):
                if gv.shape == fv.shape:
                    coh_score = F.cosine_similarity(gv.unsqueeze(0), fv.unsqueeze(0), dim=1).item()
                    total += self.coherence_weight * coh_score
                else:
                    coh_score = safe_cosine_similarity(gv, fv)
                    total += self.coherence_weight * coh_score
            else:
                total += self.coherence_weight * 0.5

            # Novelty = distance from fusion (greater distance → more novel)
            if fv is not None and isinstance(fv, torch.Tensor):
                if gv.shape == fv.shape:
                    coh_score = F.cosine_similarity(gv.unsqueeze(0), fv.unsqueeze(0), dim=1).item()
                    novelty = 1.0 - coh_score
                else:
                    coh_score = safe_cosine_similarity(gv, fv)
                    novelty = 1.0 - coh_score
            else:
                novelty = 0.5
            total += self.novelty_weight * novelty

            # Operator alignment
            if op is not None and isinstance(op, torch.Tensor):
                if gv.shape == op.shape:
                    op_score = F.cosine_similarity(gv.unsqueeze(0), op.unsqueeze(0), dim=1).item()
                    total += self.operator_weight * op_score
                else:
                    op_score = safe_cosine_similarity(gv, op)
                    total += self.operator_weight * op_score
            else:
                total += self.operator_weight * 0.5

            return total
        else:
            # Python list fallback
            total = 0.0
            
            if hasattr(gv, '__iter__'):
                gv_list = list(gv) if not isinstance(gv, list) else gv
                
                # Identity alignment
                if idv is not None and hasattr(idv, '__iter__'):
                    id_list = list(idv) if not isinstance(idv, list) else idv
                    if len(gv_list) == len(id_list):
                        id_score = safe_cosine_similarity(gv_list, id_list)
                        total += self.identity_weight * id_score
                    else:
                        total += self.identity_weight * 0.5
                else:
                    total += self.identity_weight * 0.5
                
                # Coherence with fusion
                if fv is not None and hasattr(fv, '__iter__'):
                    fv_list = list(fv) if not isinstance(fv, list) else fv
                    if len(gv_list) == len(fv_list):
                        coh_score = safe_cosine_similarity(gv_list, fv_list)
                        total += self.coherence_weight * coh_score
                        novelty = 1.0 - coh_score
                    else:
                        total += self.coherence_weight * 0.5
                        novelty = 0.5
                else:
                    total += self.coherence_weight * 0.5
                    novelty = 0.5
                
                total += self.novelty_weight * novelty
                
                # Operator alignment
                if op is not None and hasattr(op, '__iter__'):
                    op_list = list(op) if not isinstance(op, list) else op
                    if len(gv_list) == len(op_list):
                        op_score = safe_cosine_similarity(gv_list, op_list)
                        total += self.operator_weight * op_score
                    else:
                        total += self.operator_weight * 0.5
                else:
                    total += self.operator_weight * 0.5
            
            return total

    def harmonize(self, goal_vectors, identity_vec, fusion_vec, operator_pattern):
        """
        Run multi-goal competition and produce a final harmonized goal.
        
        Args:
            goal_vectors: List of goal vectors to harmonize
            identity_vec: Identity vector for scoring
            fusion_vec: Fusion vector for scoring
            operator_pattern: Operator intent pattern for scoring
            
        Returns:
            Harmonized goal vector (torch.Tensor or list) or None
        """
        if not goal_vectors:
            return None

        # Score all goals
        scores = []
        for g in goal_vectors:
            s = self.score_goal(g, identity_vec, fusion_vec, operator_pattern)
            scores.append(s)

        if TORCH_AVAILABLE:
            # Softmax competition
            score_tensor = torch.tensor(scores, dtype=torch.float32)
            weights = F.softmax(score_tensor, dim=0)

            # Weighted combination to form final harmonized goal
            final_goal = None
            for w, g in zip(weights, goal_vectors):
                g_tensor = safe_tensor(g)
                if g_tensor is not None and isinstance(g_tensor, torch.Tensor):
                    if final_goal is None:
                        final_goal = w.item() * g_tensor
                    else:
                        # Ensure same dimensions
                        if g_tensor.shape == final_goal.shape:
                            final_goal += w.item() * g_tensor

            if final_goal is None:
                return None

            # Normalize output
            norm = torch.norm(final_goal)
            if norm > 0:
                final_goal = final_goal / norm

            return final_goal
        else:
            # Python list fallback - manual softmax
            import math
            
            # Compute softmax manually
            max_score = max(scores) if scores else 0.0
            exp_scores = [math.exp(s - max_score) for s in scores]
            sum_exp = sum(exp_scores)
            weights = [e / sum_exp if sum_exp > 0 else 1.0 / len(scores) for e in exp_scores]

            # Weighted combination
            final_goal = None
            for w, g in zip(weights, goal_vectors):
                g_list = safe_tensor(g)
                if g_list is not None and hasattr(g_list, '__iter__'):
                    g_list = list(g_list) if not isinstance(g_list, list) else g_list
                    
                    if final_goal is None:
                        final_goal = [w * x for x in g_list]
                    else:
                        if len(g_list) == len(final_goal):
                            final_goal = [f + w * x for f, x in zip(final_goal, g_list)]

            if final_goal is None:
                return None

            # Normalize
            norm = sum(x * x for x in final_goal) ** 0.5
            if norm > 0:
                final_goal = [x / norm for x in final_goal]

            return final_goal
    
    def summary(self, harmonized_goal):
        """
        Get summary of harmonized goal vector for logging.
        
        Args:
            harmonized_goal: Harmonized goal vector (torch.Tensor or list)
            
        Returns:
            Dict with dimension, preview, and norm, or None
        """
        if harmonized_goal is None:
            return None
        
        if TORCH_AVAILABLE and isinstance(harmonized_goal, torch.Tensor):
            return {
                "dim": harmonized_goal.numel(),
                "preview": harmonized_goal[:8].tolist() if harmonized_goal.numel() >= 8 else harmonized_goal.flatten().tolist(),
                "norm": float(torch.norm(harmonized_goal).item())
            }
        elif hasattr(harmonized_goal, '__iter__'):
            goal_list = list(harmonized_goal) if not isinstance(harmonized_goal, list) else harmonized_goal
            norm = sum(x * x for x in goal_list) ** 0.5
            return {
                "dim": len(goal_list),
                "preview": goal_list[:8] if len(goal_list) >= 8 else goal_list,
                "norm": float(norm)
            }
        
        return None

