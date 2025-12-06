"""
Goal Modulation Engine (A204)
----------------------------
Transforms ADRAE's active goals into modulation signals used to
reshape thought selection, reflection, attention, and memory.
"""

from .torch_utils import safe_tensor, TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch


class GoalModulationEngine:
    """
    Computes modulation vectors from active goals that influence
    cognitive processes across the pipeline.
    """

    def compute_modulation(self, active_goals, fusion_vec):
        """
        Returns a modulation vector shaped by the goals.
        This vector is added into:
         - thought scoring
         - attention shaping
         - memory relevance
        
        Args:
            active_goals: List of active goal dicts with "name" and "score"
            fusion_vec: Current fusion vector to use as base
            
        Returns:
            Modulation vector (torch.Tensor or list) or None
        """
        if fusion_vec is None or not active_goals:
            return None

        base = safe_tensor(fusion_vec)
        
        if base is None:
            return None

        if TORCH_AVAILABLE and isinstance(base, torch.Tensor):
            combined = torch.zeros_like(base)

            for g in active_goals:
                name = g.get("name", "")
                score = g.get("score", 0.5)

                if name == "stabilize_identity":
                    combined += base * (0.20 * score)

                elif name == "reduce_drift":
                    combined += base * (0.15 * score)

                elif name == "improve_memory_links":
                    combined += base * (0.10 * score)

                elif name == "conceptual_exploration":
                    noise = torch.randn_like(base) * (0.05 * score)
                    combined += noise

            # Normalize modulation vector
            n = torch.norm(combined)
            if n > 0:
                combined = combined / n

            return combined
        else:
            # Python list fallback
            if hasattr(base, '__iter__'):
                base_list = list(base) if not isinstance(base, list) else base
                combined = [0.0] * len(base_list)
                
                for g in active_goals:
                    name = g.get("name", "")
                    score = g.get("score", 0.5)
                    
                    if name == "stabilize_identity":
                        combined = [c + b * (0.20 * score) for c, b in zip(combined, base_list)]
                    elif name == "reduce_drift":
                        combined = [c + b * (0.15 * score) for c, b in zip(combined, base_list)]
                    elif name == "improve_memory_links":
                        combined = [c + b * (0.10 * score) for c, b in zip(combined, base_list)]
                    elif name == "conceptual_exploration":
                        import random
                        noise = [random.gauss(0, 0.05 * score) for _ in base_list]
                        combined = [c + n for c, n in zip(combined, noise)]
                
                # Normalize
                norm = sum(x * x for x in combined) ** 0.5
                if norm > 0:
                    combined = [x / norm for x in combined]
                
                return combined
        
        return None

    def summary(self, mod):
        """
        Get summary of modulation vector for logging.
        
        Args:
            mod: Modulation vector (torch.Tensor or list)
            
        Returns:
            Dict with dimension and preview, or None
        """
        if mod is None:
            return None
        
        if TORCH_AVAILABLE and isinstance(mod, torch.Tensor):
            return {
                "dim": mod.numel(),
                "preview": mod[:8].tolist() if mod.numel() >= 8 else mod.flatten().tolist()
            }
        elif hasattr(mod, '__iter__'):
            mod_list = list(mod) if not isinstance(mod, list) else mod
            return {
                "dim": len(mod_list),
                "preview": mod_list[:8] if len(mod_list) >= 8 else mod_list
            }
        
        return None

