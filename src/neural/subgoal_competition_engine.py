"""
Subgoal Competition & Selection Layer (A209)
---------------------------------------------
ADRAE's subgoals compete, cooperate, and negotiate for cognitive priority.
This enables structured internal deliberation and adaptive planning.
"""

from .torch_utils import safe_tensor, safe_cosine_similarity, TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch
    import torch.nn.functional as F


class SubgoalCompetitionEngine:
    """
    Manages competition between subgoals, selecting winners and suppressing
    weaker alternatives. This enables structured internal deliberation.
    """
    
    def __init__(self, 
                 main_goal_weight=0.45,
                 fusion_weight=0.35,
                 strength_weight=0.15,
                 drift_weight=0.05,
                 suppression_rate=0.92,
                 enhancement_rate=1.08,
                 max_winner_strength=1.5):
        """
        Initialize competition engine.
        
        Args:
            main_goal_weight: Weight for alignment with main goal
            fusion_weight: Weight for alignment with fusion direction
            strength_weight: Weight for subgoal strength
            drift_weight: Weight for drift correction
            suppression_rate: Rate at which non-winners are suppressed
            enhancement_rate: Rate at which winner is enhanced
            max_winner_strength: Maximum strength for winning subgoal
        """
        self.main_goal_weight = main_goal_weight
        self.fusion_weight = fusion_weight
        self.strength_weight = strength_weight
        self.drift_weight = drift_weight
        self.suppression_rate = suppression_rate
        self.enhancement_rate = enhancement_rate
        self.max_winner_strength = max_winner_strength

    def compete(self, active_subgoals, goal_vector, fusion_vector, drift_value):
        """
        Run competition between subgoals and select winner.
        
        Args:
            active_subgoals: List of active subgoal dicts with "vector" and "strength"
            goal_vector: Main harmonized goal vector
            fusion_vector: Current fusion vector
            drift_value: Current drift value for penalty calculation
            
        Returns:
            Dict with competition results including winner and competition vector
        """
        if not active_subgoals or len(active_subgoals) == 0:
            return {
                "active_subgoals": 0,
                "winner": None,
                "competition_vector": None,
                "scores": []
            }
        
        goal_t = safe_tensor(goal_vector) if goal_vector is not None else None
        fusion_t = safe_tensor(fusion_vector) if fusion_vector is not None else None
        
        if TORCH_AVAILABLE:
            # Score each subgoal
            for sg in active_subgoals:
                sg_vec = safe_tensor(sg.get("vector"))
                if sg_vec is None or not isinstance(sg_vec, torch.Tensor):
                    sg["score"] = 0.0
                    continue
                
                # Distance between subgoal and main goal
                align_main = 0.5  # Default neutral
                if goal_t is not None and isinstance(goal_t, torch.Tensor):
                    if sg_vec.shape == goal_t.shape:
                        align_main = F.cosine_similarity(
                            sg_vec.unsqueeze(0), 
                            goal_t.unsqueeze(0), 
                            dim=1
                        ).item()
                    else:
                        align_main = safe_cosine_similarity(sg_vec, goal_t)
                
                # Alignment with current fusion direction
                align_fusion = 0.5  # Default neutral
                if fusion_t is not None and isinstance(fusion_t, torch.Tensor):
                    if sg_vec.shape == fusion_t.shape:
                        align_fusion = F.cosine_similarity(
                            sg_vec.unsqueeze(0),
                            fusion_t.unsqueeze(0),
                            dim=1
                        ).item()
                    else:
                        align_fusion = safe_cosine_similarity(sg_vec, fusion_t)
                
                # Drift correction: subgoals counteract drift
                drift_penalty = abs(drift_value) if drift_value is not None else 0.0
                drift_factor = 1.0 - min(drift_penalty, 1.0)  # Clamp to [0, 1]
                
                # Novelty factor: exploration is useful but controlled
                import random
                novelty = random.gauss(0, 0.02)  # tiny influence
                
                # Subgoal strength
                strength = sg.get("strength", 0.5)
                
                # Final score (weighted sum)
                score = (
                    self.main_goal_weight * align_main +
                    self.fusion_weight * align_fusion +
                    self.strength_weight * strength +
                    self.drift_weight * drift_factor +
                    novelty
                )
                
                sg["score"] = float(score)
            
            # Rank subgoals by score
            ranked = sorted(active_subgoals, key=lambda x: x.get("score", 0.0), reverse=True)
            winner = ranked[0] if ranked else None
            
            if winner is None:
                return {
                    "active_subgoals": len(active_subgoals),
                    "winner": None,
                    "competition_vector": None,
                    "scores": []
                }
            
            # Apply competitive suppression to non-winners
            for sg in ranked[1:]:
                sg["strength"] *= self.suppression_rate
            
            # Enhance the winning subgoal
            winner["strength"] = min(winner["strength"] * self.enhancement_rate, self.max_winner_strength)
            
            # Produce a competition influence vector
            winner_vec = safe_tensor(winner["vector"])
            if winner_vec is not None and isinstance(winner_vec, torch.Tensor):
                competition_vector = winner_vec * winner["strength"]
                competition_vector = F.normalize(competition_vector, dim=0)
            else:
                competition_vector = None
            
            return {
                "active_subgoals": len(active_subgoals),
                "winner": winner.get("id"),
                "winner_strength": round(winner["strength"], 3),
                "competition_vector": competition_vector,
                "scores": [
                    {
                        "id": sg.get("id", "unknown"),
                        "score": round(sg.get("score", 0.0), 4),
                        "strength": round(sg.get("strength", 0.0), 3)
                    }
                    for sg in ranked
                ]
            }
        else:
            # Python list fallback
            # Score each subgoal
            for sg in active_subgoals:
                sg_vec = safe_tensor(sg.get("vector"))
                if sg_vec is None or not hasattr(sg_vec, '__iter__'):
                    sg["score"] = 0.0
                    continue
                
                sg_list = list(sg_vec) if not isinstance(sg_vec, list) else sg_vec
                
                # Alignment with main goal
                align_main = 0.5
                if goal_t is not None and hasattr(goal_t, '__iter__'):
                    goal_list = list(goal_t) if not isinstance(goal_t, list) else goal_t
                    if len(sg_list) == len(goal_list):
                        align_main = safe_cosine_similarity(sg_list, goal_list)
                
                # Alignment with fusion
                align_fusion = 0.5
                if fusion_t is not None and hasattr(fusion_t, '__iter__'):
                    fusion_list = list(fusion_t) if not isinstance(fusion_t, list) else fusion_t
                    if len(sg_list) == len(fusion_list):
                        align_fusion = safe_cosine_similarity(sg_list, fusion_list)
                
                # Drift correction
                drift_penalty = abs(drift_value) if drift_value is not None else 0.0
                drift_factor = 1.0 - min(drift_penalty, 1.0)
                
                # Novelty
                import random
                novelty = random.gauss(0, 0.02)
                
                # Strength
                strength = sg.get("strength", 0.5)
                
                # Score
                score = (
                    self.main_goal_weight * align_main +
                    self.fusion_weight * align_fusion +
                    self.strength_weight * strength +
                    self.drift_weight * drift_factor +
                    novelty
                )
                
                sg["score"] = float(score)
            
            # Rank subgoals
            ranked = sorted(active_subgoals, key=lambda x: x.get("score", 0.0), reverse=True)
            winner = ranked[0] if ranked else None
            
            if winner is None:
                return {
                    "active_subgoals": len(active_subgoals),
                    "winner": None,
                    "competition_vector": None,
                    "scores": []
                }
            
            # Suppress non-winners
            for sg in ranked[1:]:
                sg["strength"] *= self.suppression_rate
            
            # Enhance winner
            winner["strength"] = min(winner["strength"] * self.enhancement_rate, self.max_winner_strength)
            
            # Competition vector
            winner_vec = safe_tensor(winner["vector"])
            if winner_vec is not None and hasattr(winner_vec, '__iter__'):
                winner_list = list(winner_vec) if not isinstance(winner_vec, list) else winner_vec
                competition_vector = [v * winner["strength"] for v in winner_list]
                # Normalize
                norm = sum(x * x for x in competition_vector) ** 0.5
                if norm > 0:
                    competition_vector = [x / norm for x in competition_vector]
            else:
                competition_vector = None
            
            return {
                "active_subgoals": len(active_subgoals),
                "winner": winner.get("id"),
                "winner_strength": round(winner["strength"], 3),
                "competition_vector": competition_vector,
                "scores": [
                    {
                        "id": sg.get("id", "unknown"),
                        "score": round(sg.get("score", 0.0), 4),
                        "strength": round(sg.get("strength", 0.0), 3)
                    }
                    for sg in ranked
                ]
            }
    
    def apply_competition(self, fusion_vector, competition_vector, influence_weight=0.20):
        """
        Apply competition winner's influence to fusion vector.
        
        Args:
            fusion_vector: Current fusion vector
            competition_vector: Competition winner vector
            influence_weight: Weight for competition influence (default 0.20)
            
        Returns:
            Modified fusion vector
        """
        if fusion_vector is None or competition_vector is None:
            return fusion_vector
        
        fusion_t = safe_tensor(fusion_vector)
        comp_t = safe_tensor(competition_vector)
        
        if TORCH_AVAILABLE and isinstance(fusion_t, torch.Tensor) and isinstance(comp_t, torch.Tensor):
            if fusion_t.shape == comp_t.shape:
                # Merge: 80% fusion, 20% competition
                blended = fusion_t * (1.0 - influence_weight) + comp_t * influence_weight
                return F.normalize(blended, dim=0)
        elif hasattr(fusion_t, '__iter__') and hasattr(comp_t, '__iter__'):
            fusion_list = list(fusion_t) if not isinstance(fusion_t, list) else fusion_t
            comp_list = list(comp_t) if not isinstance(comp_t, list) else comp_t
            if len(fusion_list) == len(comp_list):
                blended = [f * (1.0 - influence_weight) + c * influence_weight 
                          for f, c in zip(fusion_list, comp_list)]
                # Normalize
                norm = sum(x * x for x in blended) ** 0.5
                if norm > 0:
                    blended = [x / norm for x in blended]
                return blended
        
        return fusion_vector
    
    def summary(self, competition_state):
        """
        Get summary of competition state for logging.
        
        Args:
            competition_state: Dict returned from compete()
            
        Returns:
            Dict with previews and key metrics
        """
        if competition_state is None:
            return None
        
        summary = {
            "active_subgoals": competition_state.get("active_subgoals", 0),
            "winner": competition_state.get("winner"),
            "winner_strength": competition_state.get("winner_strength"),
            "scores": competition_state.get("scores", [])
        }
        
        # Add competition vector preview
        comp_vec = competition_state.get("competition_vector")
        if comp_vec is not None:
            if TORCH_AVAILABLE and isinstance(comp_vec, torch.Tensor):
                summary["competition_preview"] = comp_vec[:8].tolist() if comp_vec.numel() >= 8 else comp_vec.flatten().tolist()
            elif hasattr(comp_vec, '__iter__'):
                comp_list = list(comp_vec) if not isinstance(comp_vec, list) else comp_vec
                summary["competition_preview"] = comp_list[:8] if len(comp_list) >= 8 else comp_list
        
        return summary

