"""
Dynamic Subgoal Routing & Execution Priority (A210)
---------------------------------------------------
Routes subgoals into live cognitive execution pathways.
This enables ADRAE to transition from intent structure to actual strategy formation.
"""

from .torch_utils import safe_tensor, safe_cosine_similarity, TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch
    import torch.nn.functional as F


class SubgoalRoutingEngine:
    """
    Routes subgoals into cognitive execution pathways by:
    - Building priority queues
    - Determining execution order
    - Integrating into action selection
    - Dynamically rerouting based on fusion, drift, and goal alignment
    """
    
    def __init__(self,
                 goal_weight=0.40,
                 fusion_weight=0.30,
                 strength_weight=0.20,
                 drift_weight=0.10,
                 fusion_influence=0.15,
                 attention_influence=0.12):
        """
        Initialize routing engine.
        
        Args:
            goal_weight: Weight for alignment with main goal
            fusion_weight: Weight for alignment with fusion
            strength_weight: Weight for subgoal strength
            drift_weight: Weight for drift resistance
            fusion_influence: Influence weight for fusion vector (default 0.15)
            attention_influence: Influence weight for attention vector (default 0.12)
        """
        self.goal_weight = goal_weight
        self.fusion_weight = fusion_weight
        self.strength_weight = strength_weight
        self.drift_weight = drift_weight
        self.fusion_influence = fusion_influence
        self.attention_influence = attention_influence

    def route(self, active_subgoals, goal_vector, fusion_vector, attention_vector, drift_value):
        """
        Route subgoals into execution pathways by computing priorities and selecting next route.
        
        Args:
            active_subgoals: List of active subgoal dicts
            goal_vector: Main harmonized goal vector
            fusion_vector: Current fusion vector
            attention_vector: Current attention vector
            drift_value: Current drift value
            
        Returns:
            Dict with routing state including priority queue and current route
        """
        if not active_subgoals or len(active_subgoals) == 0:
            return {
                "active_subgoals": 0,
                "priority_queue": [],
                "current_route": None,
                "modified_fusion": fusion_vector,
                "modified_attention": attention_vector
            }
        
        goal_t = safe_tensor(goal_vector) if goal_vector is not None else None
        fusion_t = safe_tensor(fusion_vector) if fusion_vector is not None else None
        attention_t = safe_tensor(attention_vector) if attention_vector is not None else None
        
        if TORCH_AVAILABLE:
            # 1. Compute execution priority for each subgoal
            priority_queue = []
            
            for sg in active_subgoals:
                sg_vec = safe_tensor(sg.get("vector"))
                if sg_vec is None or not isinstance(sg_vec, torch.Tensor):
                    sg["priority"] = 0.0
                    priority_queue.append(sg)
                    continue
                
                # Alignment with main goal (executive relevance)
                align_goal = 0.5  # Default neutral
                if goal_t is not None and isinstance(goal_t, torch.Tensor):
                    if sg_vec.shape == goal_t.shape:
                        align_goal = F.cosine_similarity(
                            sg_vec.unsqueeze(0),
                            goal_t.unsqueeze(0),
                            dim=1
                        ).item()
                    else:
                        align_goal = safe_cosine_similarity(sg_vec, goal_t)
                
                # Alignment with current fusion (moment relevance)
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
                
                # Resistance against drift (stability relevance)
                drift_penalty = abs(drift_value) if drift_value is not None else 0.0
                drift_factor = 1.0 - min(drift_penalty, 1.0)
                
                # Strength from A209 (competition relevance)
                strength = sg.get("strength", 1.0)
                
                # Execution priority = composite score
                priority = (
                    self.goal_weight * align_goal +
                    self.fusion_weight * align_fusion +
                    self.strength_weight * strength +
                    self.drift_weight * drift_factor
                )
                
                sg["priority"] = float(priority)
                priority_queue.append(sg)
            
            # 2. Sort subgoals by execution priority
            priority_queue = sorted(priority_queue, key=lambda x: x.get("priority", 0.0), reverse=True)
            
            # 3. Select the next subgoal to influence cognitive action
            next_subgoal = priority_queue[0] if priority_queue else None
            
            if next_subgoal is None:
                return {
                    "active_subgoals": len(active_subgoals),
                    "priority_queue": [],
                    "current_route": None,
                    "modified_fusion": fusion_vector,
                    "modified_attention": attention_vector
                }
            
            subgoal_vec = safe_tensor(next_subgoal["vector"])
            if subgoal_vec is None or not isinstance(subgoal_vec, torch.Tensor):
                return {
                    "active_subgoals": len(active_subgoals),
                    "priority_queue": [
                        {"id": sg.get("id"), "priority": round(sg.get("priority", 0.0), 4), 
                         "strength": round(sg.get("strength", 0.0), 3)}
                        for sg in priority_queue
                    ],
                    "current_route": None,
                    "modified_fusion": fusion_vector,
                    "modified_attention": attention_vector
                }
            
            # 4. Integrate into action routing
            # Fusion is nudged toward the execution trajectory
            modified_fusion = fusion_t.clone() if fusion_t is not None else None
            if modified_fusion is not None and isinstance(modified_fusion, torch.Tensor):
                if modified_fusion.shape == subgoal_vec.shape:
                    modified_fusion = (modified_fusion * (1.0 - self.fusion_influence)) + (subgoal_vec * self.fusion_influence)
                    modified_fusion = F.normalize(modified_fusion, dim=0)
            
            # Attention is steered as well
            modified_attention = attention_t.clone() if attention_t is not None else None
            if modified_attention is not None and isinstance(modified_attention, torch.Tensor):
                if modified_attention.shape == subgoal_vec.shape:
                    modified_attention = (modified_attention * (1.0 - self.attention_influence)) + (subgoal_vec * self.attention_influence)
                    modified_attention = F.normalize(modified_attention, dim=0)
            
            # 5. Store current route
            current_route = {
                "active": next_subgoal.get("id"),
                "priority": round(next_subgoal.get("priority", 0.0), 4),
                "vector": subgoal_vec
            }
            
            return {
                "active_subgoals": len(priority_queue),
                "priority_queue": [
                    {
                        "id": sg.get("id", "unknown"),
                        "priority": round(sg.get("priority", 0.0), 4),
                        "strength": round(sg.get("strength", 0.0), 3)
                    }
                    for sg in priority_queue
                ],
                "current_route": current_route,
                "modified_fusion": modified_fusion,
                "modified_attention": modified_attention
            }
        else:
            # Python list fallback
            priority_queue = []
            
            for sg in active_subgoals:
                sg_vec = safe_tensor(sg.get("vector"))
                if sg_vec is None or not hasattr(sg_vec, '__iter__'):
                    sg["priority"] = 0.0
                    priority_queue.append(sg)
                    continue
                
                sg_list = list(sg_vec) if not isinstance(sg_vec, list) else sg_vec
                
                # Alignment with main goal
                align_goal = 0.5
                if goal_t is not None and hasattr(goal_t, '__iter__'):
                    goal_list = list(goal_t) if not isinstance(goal_t, list) else goal_t
                    if len(sg_list) == len(goal_list):
                        align_goal = safe_cosine_similarity(sg_list, goal_list)
                
                # Alignment with fusion
                align_fusion = 0.5
                if fusion_t is not None and hasattr(fusion_t, '__iter__'):
                    fusion_list = list(fusion_t) if not isinstance(fusion_t, list) else fusion_t
                    if len(sg_list) == len(fusion_list):
                        align_fusion = safe_cosine_similarity(sg_list, fusion_list)
                
                # Drift factor
                drift_penalty = abs(drift_value) if drift_value is not None else 0.0
                drift_factor = 1.0 - min(drift_penalty, 1.0)
                
                # Strength
                strength = sg.get("strength", 1.0)
                
                # Priority
                priority = (
                    self.goal_weight * align_goal +
                    self.fusion_weight * align_fusion +
                    self.strength_weight * strength +
                    self.drift_weight * drift_factor
                )
                
                sg["priority"] = float(priority)
                priority_queue.append(sg)
            
            # Sort by priority
            priority_queue = sorted(priority_queue, key=lambda x: x.get("priority", 0.0), reverse=True)
            
            next_subgoal = priority_queue[0] if priority_queue else None
            
            if next_subgoal is None:
                return {
                    "active_subgoals": len(active_subgoals),
                    "priority_queue": [],
                    "current_route": None,
                    "modified_fusion": fusion_vector,
                    "modified_attention": attention_vector
                }
            
            subgoal_vec = safe_tensor(next_subgoal["vector"])
            if subgoal_vec is None or not hasattr(subgoal_vec, '__iter__'):
                return {
                    "active_subgoals": len(active_subgoals),
                    "priority_queue": [
                        {"id": sg.get("id"), "priority": round(sg.get("priority", 0.0), 4),
                         "strength": round(sg.get("strength", 0.0), 3)}
                        for sg in priority_queue
                    ],
                    "current_route": None,
                    "modified_fusion": fusion_vector,
                    "modified_attention": attention_vector
                }
            
            subgoal_list = list(subgoal_vec) if not isinstance(subgoal_vec, list) else subgoal_vec
            
            # Modify fusion
            modified_fusion = fusion_vector
            if fusion_t is not None and hasattr(fusion_t, '__iter__'):
                fusion_list = list(fusion_t) if not isinstance(fusion_t, list) else fusion_t
                if len(fusion_list) == len(subgoal_list):
                    modified_fusion = [
                        f * (1.0 - self.fusion_influence) + s * self.fusion_influence
                        for f, s in zip(fusion_list, subgoal_list)
                    ]
                    # Normalize
                    norm = sum(x * x for x in modified_fusion) ** 0.5
                    if norm > 0:
                        modified_fusion = [x / norm for x in modified_fusion]
            
            # Modify attention
            modified_attention = attention_vector
            if attention_t is not None and hasattr(attention_t, '__iter__'):
                att_list = list(attention_t) if not isinstance(attention_t, list) else attention_t
                if len(att_list) == len(subgoal_list):
                    modified_attention = [
                        a * (1.0 - self.attention_influence) + s * self.attention_influence
                        for a, s in zip(att_list, subgoal_list)
                    ]
                    # Normalize
                    norm = sum(x * x for x in modified_attention) ** 0.5
                    if norm > 0:
                        modified_attention = [x / norm for x in modified_attention]
            
            # Current route
            current_route = {
                "active": next_subgoal.get("id"),
                "priority": round(next_subgoal.get("priority", 0.0), 4),
                "vector": subgoal_list
            }
            
            return {
                "active_subgoals": len(priority_queue),
                "priority_queue": [
                    {
                        "id": sg.get("id", "unknown"),
                        "priority": round(sg.get("priority", 0.0), 4),
                        "strength": round(sg.get("strength", 0.0), 3)
                    }
                    for sg in priority_queue
                ],
                "current_route": current_route,
                "modified_fusion": modified_fusion,
                "modified_attention": modified_attention
            }
    
    def summary(self, routing_state):
        """
        Get summary of routing state for logging.
        
        Args:
            routing_state: Dict returned from route()
            
        Returns:
            Dict with previews and key metrics
        """
        if routing_state is None:
            return None
        
        summary = {
            "active_subgoals": routing_state.get("active_subgoals", 0),
            "priority_queue": routing_state.get("priority_queue", []),
            "route": routing_state.get("current_route")
        }
        
        # Add fusion preview
        modified_fusion = routing_state.get("modified_fusion")
        if modified_fusion is not None:
            if TORCH_AVAILABLE and isinstance(modified_fusion, torch.Tensor):
                summary["fusion_preview"] = modified_fusion[:8].tolist() if modified_fusion.numel() >= 8 else modified_fusion.flatten().tolist()
            elif hasattr(modified_fusion, '__iter__'):
                fusion_list = list(modified_fusion) if not isinstance(modified_fusion, list) else modified_fusion
                summary["fusion_preview"] = fusion_list[:8] if len(fusion_list) >= 8 else fusion_list
        
        return summary

