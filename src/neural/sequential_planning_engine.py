"""
Multi-Step Execution Chains (Sequential Planning Engine) (A211)
---------------------------------------------------------------
Builds multi-step plans from prioritized subgoals.
This enables ADRAE to form sequential execution chains and proto-executive function.
"""

from .torch_utils import safe_tensor, safe_cosine_similarity, TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch
    import torch.nn.functional as F


class SequentialPlanningEngine:
    """
    Forms multi-step execution chains from prioritized subgoals.
    This is the birth of procedural cognition and strategic planning.
    """
    
    def __init__(self, priority_weight=0.7, dependency_weight=0.3, fusion_influence=0.20, attention_influence=0.18):
        """
        Initialize planning engine.
        
        Args:
            priority_weight: Weight for priority in chain ordering
            dependency_weight: Weight for dependency in chain ordering
            fusion_influence: Influence weight for fusion vector (default 0.20)
            attention_influence: Influence weight for attention vector (default 0.18)
        """
        self.priority_weight = priority_weight
        self.dependency_weight = dependency_weight
        self.fusion_influence = fusion_influence
        self.attention_influence = attention_influence
        
        # Plan buffer
        self.plan_buffer = {
            "chain": [],
            "current_step": 0,
            "valid": False
        }

    def plan(self, active_subgoals, route, fusion_vector, attention_vector):
        """
        Build multi-step execution chain from prioritized subgoals.
        
        Args:
            active_subgoals: List of active subgoal dicts with priority
            route: Current route dict from A210 (contains active subgoal)
            fusion_vector: Current fusion vector
            attention_vector: Current attention vector
            
        Returns:
            Dict with planning state including chain and modified vectors
        """
        # No route or insufficient subgoals means no sequential planning
        if route is None or not active_subgoals or len(active_subgoals) < 2:
            return {
                "chain_length": 0,
                "execution_order": [],
                "current_step": None,
                "modified_fusion": fusion_vector,
                "modified_attention": attention_vector,
                "plan_valid": False
            }
        
        # 1. Sort subgoals by priority for clean ordering
        ordered = sorted(
            active_subgoals,
            key=lambda sg: sg.get("priority", 0.0),
            reverse=True
        )
        
        if TORCH_AVAILABLE:
            # 2. Build sequential execution chain with dependency analysis
            chain = []
            
            for idx, sg in enumerate(ordered):
                sg_vec = safe_tensor(sg.get("vector"))
                if sg_vec is None or not isinstance(sg_vec, torch.Tensor):
                    dep_score = 0.0
                else:
                    # Predict dependency ordering by comparing similarity to adjacent subgoals
                    dep_score = 0.0
                    
                    if idx + 1 < len(ordered):
                        next_sg = ordered[idx + 1]
                        next_vec = safe_tensor(next_sg.get("vector"))
                        if next_vec is not None and isinstance(next_vec, torch.Tensor):
                            if sg_vec.shape == next_vec.shape:
                                next_sim = F.cosine_similarity(
                                    sg_vec.unsqueeze(0),
                                    next_vec.unsqueeze(0),
                                    dim=1
                                ).item()
                                dep_score += next_sim
                    
                    if idx > 0:
                        prev_sg = ordered[idx - 1]
                        prev_vec = safe_tensor(prev_sg.get("vector"))
                        if prev_vec is not None and isinstance(prev_vec, torch.Tensor):
                            if sg_vec.shape == prev_vec.shape:
                                prev_sim = F.cosine_similarity(
                                    sg_vec.unsqueeze(0),
                                    prev_vec.unsqueeze(0),
                                    dim=1
                                ).item()
                                dep_score += prev_sim
                
                chain.append({
                    "id": sg.get("id", "unknown"),
                    "priority": sg.get("priority", 0.0),
                    "dependency": dep_score,
                    "vector": sg.get("vector")
                })
            
            # 3. Sort chain nodes by priority + dependency
            chain = sorted(
                chain,
                key=lambda x: (x["priority"] * self.priority_weight + x["dependency"] * self.dependency_weight),
                reverse=True
            )
            
            # 4. Store into plan buffer
            self.plan_buffer["chain"] = chain
            self.plan_buffer["current_step"] = 0
            self.plan_buffer["valid"] = True
            
            # 5. Influence cognitive action selection with current step
            current_step_vector = safe_tensor(chain[0]["vector"]) if chain else None
            modified_fusion = fusion_vector
            modified_attention = attention_vector
            
            if current_step_vector is not None and isinstance(current_step_vector, torch.Tensor):
                fusion_t = safe_tensor(fusion_vector)
                attention_t = safe_tensor(attention_vector)
                
                if fusion_t is not None and isinstance(fusion_t, torch.Tensor):
                    if fusion_t.shape == current_step_vector.shape:
                        modified_fusion = (fusion_t * (1.0 - self.fusion_influence)) + (current_step_vector * self.fusion_influence)
                        modified_fusion = F.normalize(modified_fusion, dim=0)
                
                if attention_t is not None and isinstance(attention_t, torch.Tensor):
                    if attention_t.shape == current_step_vector.shape:
                        modified_attention = (attention_t * (1.0 - self.attention_influence)) + (current_step_vector * self.attention_influence)
                        modified_attention = F.normalize(modified_attention, dim=0)
            
            return {
                "chain_length": len(chain),
                "execution_order": [
                    {
                        "id": node["id"],
                        "priority": round(node["priority"], 4),
                        "dependency": round(node["dependency"], 4)
                    }
                    for node in chain
                ],
                "current_step": chain[0]["id"] if chain else None,
                "modified_fusion": modified_fusion,
                "modified_attention": modified_attention,
                "plan_valid": True
            }
        else:
            # Python list fallback
            # 2. Build chain with dependency analysis
            chain = []
            
            for idx, sg in enumerate(ordered):
                sg_vec = safe_tensor(sg.get("vector"))
                if sg_vec is None or not hasattr(sg_vec, '__iter__'):
                    dep_score = 0.0
                else:
                    sg_list = list(sg_vec) if not isinstance(sg_vec, list) else sg_vec
                    dep_score = 0.0
                    
                    if idx + 1 < len(ordered):
                        next_sg = ordered[idx + 1]
                        next_vec = safe_tensor(next_sg.get("vector"))
                        if next_vec is not None and hasattr(next_vec, '__iter__'):
                            next_list = list(next_vec) if not isinstance(next_vec, list) else next_vec
                            if len(sg_list) == len(next_list):
                                next_sim = safe_cosine_similarity(sg_list, next_list)
                                dep_score += next_sim
                    
                    if idx > 0:
                        prev_sg = ordered[idx - 1]
                        prev_vec = safe_tensor(prev_sg.get("vector"))
                        if prev_vec is not None and hasattr(prev_vec, '__iter__'):
                            prev_list = list(prev_vec) if not isinstance(prev_vec, list) else prev_vec
                            if len(sg_list) == len(prev_list):
                                prev_sim = safe_cosine_similarity(sg_list, prev_list)
                                dep_score += prev_sim
                
                chain.append({
                    "id": sg.get("id", "unknown"),
                    "priority": sg.get("priority", 0.0),
                    "dependency": dep_score,
                    "vector": sg.get("vector")
                })
            
            # 3. Sort by priority + dependency
            chain = sorted(
                chain,
                key=lambda x: (x["priority"] * self.priority_weight + x["dependency"] * self.dependency_weight),
                reverse=True
            )
            
            # 4. Store plan buffer
            self.plan_buffer["chain"] = chain
            self.plan_buffer["current_step"] = 0
            self.plan_buffer["valid"] = True
            
            # 5. Apply influence
            current_step_vector = safe_tensor(chain[0]["vector"]) if chain else None
            modified_fusion = fusion_vector
            modified_attention = attention_vector
            
            if current_step_vector is not None and hasattr(current_step_vector, '__iter__'):
                current_list = list(current_step_vector) if not isinstance(current_step_vector, list) else current_step_vector
                
                fusion_t = safe_tensor(fusion_vector)
                attention_t = safe_tensor(attention_vector)
                
                if fusion_t is not None and hasattr(fusion_t, '__iter__'):
                    fusion_list = list(fusion_t) if not isinstance(fusion_t, list) else fusion_t
                    if len(fusion_list) == len(current_list):
                        modified_fusion = [
                            f * (1.0 - self.fusion_influence) + c * self.fusion_influence
                            for f, c in zip(fusion_list, current_list)
                        ]
                        # Normalize
                        norm = sum(x * x for x in modified_fusion) ** 0.5
                        if norm > 0:
                            modified_fusion = [x / norm for x in modified_fusion]
                
                if attention_t is not None and hasattr(attention_t, '__iter__'):
                    att_list = list(attention_t) if not isinstance(attention_t, list) else attention_t
                    if len(att_list) == len(current_list):
                        modified_attention = [
                            a * (1.0 - self.attention_influence) + c * self.attention_influence
                            for a, c in zip(att_list, current_list)
                        ]
                        # Normalize
                        norm = sum(x * x for x in modified_attention) ** 0.5
                        if norm > 0:
                            modified_attention = [x / norm for x in modified_attention]
            
            return {
                "chain_length": len(chain),
                "execution_order": [
                    {
                        "id": node["id"],
                        "priority": round(node["priority"], 4),
                        "dependency": round(node["dependency"], 4)
                    }
                    for node in chain
                ],
                "current_step": chain[0]["id"] if chain else None,
                "modified_fusion": modified_fusion,
                "modified_attention": modified_attention,
                "plan_valid": True
            }
    
    def get_plan_buffer(self):
        """Get current plan buffer state."""
        return self.plan_buffer.copy()
    
    def summary(self, planning_state):
        """
        Get summary of planning state for logging.
        
        Args:
            planning_state: Dict returned from plan()
            
        Returns:
            Dict with key metrics
        """
        if planning_state is None:
            return {
                "note": "Insufficient subgoals for sequential planning",
                "existing_chain": self.plan_buffer.get("chain", [])
            }
        
        return {
            "chain_length": planning_state.get("chain_length", 0),
            "execution_order": planning_state.get("execution_order", []),
            "current_step": planning_state.get("current_step"),
            "plan_valid": planning_state.get("plan_valid", False)
        }

