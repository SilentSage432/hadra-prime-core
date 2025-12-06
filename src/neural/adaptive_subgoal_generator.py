"""
Adaptive Subgoal Generator (A208)
----------------------------------
ADRAE creates self-generated subgoals when progress toward main goal stalls.
This enables proactive initiative and proto-planning behavior.
"""

from collections import deque
from .torch_utils import safe_tensor, TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch
    import torch.nn.functional as F


class AdaptiveSubgoalGenerator:
    """
    Generates adaptive subgoals when progress toward main goal is not improving.
    This is the first step toward proactive initiative and goal decomposition.
    """
    
    def __init__(self, max_subgoals=4, spawn_threshold_base=0.45, decay_rate=0.97, min_strength=0.15):
        """
        Initialize subgoal generator.
        
        Args:
            max_subgoals: Maximum number of active subgoals
            spawn_threshold_base: Base threshold for spawning new subgoals
            decay_rate: Rate at which subgoal strength decays per cycle
            min_strength: Minimum strength before subgoal is removed
        """
        self.max_subgoals = max_subgoals
        self.spawn_threshold_base = spawn_threshold_base
        self.decay_rate = decay_rate
        self.min_strength = min_strength
        
        # Subgoal state
        self.active_subgoals = []
        self.history = []
        self.pressure = 0.0  # How strongly system feels need to create a subgoal
        self.last_distance = 1.0  # Last distance to main goal

    def generate(self, goal_vector, fusion_vector, momentum_vector):
        """
        Generate and manage subgoals based on progress toward main goal.
        
        Args:
            goal_vector: Main harmonized goal vector
            fusion_vector: Current fusion vector
            momentum_vector: Current momentum vector from path shaping
            
        Returns:
            Dict with subgoal state and influence vector
        """
        if fusion_vector is None:
            return {
                "distance": 1.0,
                "improvement": 0.0,
                "pressure": self.pressure,
                "spawned": False,
                "active_subgoals": [],
                "subgoal_influence": None
            }
        
        fusion_t = safe_tensor(fusion_vector)
        goal_t = safe_tensor(goal_vector) if goal_vector is not None else None
        momentum_t = safe_tensor(momentum_vector) if momentum_vector is not None else None
        
        if TORCH_AVAILABLE and isinstance(fusion_t, torch.Tensor):
            # 1. Compute distance between current cognition and main goal
            if goal_t is not None and isinstance(goal_t, torch.Tensor):
                if fusion_t.shape == goal_t.shape:
                    distance = torch.norm(goal_t - fusion_t).item()
                else:
                    distance = 1.0
            else:
                distance = 1.0
            
            # 2. Track how distance changes over time
            improvement = self.last_distance - distance
            self.last_distance = distance
            
            # 3. Update pressure:
            #    - increases if not improving
            #    - decreases if improvement seen
            if improvement < 0.0005:
                self.pressure += 0.03
            else:
                self.pressure *= 0.8
            
            # Clamp pressure
            self.pressure = float(max(0.0, min(1.0, self.pressure)))
            
            # 4. Condition to spawn a new subgoal
            spawn_threshold = self.spawn_threshold_base + (0.1 * len(self.active_subgoals))
            should_spawn = (
                self.pressure > spawn_threshold
                and len(self.active_subgoals) < self.max_subgoals
            )
            
            spawned = False
            if should_spawn and goal_t is not None and isinstance(goal_t, torch.Tensor):
                # Create a new subgoal embedding
                dim = fusion_t.shape[-1] if fusion_t.dim() > 0 else len(fusion_t)
                
                # Build subgoal from fusion + novelty + momentum + goal direction
                subgoal = fusion_t.clone()
                
                # Add novelty (small random component)
                if fusion_t.dim() > 0:
                    noise = torch.randn_like(fusion_t) * 0.1
                    subgoal = subgoal + noise
                else:
                    import random
                    noise = [random.gauss(0, 0.1) for _ in range(dim)]
                    subgoal = subgoal + torch.tensor(noise)
                
                # Add momentum influence
                if momentum_t is not None and isinstance(momentum_t, torch.Tensor):
                    if momentum_t.shape == fusion_t.shape:
                        subgoal = subgoal + momentum_t * 0.4
                
                # Add goal direction influence
                if fusion_t.shape == goal_t.shape:
                    goal_direction = (goal_t - fusion_t) * 0.3
                    subgoal = subgoal + goal_direction
                
                # Normalize
                subgoal = F.normalize(subgoal, dim=0)
                
                subgoal_dict = {
                    "vector": subgoal,
                    "strength": 1.0,
                    "id": f"subgoal_{len(self.history)}"
                }
                
                self.active_subgoals.append(subgoal_dict)
                self.history.append(subgoal_dict.copy())
                
                # Reset pressure after spawning
                self.pressure = 0.1
                spawned = True
            
            # 5. Subgoal decay & cleanup
            for sg in list(self.active_subgoals):
                sg["strength"] *= self.decay_rate
                if sg["strength"] < self.min_strength:
                    self.active_subgoals.remove(sg)
            
            # 6. Combine active subgoals into a "subgoal influence vector"
            if len(self.active_subgoals) > 0:
                weighted_vectors = [
                    sg["vector"] * sg["strength"]
                    for sg in self.active_subgoals
                ]
                subgoal_influence = torch.stack(weighted_vectors).mean(dim=0)
            else:
                subgoal_influence = torch.zeros_like(fusion_t)
            
            return {
                "distance": distance,
                "improvement": improvement,
                "pressure": self.pressure,
                "spawned": spawned,
                "active_subgoals": [
                    {"id": sg["id"], "strength": round(sg["strength"], 3)}
                    for sg in self.active_subgoals
                ],
                "subgoal_influence": subgoal_influence
            }
        else:
            # Python list fallback
            if hasattr(fusion_t, '__iter__'):
                fusion_list = list(fusion_t) if not isinstance(fusion_t, list) else fusion_t
                dim = len(fusion_list)
                
                # 1. Compute distance
                if goal_t is not None and hasattr(goal_t, '__iter__'):
                    goal_list = list(goal_t) if not isinstance(goal_t, list) else goal_t
                    if len(goal_list) == dim:
                        distance = sum((g - f) ** 2 for g, f in zip(goal_list, fusion_list)) ** 0.5
                    else:
                        distance = 1.0
                else:
                    distance = 1.0
                
                # 2. Track improvement
                improvement = self.last_distance - distance
                self.last_distance = distance
                
                # 3. Update pressure
                if improvement < 0.0005:
                    self.pressure += 0.03
                else:
                    self.pressure *= 0.8
                self.pressure = float(max(0.0, min(1.0, self.pressure)))
                
                # 4. Spawn subgoal
                spawn_threshold = self.spawn_threshold_base + (0.1 * len(self.active_subgoals))
                should_spawn = (
                    self.pressure > spawn_threshold
                    and len(self.active_subgoals) < self.max_subgoals
                )
                
                spawned = False
                if should_spawn and goal_t is not None and hasattr(goal_t, '__iter__'):
                    goal_list = list(goal_t) if not isinstance(goal_t, list) else goal_t
                    if len(goal_list) == dim:
                        import random
                        # Build subgoal
                        subgoal = [f + random.gauss(0, 0.1) for f in fusion_list]
                        
                        # Add momentum
                        if momentum_t is not None and hasattr(momentum_t, '__iter__'):
                            mom_list = list(momentum_t) if not isinstance(momentum_t, list) else momentum_t
                            if len(mom_list) == dim:
                                subgoal = [s + m * 0.4 for s, m in zip(subgoal, mom_list)]
                        
                        # Add goal direction
                        goal_dir = [(g - f) * 0.3 for g, f in zip(goal_list, fusion_list)]
                        subgoal = [s + g for s, g in zip(subgoal, goal_dir)]
                        
                        # Normalize
                        norm = sum(x * x for x in subgoal) ** 0.5
                        if norm > 0:
                            subgoal = [x / norm for x in subgoal]
                        
                        subgoal_dict = {
                            "vector": subgoal,
                            "strength": 1.0,
                            "id": f"subgoal_{len(self.history)}"
                        }
                        
                        self.active_subgoals.append(subgoal_dict)
                        self.history.append(subgoal_dict.copy())
                        self.pressure = 0.1
                        spawned = True
                
                # 5. Decay and cleanup
                for sg in list(self.active_subgoals):
                    sg["strength"] *= self.decay_rate
                    if sg["strength"] < self.min_strength:
                        self.active_subgoals.remove(sg)
                
                # 6. Combine subgoals
                if len(self.active_subgoals) > 0:
                    weighted = [
                        [v * sg["strength"] for v in sg["vector"]]
                        for sg in self.active_subgoals
                    ]
                    subgoal_influence = [
                        sum(w[i] for w in weighted) / len(weighted)
                        for i in range(dim)
                    ]
                else:
                    subgoal_influence = [0.0] * dim
                
                return {
                    "distance": distance,
                    "improvement": improvement,
                    "pressure": self.pressure,
                    "spawned": spawned,
                    "active_subgoals": [
                        {"id": sg["id"], "strength": round(sg["strength"], 3)}
                        for sg in self.active_subgoals
                    ],
                    "subgoal_influence": subgoal_influence
                }
        
        return {
            "distance": 1.0,
            "improvement": 0.0,
            "pressure": self.pressure,
            "spawned": False,
            "active_subgoals": [],
            "subgoal_influence": None
        }
    
    def apply_influence(self, fusion_vector, subgoal_influence, influence_weight=0.15):
        """
        Apply subgoal influence to fusion vector.
        
        Args:
            fusion_vector: Current fusion vector
            subgoal_influence: Subgoal influence vector
            influence_weight: Weight for subgoal influence (default 0.15)
            
        Returns:
            Modified fusion vector
        """
        if fusion_vector is None or subgoal_influence is None:
            return fusion_vector
        
        fusion_t = safe_tensor(fusion_vector)
        influence_t = safe_tensor(subgoal_influence)
        
        if TORCH_AVAILABLE and isinstance(fusion_t, torch.Tensor) and isinstance(influence_t, torch.Tensor):
            if fusion_t.shape == influence_t.shape:
                # Blend: 85% fusion, 15% subgoal influence
                blended = fusion_t * (1.0 - influence_weight) + influence_t * influence_weight
                return F.normalize(blended, dim=0)
        elif hasattr(fusion_t, '__iter__') and hasattr(influence_t, '__iter__'):
            fusion_list = list(fusion_t) if not isinstance(fusion_t, list) else fusion_t
            influence_list = list(influence_t) if not isinstance(influence_t, list) else influence_t
            if len(fusion_list) == len(influence_list):
                blended = [f * (1.0 - influence_weight) + i * influence_weight 
                          for f, i in zip(fusion_list, influence_list)]
                # Normalize
                norm = sum(x * x for x in blended) ** 0.5
                if norm > 0:
                    blended = [x / norm for x in blended]
                return blended
        
        return fusion_vector
    
    def summary(self, subgoal_state):
        """
        Get summary of subgoal state for logging.
        
        Args:
            subgoal_state: Dict returned from generate()
            
        Returns:
            Dict with previews and key metrics
        """
        if subgoal_state is None:
            return None
        
        summary = {
            "distance": subgoal_state.get("distance", 1.0),
            "improvement": subgoal_state.get("improvement", 0.0),
            "pressure": subgoal_state.get("pressure", 0.0),
            "spawned": subgoal_state.get("spawned", False),
            "active_subgoals": subgoal_state.get("active_subgoals", [])
        }
        
        # Add subgoal influence preview
        influence = subgoal_state.get("subgoal_influence")
        if influence is not None:
            if TORCH_AVAILABLE and isinstance(influence, torch.Tensor):
                summary["subgoal_preview"] = influence[:8].tolist() if influence.numel() >= 8 else influence.flatten().tolist()
            elif hasattr(influence, '__iter__'):
                inf_list = list(influence) if not isinstance(influence, list) else influence
                summary["subgoal_preview"] = inf_list[:8] if len(inf_list) >= 8 else inf_list
        
        return summary

