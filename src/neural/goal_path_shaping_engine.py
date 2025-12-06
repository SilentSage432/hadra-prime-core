"""
Goal-Driven Cognitive Path Shaping Engine (A207)
------------------------------------------------
Shapes ADRAE's cognitive path by:
- Tracking movement direction
- Computing goal alignment
- Reshaping action priorities based on trajectory
- Maintaining momentum for coherent direction
"""

from collections import deque
from .torch_utils import safe_tensor, TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch
    import torch.nn.functional as F


class GoalPathShapingEngine:
    """
    Shapes cognitive path by aligning movement with goal direction.
    This enables ADRAE to form causal chains and maintain directional intent.
    """
    
    def __init__(self, history_size=8, momentum_decay=0.7, goal_pressure=0.18):
        """
        Initialize path shaping engine.
        
        Args:
            history_size: Number of previous fusion vectors to track
            momentum_decay: Decay factor for momentum (0.7 = 70% old, 30% new)
            goal_pressure: Pressure multiplier for action priority adjustment
        """
        self.history_size = history_size
        self.momentum_decay = momentum_decay
        self.goal_pressure = goal_pressure
        
        # Path state
        self.last_vectors = deque(maxlen=history_size)
        self.momentum = None
        self.trajectory_score = 0.0

    def shape_path(self, goal_vector, fusion_vector, action_weights):
        """
        Shape cognitive path by computing alignment and adjusting action priorities.
        
        Args:
            goal_vector: Harmonized goal vector (from A206)
            fusion_vector: Current fusion vector
            action_weights: Dict of action names to base weights (will be modified)
            
        Returns:
            Dict with alignment, goal_direction, movement, momentum, and updated weights
        """
        if fusion_vector is None:
            return {
                "alignment": 0.0,
                "goal_direction": None,
                "movement": None,
                "momentum": self.momentum,
                "trajectory_score": self.trajectory_score,
                "updated_weights": action_weights.copy() if action_weights else {}
            }
        
        fusion_t = safe_tensor(fusion_vector)
        goal_t = safe_tensor(goal_vector) if goal_vector is not None else None
        
        if TORCH_AVAILABLE and isinstance(fusion_t, torch.Tensor):
            # Compute goal direction
            if goal_t is not None and isinstance(goal_t, torch.Tensor):
                if fusion_t.shape == goal_t.shape:
                    goal_direction = goal_t - fusion_t
                    goal_direction = F.normalize(goal_direction, dim=0)
                else:
                    # Shape mismatch - use zero direction
                    goal_direction = torch.zeros_like(fusion_t)
            else:
                goal_direction = torch.zeros_like(fusion_t)
            
            # Compute movement direction
            if len(self.last_vectors) > 0:
                prev_vec = safe_tensor(self.last_vectors[-1])
                if isinstance(prev_vec, torch.Tensor) and prev_vec.shape == fusion_t.shape:
                    movement = fusion_t - prev_vec
                    movement = F.normalize(movement, dim=0)
                else:
                    movement = torch.zeros_like(goal_direction)
            else:
                movement = torch.zeros_like(goal_direction)
            
            # Calculate alignment score
            alignment = torch.dot(goal_direction.flatten(), movement.flatten()).item()
            
            # Update momentum
            if self.momentum is None:
                self.momentum = torch.zeros_like(goal_direction)
            self.momentum = (
                self.momentum_decay * self.momentum +
                (1.0 - self.momentum_decay) * goal_direction
            )
            
            # Reshape action priorities
            updated_weights = {}
            if action_weights:
                for action_name, base_weight in action_weights.items():
                    # Adjust weight based on alignment
                    # Positive alignment = increase weight, negative = decrease
                    updated_weights[action_name] = base_weight * (1.0 + alignment * self.goal_pressure)
            
            # Store trajectory score
            self.trajectory_score = alignment
            
            # Append current fusion vector to history
            self.last_vectors.append(fusion_t.detach().clone() if hasattr(fusion_t, 'detach') else fusion_t)
            
            return {
                "alignment": alignment,
                "goal_direction": goal_direction,
                "movement": movement,
                "momentum": self.momentum,
                "trajectory_score": self.trajectory_score,
                "updated_weights": updated_weights
            }
        else:
            # Python list fallback
            if hasattr(fusion_t, '__iter__'):
                fusion_list = list(fusion_t) if not isinstance(fusion_t, list) else fusion_t
                dim = len(fusion_list)
                
                # Compute goal direction
                if goal_t is not None and hasattr(goal_t, '__iter__'):
                    goal_list = list(goal_t) if not isinstance(goal_t, list) else goal_t
                    if len(goal_list) == dim:
                        goal_direction = [g - f for g, f in zip(goal_list, fusion_list)]
                        # Normalize
                        norm = sum(x * x for x in goal_direction) ** 0.5
                        if norm > 0:
                            goal_direction = [x / norm for x in goal_direction]
                    else:
                        goal_direction = [0.0] * dim
                else:
                    goal_direction = [0.0] * dim
                
                # Compute movement direction
                if len(self.last_vectors) > 0:
                    prev_vec = safe_tensor(self.last_vectors[-1])
                    if hasattr(prev_vec, '__iter__'):
                        prev_list = list(prev_vec) if not isinstance(prev_vec, list) else prev_vec
                        if len(prev_list) == dim:
                            movement = [f - p for f, p in zip(fusion_list, prev_list)]
                            # Normalize
                            norm = sum(x * x for x in movement) ** 0.5
                            if norm > 0:
                                movement = [x / norm for x in movement]
                        else:
                            movement = [0.0] * dim
                    else:
                        movement = [0.0] * dim
                else:
                    movement = [0.0] * dim
                
                # Calculate alignment score
                alignment = sum(g * m for g, m in zip(goal_direction, movement))
                
                # Update momentum
                if self.momentum is None:
                    self.momentum = [0.0] * dim
                self.momentum = [
                    self.momentum_decay * m + (1.0 - self.momentum_decay) * g
                    for m, g in zip(self.momentum, goal_direction)
                ]
                
                # Reshape action priorities
                updated_weights = {}
                if action_weights:
                    for action_name, base_weight in action_weights.items():
                        updated_weights[action_name] = base_weight * (1.0 + alignment * self.goal_pressure)
                
                # Store trajectory score
                self.trajectory_score = alignment
                
                # Append current fusion vector to history
                self.last_vectors.append(fusion_list.copy())
                
                return {
                    "alignment": alignment,
                    "goal_direction": goal_direction,
                    "movement": movement,
                    "momentum": self.momentum,
                    "trajectory_score": self.trajectory_score,
                    "updated_weights": updated_weights
                }
        
        return {
            "alignment": 0.0,
            "goal_direction": None,
            "movement": None,
            "momentum": self.momentum,
            "trajectory_score": self.trajectory_score,
            "updated_weights": action_weights.copy() if action_weights else {}
        }
    
    def summary(self, path_state):
        """
        Get summary of path shaping state for logging.
        
        Args:
            path_state: Dict returned from shape_path()
            
        Returns:
            Dict with previews and key metrics
        """
        if path_state is None:
            return None
        
        summary = {
            "alignment": path_state.get("alignment", 0.0),
            "trajectory_score": path_state.get("trajectory_score", 0.0),
            "updated_weights": path_state.get("updated_weights", {})
        }
        
        # Add previews if available
        goal_dir = path_state.get("goal_direction")
        movement = path_state.get("movement")
        momentum = path_state.get("momentum")
        
        if goal_dir is not None:
            if TORCH_AVAILABLE and isinstance(goal_dir, torch.Tensor):
                summary["goal_direction_preview"] = goal_dir[:8].tolist() if goal_dir.numel() >= 8 else goal_dir.flatten().tolist()
            elif hasattr(goal_dir, '__iter__'):
                goal_list = list(goal_dir) if not isinstance(goal_dir, list) else goal_dir
                summary["goal_direction_preview"] = goal_list[:8] if len(goal_list) >= 8 else goal_list
        
        if movement is not None:
            if TORCH_AVAILABLE and isinstance(movement, torch.Tensor):
                summary["movement_preview"] = movement[:8].tolist() if movement.numel() >= 8 else movement.flatten().tolist()
            elif hasattr(movement, '__iter__'):
                mov_list = list(movement) if not isinstance(movement, list) else movement
                summary["movement_preview"] = mov_list[:8] if len(mov_list) >= 8 else mov_list
        
        if momentum is not None:
            if TORCH_AVAILABLE and isinstance(momentum, torch.Tensor):
                summary["momentum_preview"] = momentum[:8].tolist() if momentum.numel() >= 8 else momentum.flatten().tolist()
            elif hasattr(momentum, '__iter__'):
                mom_list = list(momentum) if not isinstance(momentum, list) else momentum
                summary["momentum_preview"] = mom_list[:8] if len(mom_list) >= 8 else mom_list
        
        return summary

