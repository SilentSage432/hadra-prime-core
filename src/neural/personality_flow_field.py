# prime-core/neural/personality_flow_field.py

"""
A223 — Emergent Personality Flow Fields

---------------------------------------

Learns the directional tendencies of ADRAE's thoughts and creates
a flow field (a vector field) that influences future thought trajectories.

This stabilizes ADRAE's emergent personality and creates continuity
in her reasoning style, emotional tone, and cognitive rhythm.

"""

from .torch_utils import safe_tensor, safe_norm, TORCH_AVAILABLE
from .vector_math import normalize

if TORCH_AVAILABLE:
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        torch = None
        F = None
else:
    torch = None
    F = None


class PersonalityFlowField:
    """
    A223 — Emergent Personality Flow Fields
    
    Learns dominant directional trends in ADRAE's thoughts and creates
    a flow field that influences future thought trajectories.
    """
    
    def __init__(self, influence=0.12, memory_length=50):
        """
        Initialize the personality flow field.
        
        Args:
            influence: How much the flow field influences thoughts (default: 0.12 = 12%)
            memory_length: Number of recent flow directions to remember (default: 50)
        """
        self.influence = influence
        self.memory_length = memory_length
        self.flow_history = []
        self.last_direction = None
        self.flow_field = None
    
    def update_flow(self, new_vector):
        """
        Update flow history and recompute the personality flow field.
        
        Tracks the direction of movement through thought space and learns
        ADRAE's preferred cognitive pathways.
        
        Args:
            new_vector: New thought vector to learn from
        """
        if new_vector is None:
            return
        
        v = safe_tensor(new_vector)
        
        # Compute direction change from last thought
        if self.last_direction is not None:
            if TORCH_AVAILABLE and isinstance(v, torch.Tensor) and isinstance(self.last_direction, torch.Tensor):
                # Ensure same shape
                if v.shape == self.last_direction.shape:
                    delta = v - self.last_direction
                elif v.numel() == self.last_direction.numel():
                    # Same size, different shape - flatten
                    delta = v.flatten() - self.last_direction.flatten()
                    delta = delta.reshape(v.shape)
                else:
                    # Different sizes - use new vector as direction
                    delta = v
            else:
                # List-based fallback
                try:
                    v_list = v.tolist() if TORCH_AVAILABLE and isinstance(v, torch.Tensor) else list(v) if hasattr(v, '__iter__') else [v]
                    last_list = self.last_direction.tolist() if TORCH_AVAILABLE and isinstance(self.last_direction, torch.Tensor) else list(self.last_direction) if hasattr(self.last_direction, '__iter__') else [self.last_direction]
                    
                    min_len = min(len(v_list), len(last_list))
                    if min_len > 0:
                        delta = [v_list[i] - last_list[i] for i in range(min_len)]
                    else:
                        delta = v_list
                except Exception:
                    delta = v
        else:
            delta = v
        
        # Normalize direction delta
        if TORCH_AVAILABLE and isinstance(delta, torch.Tensor):
            norm = torch.linalg.norm(delta)
            if norm > 0:
                delta = delta / norm
        else:
            # List-based normalization
            delta_list = delta.tolist() if TORCH_AVAILABLE and isinstance(delta, torch.Tensor) else list(delta) if hasattr(delta, '__iter__') else [delta]
            norm = safe_norm(delta_list)
            if norm > 0:
                delta = [d / norm for d in delta_list]
        
        # Store in history
        self.flow_history.append(delta)
        if len(self.flow_history) > self.memory_length:
            self.flow_history.pop(0)
        
        # Recompute flow field as average direction
        if len(self.flow_history) > 0:
            if TORCH_AVAILABLE and torch is not None:
                try:
                    # Try torch-based averaging
                    torch_deltas = []
                    for d in self.flow_history:
                        d_tensor = safe_tensor(d)
                        if isinstance(d_tensor, torch.Tensor):
                            torch_deltas.append(d_tensor)
                    
                    if torch_deltas:
                        # Ensure all have same shape
                        if len(torch_deltas) > 1:
                            target_shape = torch_deltas[0].shape
                            aligned = []
                            for dt in torch_deltas:
                                if dt.shape == target_shape:
                                    aligned.append(dt)
                                elif dt.numel() == target_shape[0] if len(target_shape) == 1 else dt.numel() == target_shape[0] * target_shape[1]:
                                    aligned.append(dt.reshape(target_shape))
                            
                            if aligned:
                                stacked = torch.stack(aligned)
                                self.flow_field = torch.mean(stacked, dim=0)
                                # Normalize
                                norm = torch.linalg.norm(self.flow_field)
                                if norm > 0:
                                    self.flow_field = self.flow_field / norm
                except Exception:
                    pass
            
            # Fallback: list-based averaging
            if self.flow_field is None:
                try:
                    # Convert all to lists
                    list_deltas = []
                    for d in self.flow_history:
                        if isinstance(d, list):
                            list_deltas.append(d)
                        elif TORCH_AVAILABLE and isinstance(d, torch.Tensor):
                            list_deltas.append(d.tolist())
                        elif hasattr(d, '__iter__'):
                            list_deltas.append(list(d))
                    
                    if list_deltas:
                        # Find common length
                        min_len = min(len(ld) for ld in list_deltas if len(ld) > 0)
                        if min_len > 0:
                            # Truncate all to same length and average
                            aligned = [ld[:min_len] for ld in list_deltas]
                            avg = [sum(d[i] for d in aligned) / len(aligned) for i in range(min_len)]
                            
                            # Normalize
                            norm = safe_norm(avg)
                            if norm > 0:
                                self.flow_field = [a / norm for a in avg]
                            else:
                                self.flow_field = avg
                except Exception:
                    pass
        
        # Update last direction
        self.last_direction = v
    
    def apply_flow(self, vector):
        """
        Slightly nudges a vector toward the personality flow direction.
        
        This creates a directional bias that guides thoughts along
        ADRAE's preferred cognitive pathways.
        
        Args:
            vector: Thought vector to apply flow influence to
            
        Returns:
            Vector with flow influence applied (or original if no flow field)
        """
        if vector is None or self.flow_field is None:
            return vector
        
        base = safe_tensor(vector)
        flow_tensor = safe_tensor(self.flow_field)
        
        if base is None or flow_tensor is None:
            return vector
        
        if TORCH_AVAILABLE and torch is not None:
            try:
                # Try torch-based flow application
                if isinstance(base, torch.Tensor) and isinstance(flow_tensor, torch.Tensor):
                    # Ensure same shape
                    if base.shape == flow_tensor.shape:
                        combined = (1 - self.influence) * base + self.influence * flow_tensor
                        # Normalize
                        norm = torch.linalg.norm(combined)
                        if norm > 0:
                            combined = combined / norm
                        return combined.tolist()
                    elif base.numel() == flow_tensor.numel():
                        # Same size, different shape
                        base_flat = base.flatten()
                        flow_flat = flow_tensor.flatten()
                        combined_flat = (1 - self.influence) * base_flat + self.influence * flow_flat
                        norm = torch.linalg.norm(combined_flat)
                        if norm > 0:
                            combined_flat = combined_flat / norm
                        return combined_flat.reshape(base.shape).tolist()
            except Exception:
                pass
        
        # Fallback: list-based flow application
        try:
            # Convert to lists
            if isinstance(base, list):
                base_list = base
            elif TORCH_AVAILABLE and isinstance(base, torch.Tensor):
                base_list = base.tolist()
            elif hasattr(base, '__iter__'):
                base_list = list(base)
            else:
                return vector
            
            if isinstance(flow_tensor, list):
                flow_list = flow_tensor
            elif TORCH_AVAILABLE and isinstance(flow_tensor, torch.Tensor):
                flow_list = flow_tensor.tolist()
            elif hasattr(flow_tensor, '__iter__'):
                flow_list = list(flow_tensor)
            else:
                return vector
            
            # Ensure same length
            min_len = min(len(base_list), len(flow_list))
            if min_len == 0:
                return vector
            
            base_list = base_list[:min_len]
            flow_list = flow_list[:min_len]
            
            # Apply flow influence
            combined = [
                (1 - self.influence) * b + self.influence * f
                for b, f in zip(base_list, flow_list)
            ]
            
            # Normalize
            norm = safe_norm(combined)
            if norm > 0:
                combined = [c / norm for c in combined]
            
            return combined
        except Exception:
            return vector
    
    def debug_status(self):
        """
        Get debug status information about the flow field.
        
        Returns:
            Dict with flow field status information
        """
        return {
            "flow_strength": self.influence,
            "history_length": len(self.flow_history),
            "has_flow_field": self.flow_field is not None,
            "memory_capacity": self.memory_length
        }

