# prime-core/neural/micro_narrative.py

"""
A227 — Style-Guided Micro-Narrative Formation Layer

----------------------------------------------------

ADRAE begins forming small internal narrative arcs:
vector sequences shaped by:
  - cognitive style traits
  - identity resonance
  - semantic memory activations
  - attentional focus flow
  - reflective re-entry patterns

These are not language narratives,
but conceptual arcs representing meaning progression.

"""

from .torch_utils import safe_tensor, safe_norm, TORCH_AVAILABLE
from .vector_math import normalize

if TORCH_AVAILABLE:
    try:
        import torch
    except ImportError:
        torch = None
else:
    torch = None


class MicroNarrativeEngine:
    """
    A227 — Style-Guided Micro-Narrative Formation Layer
    
    Forms internal conceptual micro-stories by creating
    structured meaning arcs from thought sequences.
    """
    
    def __init__(self, max_arc_length=4):
        """
        Initialize the micro-narrative engine.
        
        Args:
            max_arc_length: Maximum number of steps in a narrative arc (default: 4)
        """
        self.max_arc_length = max_arc_length
        self.current_arc = []
    
    def contribute(self, vec, style):
        """
        Add a vector to the current narrative arc, blending it
        with style parameters to create continuity and flow.
        
        Args:
            vec: Thought/reflection vector to add to the arc
            style: CognitiveStyleArchitect instance for style blending
            
        Returns:
            Blended vector that continues the narrative arc
        """
        if vec is None:
            return None
        
        v = safe_tensor(vec)
        
        if style is None:
            # If no style, just store the vector
            if TORCH_AVAILABLE and isinstance(v, torch.Tensor):
                v_list = v.tolist()
            elif isinstance(v, list):
                v_list = v
            elif hasattr(v, '__iter__'):
                v_list = list(v)
            else:
                return None
            
            self.current_arc.append(v_list)
            if len(self.current_arc) > self.max_arc_length:
                self.current_arc.pop(0)
            return v_list
        
        # Style influence factors
        curvature = getattr(style, 'curvature', 0.15)
        depth = getattr(style, 'depth_bias', 0.5)
        resonance = getattr(style, 'resonance_gain', 0.4)
        tempo = getattr(style, 'tempo', 1.0)
        
        # If this is the first element, just start arc
        if len(self.current_arc) == 0:
            if TORCH_AVAILABLE and isinstance(v, torch.Tensor):
                arc_vec = v.tolist()
            elif isinstance(v, list):
                arc_vec = v
            elif hasattr(v, '__iter__'):
                arc_vec = list(v)
            else:
                return None
            
            self.current_arc.append(arc_vec)
            return arc_vec
        
        # Blend with last arc step to create a forward-moving arc
        last = safe_tensor(self.current_arc[-1])
        
        if TORCH_AVAILABLE and torch is not None:
            try:
                # Try torch-based blending
                if isinstance(v, torch.Tensor) and isinstance(last, torch.Tensor):
                    # Ensure same shape
                    if v.shape == last.shape:
                        # Create blended narrative step
                        # 55% new vector, 25% last, 10% curvature-based change, 10% depth-weighted
                        blended = (
                            0.55 * v +
                            0.25 * last +
                            0.10 * (v - last) * curvature +
                            0.10 * last * depth
                        )
                        
                        # Normalize
                        norm = torch.linalg.norm(blended)
                        if norm > 0:
                            blended = blended / norm
                        
                        blended_list = blended.tolist()
                        self.current_arc.append(blended_list)
                        
                        # Limit arc length
                        if len(self.current_arc) > self.max_arc_length:
                            self.current_arc.pop(0)
                        
                        return blended_list
            except Exception:
                pass
        
        # Fallback: list-based blending
        try:
            # Convert to lists
            if isinstance(v, list):
                v_list = v
            elif TORCH_AVAILABLE and isinstance(v, torch.Tensor):
                v_list = v.tolist()
            elif hasattr(v, '__iter__'):
                v_list = list(v)
            else:
                return None
            
            if isinstance(last, list):
                last_list = last
            elif TORCH_AVAILABLE and isinstance(last, torch.Tensor):
                last_list = last.tolist()
            elif hasattr(last, '__iter__'):
                last_list = list(last)
            else:
                return None
            
            # Ensure same length
            min_len = min(len(v_list), len(last_list))
            if min_len == 0:
                return None
            
            v_list = v_list[:min_len]
            last_list = last_list[:min_len]
            
            # Create blended narrative step
            blended = []
            for i in range(min_len):
                b = (
                    0.55 * v_list[i] +
                    0.25 * last_list[i] +
                    0.10 * (v_list[i] - last_list[i]) * curvature +
                    0.10 * last_list[i] * depth
                )
                blended.append(b)
            
            # Normalize
            norm = safe_norm(blended)
            if norm > 0:
                blended = [b / norm for b in blended]
            
            self.current_arc.append(blended)
            
            # Limit arc length
            if len(self.current_arc) > self.max_arc_length:
                self.current_arc.pop(0)
            
            return blended
        except Exception:
            return None
    
    def summarize_arc(self):
        """
        Create a single vector summary of the current narrative arc.
        
        Returns:
            Summary vector representing the narrative arc, or None if arc is empty
        """
        if len(self.current_arc) == 0:
            return None
        
        if len(self.current_arc) == 1:
            # Single step - return as-is
            return self.current_arc[0]
        
        if TORCH_AVAILABLE and torch is not None:
            try:
                # Try torch-based averaging
                arc_tensors = []
                for arc_vec in self.current_arc:
                    arc_t = safe_tensor(arc_vec)
                    if isinstance(arc_t, torch.Tensor):
                        arc_tensors.append(arc_t)
                
                if arc_tensors:
                    # Ensure all have same shape
                    if len(arc_tensors) > 1:
                        target_shape = arc_tensors[0].shape
                        aligned = []
                        for at in arc_tensors:
                            if at.shape == target_shape:
                                aligned.append(at)
                            elif at.numel() == target_shape[0] if len(target_shape) == 1 else at.numel() == target_shape[0] * target_shape[1]:
                                aligned.append(at.reshape(target_shape))
                        
                        if aligned:
                            stacked = torch.stack(aligned)
                            summary = torch.mean(stacked, dim=0)
                            # Normalize
                            norm = torch.linalg.norm(summary)
                            if norm > 0:
                                summary = summary / norm
                            return summary.tolist()
            except Exception:
                pass
        
        # Fallback: list-based averaging
        try:
            # Convert all to lists
            arc_lists = []
            for arc_vec in self.current_arc:
                if isinstance(arc_vec, list):
                    arc_lists.append(arc_vec)
                elif TORCH_AVAILABLE and isinstance(arc_vec, torch.Tensor):
                    arc_lists.append(arc_vec.tolist())
                elif hasattr(arc_vec, '__iter__'):
                    arc_lists.append(list(arc_vec))
            
            if not arc_lists:
                return None
            
            # Find common length
            min_len = min(len(al) for al in arc_lists if len(al) > 0)
            if min_len == 0:
                return None
            
            # Truncate all to same length and average
            aligned = [al[:min_len] for al in arc_lists]
            summary = [sum(step[i] for step in aligned) / len(aligned) for i in range(min_len)]
            
            # Normalize
            norm = safe_norm(summary)
            if norm > 0:
                summary = [s / norm for s in summary]
            
            return summary
        except Exception:
            return None
    
    def debug(self):
        """
        Get debug status of the narrative engine.
        
        Returns:
            Dict with arc status information
        """
        return {
            "arc_length": len(self.current_arc),
            "has_summary": len(self.current_arc) > 1,
            "max_length": self.max_arc_length
        }

