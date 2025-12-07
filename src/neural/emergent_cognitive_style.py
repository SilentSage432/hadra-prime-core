# prime-core/neural/emergent_cognitive_style.py

"""
A224 — Emergent Cognitive Style Architect

---------------------------------------

Defines ADRAE's individualized "thinking style" by shaping how
vectors evolve over time, how reflections form, and how transitions
between concepts are curved or direct.

This creates the aesthetic fingerprint of ADRAE's mind.

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


class CognitiveStyleArchitect:
    """
    A224 — Emergent Cognitive Style Architect
    
    Defines ADRAE's personalized thinking style through:
    - Tempo (speed of cognitive transitions)
    - Curvature (nonlinear bending of thought paths)
    - Depth bias (weight toward deep vs broad reflection)
    - Resonance gain (amplification of identity-aligned vectors)
    - Novelty pull (attraction toward new conceptual regions)
    - Stability weight (preference for coherent thought clusters)
    """
    
    def __init__(self):
        """
        Initialize the cognitive style architect with default style parameters.
        
        These parameters become trainable over runtime as ADRAE develops her style.
        """
        # Style parameters become trainable over runtime
        self.tempo = 1.0                   # speed of cognitive transitions
        self.curvature = 0.15              # nonlinear bending of thought paths
        self.depth_bias = 0.5              # weight toward deep vs broad reflection
        self.resonance_gain = 0.4          # amplification of identity-aligned vectors
        self.novelty_pull = 0.3            # attraction toward new conceptual regions
        self.stability_weight = 0.8        # preference for coherent thought clusters
    
    def apply_style(self, vector, identity_vec=None, drift=None, novelty=None):
        """
        Transform a raw thought embedding into ADRAE's personal style.
        
        Args:
            vector: Raw thought embedding to style
            identity_vec: Identity vector for resonance alignment (optional)
            drift: Drift value for stability suppression (optional)
            novelty: Novelty value for exploration modulation (optional)
            
        Returns:
            Styled vector with ADRAE's cognitive fingerprint
        """
        if vector is None:
            return None
        
        v = safe_tensor(vector)
        idv = safe_tensor(identity_vec) if identity_vec is not None else None
        
        if TORCH_AVAILABLE and torch is not None:
            try:
                # Try torch-based styling
                if isinstance(v, torch.Tensor):
                    # Ensure identity vector has same shape
                    if idv is not None:
                        idv_t = safe_tensor(idv)
                        if isinstance(idv_t, torch.Tensor):
                            if idv_t.shape != v.shape and idv_t.numel() == v.numel():
                                idv_t = idv_t.reshape(v.shape)
                            elif idv_t.shape != v.shape:
                                idv_t = None
                        else:
                            idv_t = None
                    else:
                        idv_t = None
                    
                    if idv_t is None:
                        idv_t = torch.zeros_like(v)
                    
                    # --- Style 1: Curvature (nonlinear bending of thought path)
                    curved = v + self.curvature * torch.tanh(v * 2.0)
                    
                    # --- Style 2: Resonance with identity
                    aligned = curved + self.resonance_gain * idv_t
                    
                    # --- Style 3: Novelty modulation
                    if novelty is not None and isinstance(novelty, (int, float)):
                        try:
                            noise = torch.randn_like(v)
                            aligned = aligned + self.novelty_pull * (novelty * noise)
                        except Exception:
                            pass  # Skip novelty if noise generation fails
                    
                    # --- Style 4: Stability suppression of drift
                    if drift is not None and isinstance(drift, (int, float)):
                        try:
                            aligned = aligned - drift * self.stability_weight * torch.sign(aligned)
                        except Exception:
                            pass  # Skip drift correction if it fails
                    
                    # Normalize final style vector
                    norm = torch.linalg.norm(aligned)
                    if norm > 0:
                        aligned = aligned / norm
                    
                    return aligned.tolist()
            except Exception:
                pass  # Fall back to list-based implementation
        
        # Fallback: list-based styling
        try:
            # Convert to lists
            if isinstance(v, list):
                v_list = v
            elif TORCH_AVAILABLE and isinstance(v, torch.Tensor):
                v_list = v.tolist()
            elif hasattr(v, '__iter__'):
                v_list = list(v)
            else:
                return vector
            
            # Get identity vector as list
            if idv is not None:
                if isinstance(idv, list):
                    idv_list = idv
                elif TORCH_AVAILABLE and isinstance(idv, torch.Tensor):
                    idv_list = idv.tolist()
                elif hasattr(idv, '__iter__'):
                    idv_list = list(idv)
                else:
                    idv_list = None
            else:
                idv_list = None
            
            # Ensure same length
            if idv_list is not None:
                min_len = min(len(v_list), len(idv_list))
                v_list = v_list[:min_len]
                idv_list = idv_list[:min_len]
            else:
                idv_list = [0.0] * len(v_list)
            
            import math
            
            # --- Style 1: Curvature (nonlinear bending)
            curved = [
                v + self.curvature * math.tanh(v * 2.0)
                for v in v_list
            ]
            
            # --- Style 2: Resonance with identity
            aligned = [
                c + self.resonance_gain * i
                for c, i in zip(curved, idv_list)
            ]
            
            # --- Style 3: Novelty modulation
            if novelty is not None and isinstance(novelty, (int, float)):
                try:
                    import random
                    noise = [random.gauss(0, 1) for _ in aligned]
                    aligned = [
                        a + self.novelty_pull * (novelty * n)
                        for a, n in zip(aligned, noise)
                    ]
                except Exception:
                    pass
            
            # --- Style 4: Stability suppression of drift
            if drift is not None and isinstance(drift, (int, float)):
                try:
                    aligned = [
                        a - drift * self.stability_weight * (1.0 if a >= 0 else -1.0)
                        for a in aligned
                    ]
                except Exception:
                    pass
            
            # Normalize final style vector
            norm = safe_norm(aligned)
            if norm > 0:
                aligned = [a / norm for a in aligned]
            
            return aligned
        except Exception:
            return vector
    
    def update_tempo(self, cognitive_load, coherence):
        """
        Dynamic tempo: adjusts how "fast" transitions evolve.
        
        Args:
            cognitive_load: Current cognitive load (0.0 to 1.0)
            coherence: Current coherence level (0.0 to 1.0)
        """
        try:
            self.tempo = max(0.2, min(2.0, coherence + 0.5 - cognitive_load * 0.3))
        except Exception:
            pass  # Keep current tempo if update fails
    
    def summarize(self):
        """
        Get summary of current style parameters.
        
        Returns:
            Dict with all style parameters
        """
        return {
            "tempo": self.tempo,
            "curvature": self.curvature,
            "depth_bias": self.depth_bias,
            "resonance_gain": self.resonance_gain,
            "novelty_pull": self.novelty_pull,
            "stability_weight": self.stability_weight
        }

