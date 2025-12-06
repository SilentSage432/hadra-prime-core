# prime-core/neural/neural_state_tracker.py

"""
Neural State Tracker

--------------------

Central monitoring system for neural activity.

Aggregates:

- latest embedding

- drift engine metrics

- embedding stats

- system-level coherence signals (added later)

"""

from .neural_drift_engine import NeuralDriftEngine
from .neural_timescales import NeuralTimescales

from .torch_utils import is_tensor

class NeuralStateTracker:

    def __init__(self):

        self.last_embedding = None
        self.last_perception = None
        self.task_embeddings = []

        self.drift = NeuralDriftEngine()
        self.timescales = NeuralTimescales()

    def update(self, embedding):

        """

        Update global neural state and push into drift engine.

        """

        if embedding is None:

            return

        self.last_embedding = embedding

        self.drift.record(embedding)
        self.timescales.update(embedding)

    def update_identity_vector(self, new_embedding):
        """
        ADRAE Identity Blend (A-SOV-02)
        
        Weighted for stable identity evolution.
        Updates the identity vector in timescales with ADRAE-weighted blend.
        
        Args:
            new_embedding: New embedding to blend into identity
        """
        if new_embedding is None or self.timescales.identity_vector is None:
            return
        
        try:
            from .torch_utils import safe_tensor
            import torch
            
            current = safe_tensor(self.timescales.identity_vector)
            new = safe_tensor(new_embedding)
            
            if current is None or new is None:
                return
            
            # -----------------------------------------------------------
            # A-SOV-03: Identity Migration Safety Layer
            # -----------------------------------------------------------
            # Prevent sudden identity jumps by enforcing:
            # - bounded drift
            # - continuity preservation
            # - ADRAE-weighted stabilization
            
            alpha = 0.25      # stable evolution weight
            max_delta = 0.15  # limit sudden identity shifts
            
            if isinstance(current, torch.Tensor) and isinstance(new, torch.Tensor):
                if current.shape == new.shape:
                    old = current
                    proposed = (1 - alpha) * old + alpha * new
                    
                    # clamp delta magnitude for safety
                    delta = proposed - old
                    delta_norm = torch.norm(delta)
                    if delta_norm > max_delta:
                        delta = delta / delta_norm * max_delta
                    self.timescales.identity_vector = old + delta
                else:
                    # If shapes don't match, use simple blend
                    self.timescales.identity_vector = (
                        (1 - alpha) * current +
                        alpha * new
                    )
        except Exception:
            # If update fails, continue with existing identity
            pass

    def compute_uncertainty(self, fusion_vec, attention_vec, drift_state):
        """
        A216 — Computes an uncertainty score based on:
        - Low similarity = high uncertainty
        - High drift = high uncertainty
        - Weak attention = high uncertainty
        - Fusion instability = high uncertainty
        
        Returns:
            float: Uncertainty value in [0.0, 1.0]
        """
        from .torch_utils import safe_tensor, safe_norm, TORCH_AVAILABLE
        
        # Drift contribution
        drift_val = drift_state.get("latest_drift") if drift_state else 0.0
        if drift_val is None:
            drift_val = 0.0
        drift_term = min(1.0, abs(float(drift_val)))
        
        # Fusion stability
        fusion_term = 0.0
        if fusion_vec is None:
            fusion_term = 1.0  # No fusion = high uncertainty
        else:
            fusion_tensor = safe_tensor(fusion_vec)
            if fusion_tensor is not None:
                if TORCH_AVAILABLE and hasattr(fusion_tensor, 'norm'):
                    norm = float(fusion_tensor.norm().item())
                else:
                    norm = safe_norm(fusion_tensor)
                
                # Low norm = unstable fusion = high uncertainty
                if norm < 0.5:
                    fusion_term = 0.7
                elif norm < 0.3:
                    fusion_term = 1.0
                else:
                    fusion_term = 0.0
            else:
                fusion_term = 0.5  # Unknown state = moderate uncertainty
        
        # Attention coherence
        attention_term = 0.0
        if attention_vec is None:
            attention_term = 1.0  # No attention = high uncertainty
        else:
            attention_tensor = safe_tensor(attention_vec)
            if attention_tensor is not None:
                if TORCH_AVAILABLE and hasattr(attention_tensor, 'norm'):
                    norm = float(attention_tensor.norm().item())
                else:
                    norm = safe_norm(attention_tensor)
                
                # Low norm = weak attention = high uncertainty
                if norm < 0.3:
                    attention_term = 0.8
                elif norm < 0.5:
                    attention_term = 0.4
                else:
                    attention_term = 0.0
            else:
                attention_term = 0.3  # Unknown state = moderate uncertainty
        
        # Combine terms (weighted average)
        uncertainty = (drift_term * 0.4 + fusion_term * 0.35 + attention_term * 0.25)
        
        # Clamp to [0.0, 1.0]
        return min(1.0, max(0.0, uncertainty))

    def summary(self):

        emb = self.last_embedding

        return {

            "last_embedding_dim": emb.numel() if emb is not None and is_tensor(emb) else None,

            "drift": self.drift.get_status(),

            "timescales": self.timescales.summary()

        }

