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

    def summary(self):

        emb = self.last_embedding

        return {

            "last_embedding_dim": emb.numel() if emb is not None and is_tensor(emb) else None,

            "drift": self.drift.get_status(),

            "timescales": self.timescales.summary()

        }

