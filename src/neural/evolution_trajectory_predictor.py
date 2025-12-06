# prime-core/neural/evolution_trajectory_predictor.py

"""
Evolutionary Trajectory Predictor (A160)
---------------------------------------
Predicts PRIME's future developmental direction based on:
 - evolution events
 - drift/stability history
 - identity vector movement
 - semantic memory consolidation
 - reflection thematic patterns

This module provides:
 - A predicted next-step evolutionary vector
 - A narrative explanation of expected developmental trends
 - Indicators of upward/downward stability motion
"""

try:
    import torch
except ImportError:
    torch = None

from .torch_utils import safe_tensor, TORCH_AVAILABLE


class EvolutionaryTrajectoryPredictor:

    def __init__(self, window=20):
        self.window = window
        self.history = []  # Store final evo events after consolidation

    def record(self, evo_summary):
        """Store narrative evolution summary for trajectory modeling."""
        if not evo_summary:
            return
        self.history.append(evo_summary)
        if len(self.history) > self.window:
            self.history.pop(0)

    def compute_vector_trend(self, timescales):
        """Analyze identity vector movement as a trend indicator."""
        if not timescales or not hasattr(timescales, "identity_vector"):
            return None
        
        iv = timescales.identity_vector
        iv_tensor = safe_tensor(iv)
        
        if not TORCH_AVAILABLE or not isinstance(iv_tensor, torch.Tensor):
            return None
        
        # Normalize
        norm = torch.norm(iv_tensor)
        if norm > 0:
            return iv_tensor / norm
        return iv_tensor

    def predict(self, drift_state, timescales, hooks):
        """Produce a forward-looking prediction of PRIME's evolution."""

        if not self.history:
            return {
                "prediction": "Insufficient data for trajectory modeling.",
                "vector": None,
                "trend": "unknown"
            }

        # Build narrative coherence trend
        upward = sum("improved" in h.get("summary_text", "").lower() for h in self.history)
        rollback = sum("reverted" in h.get("summary_text", "").lower() for h in self.history)

        # Determine overall trajectory
        if upward > rollback:
            direction = "upward"
        elif rollback > upward:
            direction = "unstable"
        else:
            direction = "neutral"

        # Compute vector trend
        vector_trend = self.compute_vector_trend(timescales)

        # Predict next-step direction
        if direction == "upward":
            predicted_msg = (
                "Evolution trending upward: PRIME is increasingly stable. "
                "Expected next step: refinement of attention + fusion coherence."
            )
        elif direction == "unstable":
            predicted_msg = (
                "Evolution shows instability tendencies. "
                "Expected next step: consolidation of identity and drift reduction."
            )
        else:
            predicted_msg = (
                "Evolution trend neutral. PRIME may explore new conceptual directions."
            )

        # Encode prediction as embedding for internal use
        encoded = None
        if hooks and hasattr(hooks, "on_reflection"):
            try:
                encoded = hooks.on_reflection(predicted_msg)
            except Exception:
                encoded = None

        return {
            "trend": direction,
            "prediction": predicted_msg,
            "vector": vector_trend,
            "raw_events": [h.get("summary_text", "") for h in self.history],
            "encoded": encoded,
            "upward_count": upward,
            "rollback_count": rollback
        }

