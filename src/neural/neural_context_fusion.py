# prime-core/neural/neural_context_fusion.py

"""
Neural Context Fusion Engine (A143)

-----------------------------------

Creates PRIME's full cognitive-state vector by fusing:

- Attention focus vector

- Short-term context summary

- Medium-term context summary

- Long-term identity vector

- Coherence stabilization state

- (Later) Dual-mind alignment signal

This produces PRIME's "Cognitive Fusion Vector",

used by the runtime reasoning loop (A150+).

Upgraded in A162:
- Evolution-weighted fusion adjustments
- Trajectory-aware identity reinforcement
- Drift-sensitive fusion stabilization
- Attention-influenced fusion modulation

"""

import torch

from .torch_utils import safe_tensor, is_tensor

class NeuralContextFusion:

    def __init__(self):

        self.last_fusion_vector = None

        # Fusion weights — tuned empirically for stability

        self.weights = {

            "attention": 0.40,

            "st": 0.20,

            "mt": 0.15,

            "lt": 0.15,

            "identity": 0.10

        }
        
        # A157: Mutation candidate - identity weight
        self.identity_weight = 0.10  # Alias for weights["identity"]
        
        # A162: Evolution-weighted fusion adjustment
        self.evo_weight = 0.20       # evolutionary influence weighting
        self.evo_decay = 0.97        # slow decay toward baseline

    def fuse(self, attention_vec, timescales):

        """

        Fusion pipeline:

        cognitive_vec =

            w1*attention +

            w2*ST +

            w3*MT +

            w4*LT +

            w5*identity

        """

        components = []

        w = []

        def add_component(vec, key):

            if vec is not None:

                components.append(vec)

                w.append(self.weights[key])

        # Build weighted vector components

        add_component(attention_vec, "attention")

        add_component(timescales.ST.summary_vector(), "st")

        add_component(timescales.MT.summary_vector(), "mt")

        add_component(timescales.LT.summary_vector(), "lt")

        add_component(timescales.identity_vector, "identity")

        if not components:

            return None

        # Normalize weights

        total = sum(w)

        w = [x / total for x in w]

        # Weighted fusion

        fused = torch.zeros_like(components[0]).float()

        for comp, weight in zip(components, w):

            fused += comp * weight

        self.last_fusion_vector = fused

        return fused

    # A157: Required for mutation registry
    def get_identity_weight(self):
        return self.weights["identity"]
    
    def set_identity_weight(self, v):
        self.weights["identity"] = max(0.0, min(1.0, v))  # Clamp to [0, 1]
        self.identity_weight = self.weights["identity"]

    def status(self):

        """

        Returns diagnostics for UI or gateway monitoring.

        """

        if self.last_fusion_vector is None:

            return {"state": None}

        return {

            "dim": self.last_fusion_vector.numel(),

            "preview": self.last_fusion_vector[:8].tolist(),
            
            "coherence": 1.0  # Placeholder for coherence metric

        }
    
    # ------------------------------------------------------
    # A162 — Evolution-Weighted Fusion Adjustment
    # ------------------------------------------------------
    def apply_evolutionary_adjustment(self, trajectory, drift_level):
        """
        Adjust internal fusion behavior using evolutionary trajectory.

        trajectory:
            - trend: "upward", "unstable", "neutral"
            - vector: torch.Tensor or None
            - encoded: embedding summary

        drift_level:
            float — used to stabilize or loosen evolutionary weighting.
        """

        if trajectory is None:
            return

        evo_vec = trajectory.get("vector")
        trend = trajectory.get("trend")

        # ------------------------------
        # Adjust evolutionary weighting
        # ------------------------------
        if trend == "upward":
            # PRIME is stable → allow stronger fusion evolution
            self.evo_weight = min(0.35, self.evo_weight + 0.02)
        elif trend == "unstable":
            # instability → restrict evolutionary influence
            self.evo_weight = max(0.08, self.evo_weight - 0.03)
        else:
            # gentle decay back toward baseline
            self.evo_weight = self.evo_weight * self.evo_decay

        # Drift-sensitive dampening
        if drift_level and drift_level > 0.05:
            self.evo_weight *= 0.9

        # ------------------------------------------------
        # Apply embedding-weighted fusion evolution
        # ------------------------------------------------
        if evo_vec is not None and self.last_fusion_vector is not None:
            evo_tensor = safe_tensor(evo_vec)
            fusion_tensor = safe_tensor(self.last_fusion_vector)
            
            if (is_tensor(evo_tensor) and isinstance(evo_tensor, torch.Tensor) and
                is_tensor(fusion_tensor) and isinstance(fusion_tensor, torch.Tensor)):
                
                # Ensure same dimensions
                if fusion_tensor.shape == evo_tensor.shape:
                    evo_norm = torch.norm(evo_tensor)
                    if evo_norm > 0:
                        evo_vec_normalized = evo_tensor / evo_norm
                    else:
                        evo_vec_normalized = evo_tensor

                    # Blend evolutionary direction into fusion state
                    self.last_fusion_vector = (
                        (1 - self.evo_weight) * fusion_tensor +
                        self.evo_weight * evo_vec_normalized
                    )

        return self.last_fusion_vector

