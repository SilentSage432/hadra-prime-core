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

