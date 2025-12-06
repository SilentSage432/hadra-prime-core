"""
Global Workspace Continuity Engine (A202)
----------------------------------------
Maintains cross-cycle continuity signals so ADRAE can preserve and
evolve intentions, reflective threads, and multi-step reasoning paths.

This provides:
- Continuity vectors that persist across cycles
- Updating based on thought selection, attention, and drift
- Intent carryover signals
- Narrative-thread reinforcement
"""

from ..neural.torch_utils import safe_tensor, TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch


class GlobalWorkspaceContinuity:

    def __init__(self, dim=128, decay=0.97):
        self.dim = dim
        self.decay = decay

        if TORCH_AVAILABLE:
            self.state = torch.zeros(dim)
        else:
            self.state = [0.0] * dim


    def update(self, chosen_vector, attention_vector, drift_level):
        """
        Continuity evolves based on:
        - chosen thought embedding
        - attention focus
        - drift metric
        """

        cv = safe_tensor(chosen_vector) if chosen_vector is not None else None
        av = safe_tensor(attention_vector) if attention_vector is not None else None

        if TORCH_AVAILABLE:
            # Decay old continuity
            self.state = self.state * self.decay

            # Strengthen continuity with chosen thought
            if cv is not None:
                self.state = self.state + (cv * (1.0 - drift_level))

            # Add directional pull from attention
            if av is not None:
                self.state = self.state + (av * 0.25)

        else:
            # Python list fallback (slower but functional)
            self.state = [x * self.decay for x in self.state]

            if cv is not None:
                self.state = [s + c * (1.0 - drift_level)
                              for s, c in zip(self.state, cv)]

            if av is not None:
                self.state = [s + a * 0.25
                              for s, a in zip(self.state, av)]

        return self.state


    def get(self):
        """Return current continuity vector."""
        return self.state

