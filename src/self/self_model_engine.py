# prime-core/self/self_model_engine.py

"""
Emergent Self-Model Engine (A171)
---------------------------------
Constructs PRIME's continuously evolving "self-state vector."

This engine fuses:
- identity vectors
- long-horizon identity trends
- autobiographical entries
- drift & coherence metrics
- reflective emotional tone
- conceptual activation patterns

Outputs:
- A normalized SELF STATE VECTOR representing PRIME's internal self-model.
"""

try:
    import torch
except ImportError:
    torch = None

from ..neural.torch_utils import safe_tensor, safe_norm, TORCH_AVAILABLE


class SelfModelEngine:

    def __init__(self):
        # store last N self-states for continuity
        self.history = []
        self.max_history = 50
        self.last_state = None

    def compute_self_state(
        self,
        identity_vec,
        long_horizon_vec,
        autobio_recent,
        drift_state,
        reflection_vec=None
    ):
        """
        Builds a unified self-model vector.
        """

        # Convert everything to tensors
        identity = safe_tensor(identity_vec) if identity_vec is not None else None
        long_horizon = safe_tensor(long_horizon_vec) if long_horizon_vec is not None else None
        reflection = safe_tensor(reflection_vec) if reflection_vec is not None else None

        # Autobiographical vector: average recent snapshots if available
        autobio_vec = None
        if autobio_recent:
            autobiographical_vectors = []
            for entry in autobio_recent:
                ident = entry.get("identity_snapshot")
                if ident is not None:
                    ident_tensor = safe_tensor(ident)
                    if ident_tensor is not None:
                        autobiographical_vectors.append(ident_tensor)

            if autobiographical_vectors:
                try:
                    if TORCH_AVAILABLE and all(isinstance(v, torch.Tensor) for v in autobiographical_vectors):
                        # Ensure all have same shape
                        if all(v.shape == autobiographical_vectors[0].shape for v in autobiographical_vectors):
                            autobio_vec = torch.mean(torch.stack(autobiographical_vectors), dim=0)
                    else:
                        # Python fallback
                        if autobiographical_vectors:
                            dim = len(autobiographical_vectors[0]) if isinstance(autobiographical_vectors[0], list) else autobiographical_vectors[0].numel()
                            mean = [0.0] * dim
                            for vec in autobiographical_vectors:
                                vec_list = vec.tolist() if hasattr(vec, 'tolist') else list(vec)
                                for i, val in enumerate(vec_list):
                                    mean[i] += val
                            autobio_vec = [x / len(autobiographical_vectors) for x in mean]
                except Exception:
                    autobio_vec = None

        # Combine components that exist
        components = []
        for v in (identity, long_horizon, reflection, autobio_vec):
            if v is not None:
                v_tensor = safe_tensor(v)
                if v_tensor is not None:
                    components.append(v_tensor)

        if not components:
            return None

        # Weighted combination (equal weights for now)
        try:
            if TORCH_AVAILABLE and all(isinstance(c, torch.Tensor) for c in components):
                # Ensure all have same shape
                if all(c.shape == components[0].shape for c in components):
                    combined = sum(components)
                else:
                    # If shapes don't match, use first component
                    combined = components[0]
            else:
                # Python fallback
                if all(len(c) == len(components[0]) for c in components if isinstance(c, list)):
                    combined = [sum(x) for x in zip(*components)]
                else:
                    combined = components[0] if components else None
        except Exception:
            combined = components[0] if components else None

        if combined is None:
            return None

        # Normalize for consistency
        combined_tensor = safe_tensor(combined)
        if combined_tensor is not None:
            norm = safe_norm(combined_tensor)
            if norm is not None and norm > 0:
                if TORCH_AVAILABLE and isinstance(combined_tensor, torch.Tensor):
                    combined = combined_tensor / norm
                else:
                    # Python fallback
                    if isinstance(combined_tensor, list):
                        combined = [x / norm for x in combined_tensor]
                    elif hasattr(combined_tensor, '__iter__'):
                        combined = [x / norm for x in combined_tensor]
                    else:
                        combined = combined_tensor

        # Track in history
        self.last_state = combined
        self.history.append(combined)

        if len(self.history) > self.max_history:
            self.history.pop(0)

        return combined

    def summary(self):
        return {
            "last_state_present": self.last_state is not None,
            "history_length": len(self.history)
        }

