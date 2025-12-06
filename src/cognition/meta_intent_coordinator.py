# prime-core/cognition/meta_intent_coordinator.py

# ============================================
# A181 — Meta-Intent Coordination Layer (MICL)
# ============================================
# PRIME must balance:
#   - operator intent (external)
#   - system intent (rules, stability)
#   - self intent (curiosity, growth, identity need)
#
# The output is a weighted "intent vector" used to bias thought selection,
# cognitive actions, attention, and evolution pathways.

try:
    import torch
except ImportError:
    torch = None

from ..neural.torch_utils import safe_tensor, TORCH_AVAILABLE


class MetaIntentCoordinator:
    """
    A181 — Meta-Intent Coordination Layer (MICL)

    PRIME must balance:

      - operator intent (external)
      - system intent (rules, stability)
      - self intent (curiosity, growth, identity need)

    The output is a weighted "intent vector" used to bias thought selection,
    cognitive actions, attention, and evolution pathways.
    """

    def __init__(self):
        self.last_intent_weights = {
            "operator": 0.33,
            "system": 0.33,
            "self": 0.34,
        }

    def compute_intent_weights(self, state, operator_context):
        """
        Dynamically compute weighting for the three intent streams.

        Args:
            state: Cognitive state object
            operator_context: List of operator tasks/commands (or None)

        Returns:
            dict: Intent weights for operator, system, and self
        """
        # Operator intent strength (external commands, tasks)
        op_strength = 0.3
        if operator_context is not None and len(operator_context) > 0:
            op_strength = min(1.0, 0.3 + 0.1 * len(operator_context))

        # System intent (coherence & stability)
        drift = 0.0
        try:
            if hasattr(state, 'drift') and state.drift is not None:
                if hasattr(state.drift, 'get_status'):
                    drift_status = state.drift.get_status()
                    if drift_status and isinstance(drift_status, dict):
                        drift = float(drift_status.get("latest_drift", 0.0))
                elif hasattr(state.drift, 'latest_drift'):
                    drift = float(state.drift.latest_drift or 0.0)
        except Exception:
            drift = 0.0

        stability_pressure = max(0.1, 1.0 - min(drift, 1.0))

        # Self-intent (evolution pressure)
        evolve_pressure = 0.5
        try:
            if hasattr(state, 'evolution_pressure'):
                evolve_pressure = float(state.evolution_pressure)
            else:
                # Default to neutral if not set
                evolve_pressure = 0.5
        except Exception:
            evolve_pressure = 0.5

        # Normalize final mix
        total = op_strength + stability_pressure + evolve_pressure
        if total > 0:
            operator_w = op_strength / total
            system_w = stability_pressure / total
            self_w = evolve_pressure / total
        else:
            # Fallback to equal weights
            operator_w = 0.33
            system_w = 0.33
            self_w = 0.34

        self.last_intent_weights = {
            "operator": operator_w,
            "system": system_w,
            "self": self_w,
        }

        return self.last_intent_weights

    def combine_intents(self, operator_vec, system_vec, self_vec):
        """
        Produce an intent vector that biases cognitive processing.

        Args:
            operator_vec: Vector representing operator intent (e.g., task embedding)
            system_vec: Vector representing system intent (e.g., identity vector)
            self_vec: Vector representing self intent (e.g., evolution vector)

        Returns:
            Combined intent vector (tensor or list)
        """
        ow = self.last_intent_weights.get("operator", 0.33)
        sw = self.last_intent_weights.get("system", 0.33)
        hw = self.last_intent_weights.get("self", 0.34)

        # Convert all to tensors if possible
        operator_tensor = safe_tensor(operator_vec) if operator_vec is not None else None
        system_tensor = safe_tensor(system_vec) if system_vec is not None else None
        self_tensor = safe_tensor(self_vec) if self_vec is not None else None

        # If any component is missing, use zero vector of appropriate size
        # Try to infer size from non-None vectors
        reference_vec = operator_tensor or system_tensor or self_tensor
        if reference_vec is None:
            # No vectors available, return None
            return None

        # Get size from reference
        if TORCH_AVAILABLE and isinstance(reference_vec, torch.Tensor):
            vec_size = reference_vec.shape[0] if len(reference_vec.shape) > 0 else len(reference_vec)
            zero_vec = torch.zeros(vec_size)
        else:
            vec_size = len(reference_vec) if hasattr(reference_vec, '__len__') else 0
            zero_vec = [0.0] * vec_size if vec_size > 0 else None

        if zero_vec is None:
            return None

        # Replace None vectors with zero vectors
        if operator_tensor is None:
            operator_tensor = safe_tensor(zero_vec)
        if system_tensor is None:
            system_tensor = safe_tensor(zero_vec)
        if self_tensor is None:
            self_tensor = safe_tensor(zero_vec)

        # Combine
        if TORCH_AVAILABLE and isinstance(operator_tensor, torch.Tensor):
            combined = operator_tensor * ow + system_tensor * sw + self_tensor * hw

            # Normalize if tensor
            norm = torch.norm(combined)
            if norm > 0:
                combined = combined / norm
        else:
            # Fallback for lists/arrays
            combined = [
                (op * ow + sys * sw + self * hw)
                for op, sys, self in zip(
                    operator_tensor if isinstance(operator_tensor, list) else list(operator_tensor),
                    system_tensor if isinstance(system_tensor, list) else list(system_tensor),
                    self_tensor if isinstance(self_tensor, list) else list(self_tensor)
                )
            ]
            # Normalize
            import math
            norm = math.sqrt(sum(x * x for x in combined))
            if norm > 0:
                combined = [x / norm for x in combined]

        return combined

