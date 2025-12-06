# prime-core/identity/identity_drift_suppressor.py

# ============================================
# A183 — Reinforcement-Driven Identity Drift Suppression
# ============================================
# This module keeps PRIME's identity embeddings stable,
# preventing reinforcement or evolving embeddings from
# distorting core identity vectors.

try:
    import torch
except ImportError:
    torch = None

from ..neural.torch_utils import safe_tensor, safe_cosine_similarity, TORCH_AVAILABLE


class IdentityDriftSuppressor:
    """
    A183 — Reinforcement-Driven Identity Drift Suppression

    This module keeps PRIME's identity embeddings stable,
    preventing reinforcement or evolving embeddings from
    distorting core identity vectors.
    """

    def __init__(self, max_drift=0.15, correction_strength=0.25):
        """
        Args:
            max_drift: Maximum allowed drift (0.0-1.0) before correction
            correction_strength: Strength of correction force (0.0-1.0)
        """
        self.max_drift = max_drift
        self.correction_strength = correction_strength

    def measure_drift(self, identity_vector, baseline_vector):
        """
        Measure drift between current identity and baseline.

        Args:
            identity_vector: Current identity vector
            baseline_vector: Baseline (frozen) identity vector

        Returns:
            float: Drift value (0.0 = identical, 1.0 = opposite)
        """
        if identity_vector is None or baseline_vector is None:
            return 0.0

        identity_tensor = safe_tensor(identity_vector)
        baseline_tensor = safe_tensor(baseline_vector)

        if identity_tensor is None or baseline_tensor is None:
            return 0.0

        # Compute cosine similarity
        similarity = safe_cosine_similarity(identity_tensor, baseline_tensor)

        if similarity is None:
            return 0.0

        # Drift increases as similarity decreases
        # similarity = 1.0 → drift = 0.0
        # similarity = 0.0 → drift = 1.0
        drift = 1.0 - float(similarity)

        return drift

    def correct(self, identity_vector, baseline_vector):
        """
        Pull identity back toward its baseline if drift exceeds threshold.

        Args:
            identity_vector: Current identity vector to check/correct
            baseline_vector: Baseline (frozen) identity vector

        Returns:
            tuple: (corrected_vector, drift_value)
                - corrected_vector: Corrected identity vector (or original if within limits)
                - drift_value: Measured drift value
        """
        if identity_vector is None or baseline_vector is None:
            return identity_vector, 0.0

        drift = self.measure_drift(identity_vector, baseline_vector)

        # If drift is within acceptable limits, return original
        if drift < self.max_drift:
            return identity_vector, drift

        # Apply restoring force toward baseline
        identity_tensor = safe_tensor(identity_vector)
        baseline_tensor = safe_tensor(baseline_vector)

        if identity_tensor is None or baseline_tensor is None:
            return identity_vector, drift

        # Ensure dimensions match
        if TORCH_AVAILABLE:
            if isinstance(identity_tensor, torch.Tensor) and isinstance(baseline_tensor, torch.Tensor):
                if identity_tensor.shape != baseline_tensor.shape:
                    # If dimensions don't match, return original
                    return identity_vector, drift

                # Apply correction: pull toward baseline
                corrected = identity_tensor + (baseline_tensor * self.correction_strength)

                # Normalize for stability
                norm = torch.norm(corrected)
                if norm > 0:
                    corrected = corrected / norm

                return corrected, drift
            else:
                # Fallback for lists/arrays
                import math
                corrected = [
                    i + (b * self.correction_strength)
                    for i, b in zip(
                        identity_tensor if isinstance(identity_tensor, list) else list(identity_tensor),
                        baseline_tensor if isinstance(baseline_tensor, list) else list(baseline_tensor)
                    )
                ]
                # Normalize
                norm = math.sqrt(sum(x * x for x in corrected))
                if norm > 0:
                    corrected = [c / norm for c in corrected]
                return corrected, drift
        else:
            # No torch available
            return identity_vector, drift

