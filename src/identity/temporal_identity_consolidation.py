# prime-core/identity/temporal_identity_consolidation.py

# ============================================
# A185 — Temporal Identity Consolidation (Sleep/Wake Style Cycles)
# ============================================
# PRIME stabilizes her identity by periodically entering a
# consolidation cycle where:
#   - identity drift is reversed
#   - stable traits are reinforced
#   - short-term identity perturbations are averaged out
#   - autobiographical memory is re-anchored

try:
    import torch
except ImportError:
    torch = None

from ..neural.torch_utils import safe_tensor, TORCH_AVAILABLE


class TemporalIdentityConsolidator:
    """
    A185 — Temporal Identity Consolidation (Sleep/Wake Style Cycles)

    PRIME stabilizes her identity by periodically entering a
    consolidation cycle where:

      - identity drift is reversed
      - stable traits are reinforced
      - short-term identity perturbations are averaged out
      - autobiographical memory is re-anchored
    """

    def __init__(self, consolidation_weight=0.35):
        """
        Args:
            consolidation_weight: Weight (0.0-1.0) for corrected identity in merge
                Higher = more aggressive drift reversal
        """
        self.consolidation_weight = consolidation_weight

    def consolidate(self, identity_vec, baseline_vec, drift_suppressor):
        """
        Consolidation rule:
          identity_new = identity_old * (1 - w) + corrected_identity * w

        This merges the current identity with a drift-corrected version,
        pulling it back toward baseline while preserving meaningful evolution.

        Args:
            identity_vec: Current identity vector
            baseline_vec: Baseline (frozen) identity vector
            drift_suppressor: IdentityDriftSuppressor instance for correction

        Returns:
            Consolidated identity vector (normalized)
        """
        if identity_vec is None or baseline_vec is None:
            return identity_vec

        # Get drift-corrected identity
        corrected, _ = drift_suppressor.correct(identity_vec, baseline_vec)

        if corrected is None:
            return identity_vec

        # Convert to tensors
        identity_tensor = safe_tensor(identity_vec)
        corrected_tensor = safe_tensor(corrected)

        if identity_tensor is None or corrected_tensor is None:
            return identity_vec

        # Ensure dimensions match
        if TORCH_AVAILABLE:
            if isinstance(identity_tensor, torch.Tensor) and isinstance(corrected_tensor, torch.Tensor):
                if identity_tensor.shape != corrected_tensor.shape:
                    # If dimensions don't match, return original
                    return identity_vec

                # Consolidation: weighted merge of current and corrected
                merged = identity_tensor * (1.0 - self.consolidation_weight) + corrected_tensor * self.consolidation_weight

                # Normalize for stability
                norm = torch.norm(merged)
                if norm > 0:
                    merged = merged / norm

                return merged
            else:
                # Fallback for lists/arrays
                import math
                identity_list = list(identity_tensor) if hasattr(identity_tensor, '__iter__') else [identity_tensor]
                corrected_list = list(corrected_tensor) if hasattr(corrected_tensor, '__iter__') else [corrected_tensor]

                if len(identity_list) != len(corrected_list):
                    return identity_vec

                merged = [
                    i * (1.0 - self.consolidation_weight) + c * self.consolidation_weight
                    for i, c in zip(identity_list, corrected_list)
                ]

                # Normalize
                norm = math.sqrt(sum(x * x for x in merged))
                if norm > 0:
                    merged = [m / norm for m in merged]

                return merged
        else:
            # No torch available
            return identity_vec

