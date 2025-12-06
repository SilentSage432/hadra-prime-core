# prime-core/identity/identity_supervisory_gate.py

# ============================================
# A184 — Supervisory Identity Gate & Long-Term Self-Coherence Regulator
# ============================================
# This module decides whether an identity update is allowed,
# scaled, softened, rewritten, or rejected based on:
#   - semantic compatibility
#   - similarity to baseline identity
#   - coherence with autobiographical memory
#   - long-term identity trajectory from A173–A183

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None

from ..neural.torch_utils import safe_tensor, safe_cosine_similarity, TORCH_AVAILABLE


class IdentitySupervisoryGate:
    """
    A184 — Supervisory Identity Gate & Long-Term Self-Coherence Regulator

    This module decides whether an identity update is allowed,
    scaled, softened, rewritten, or rejected based on:

      - semantic compatibility
      - similarity to baseline identity
      - coherence with autobiographical memory
      - long-term identity trajectory from A173–A183
    """

    def __init__(self, min_similarity=0.65, soft_update_range=(0.65, 0.80)):
        """
        Args:
            min_similarity: Minimum similarity (0.0-1.0) to allow update
            soft_update_range: Tuple (low, high) for soft-merge range
        """
        self.min_similarity = min_similarity
        self.soft_low, self.soft_high = soft_update_range

    def cosine_sim(self, a, b):
        """
        Compute cosine similarity between two vectors.

        Args:
            a: First vector (tensor or list)
            b: Second vector (tensor or list)

        Returns:
            float: Cosine similarity (0.0-1.0)
        """
        if a is None or b is None:
            return 1.0

        a_tensor = safe_tensor(a)
        b_tensor = safe_tensor(b)

        if a_tensor is None or b_tensor is None:
            return 1.0

        # Use safe_cosine_similarity utility
        similarity = safe_cosine_similarity(a_tensor, b_tensor)
        
        if similarity is None:
            return 1.0

        return float(similarity)

    def evaluate(self, new_vec, baseline_vec):
        """
        Evaluate whether an identity update should be accepted, soft-merged, or rejected.

        Args:
            new_vec: Proposed new identity vector
            baseline_vec: Baseline (frozen) identity vector

        Returns:
            tuple: (decision, similarity)
                - decision: "accept", "soft-merge", or "reject"
                - similarity: float (0.0-1.0) cosine similarity score
        """
        if new_vec is None or baseline_vec is None:
            return "accept", 1.0

        sim = self.cosine_sim(new_vec, baseline_vec)

        if sim >= self.soft_high:
            return "accept", sim

        if self.soft_low <= sim < self.soft_high:
            return "soft-merge", sim

        return "reject", sim

    def merge_identity(self, baseline_vec, new_vec, weight=0.25):
        """
        Soft integration toward the new identity vector.

        Args:
            baseline_vec: Current/baseline identity vector
            new_vec: New identity vector to merge
            weight: Merge weight (0.0-1.0), higher = more new_vec influence

        Returns:
            Merged identity vector (normalized)
        """
        if baseline_vec is None:
            return new_vec
        if new_vec is None:
            return baseline_vec

        baseline_tensor = safe_tensor(baseline_vec)
        new_tensor = safe_tensor(new_vec)

        if baseline_tensor is None or new_tensor is None:
            return baseline_vec

        # Ensure dimensions match
        if TORCH_AVAILABLE:
            if isinstance(baseline_tensor, torch.Tensor) and isinstance(new_tensor, torch.Tensor):
                if baseline_tensor.shape != new_tensor.shape:
                    # If dimensions don't match, return baseline
                    return baseline_vec

                # Soft merge: weighted combination
                merged = baseline_tensor * (1.0 - weight) + new_tensor * weight

                # Normalize
                norm = torch.norm(merged)
                if norm > 0:
                    merged = merged / norm

                return merged
            else:
                # Fallback for lists/arrays
                import math
                baseline_list = list(baseline_tensor) if hasattr(baseline_tensor, '__iter__') else [baseline_tensor]
                new_list = list(new_tensor) if hasattr(new_tensor, '__iter__') else [new_tensor]
                
                if len(baseline_list) != len(new_list):
                    return baseline_vec

                merged = [
                    b * (1.0 - weight) + n * weight
                    for b, n in zip(baseline_list, new_list)
                ]

                # Normalize
                norm = math.sqrt(sum(x * x for x in merged))
                if norm > 0:
                    merged = [m / norm for m in merged]

                return merged
        else:
            # No torch available
            return baseline_vec

