# prime-core/cognition/global_workspace_reinforcement.py

# ============================================
# A182 — Intent-Aware Global Workspace Reinforcement
# ============================================
# Learns which cognitive patterns should be:
#   - strengthened
#   - weakened
#   - preserved
#   - suppressed
#
# based on the Meta-Intent Coordinator (A181).

try:
    import torch
except ImportError:
    torch = None

from ..neural.torch_utils import safe_tensor, safe_cosine_similarity, TORCH_AVAILABLE


class GlobalWorkspaceReinforcement:
    """
    A182 — Intent-Aware Global Workspace Reinforcement

    Learns which cognitive patterns should be:

      - strengthened
      - weakened
      - preserved
      - suppressed

    based on the Meta-Intent Coordinator (A181).
    """

    def __init__(self, dim=128):
        self.dim = dim

        # Reinforcement vector accumulates long-term bias
        if TORCH_AVAILABLE:
            self.reinforcement_vector = torch.zeros(dim)
        else:
            self.reinforcement_vector = [0.0] * dim

        # Learning rate controls how quickly reinforcement adapts
        self.lr = 0.05

    def reinforce(self, intent_vector, workspace_vector):
        """
        Strengthen pathways aligned with intent.
        Weaken pathways misaligned.

        Args:
            intent_vector: Intent vector from meta-intent coordinator
            workspace_vector: Chosen workspace vector (selected thought)

        Returns:
            Updated reinforcement vector
        """
        if intent_vector is None or workspace_vector is None:
            return self.reinforcement_vector

        intent_tensor = safe_tensor(intent_vector)
        workspace_tensor = safe_tensor(workspace_vector)

        if intent_tensor is None or workspace_tensor is None:
            return self.reinforcement_vector

        # Ensure reinforcement vector matches dimensions
        if TORCH_AVAILABLE:
            if isinstance(self.reinforcement_vector, torch.Tensor):
                if self.reinforcement_vector.shape[0] != intent_tensor.shape[0]:
                    # Resize reinforcement vector to match
                    self.reinforcement_vector = torch.zeros(intent_tensor.shape[0])
            else:
                # Convert list to tensor
                self.reinforcement_vector = torch.zeros(intent_tensor.shape[0])

        # Similarity determines whether reinforcement should increase
        similarity = safe_cosine_similarity(intent_tensor, workspace_tensor)
        
        if similarity is None:
            return self.reinforcement_vector

        similarity = float(similarity)

        # Positive similarity → strengthen
        # Negative similarity → weaken
        if TORCH_AVAILABLE and isinstance(intent_tensor, torch.Tensor):
            delta = intent_tensor * (similarity * self.lr)
            
            if isinstance(self.reinforcement_vector, torch.Tensor):
                self.reinforcement_vector = self.reinforcement_vector + delta
            else:
                # Convert to tensor
                self.reinforcement_vector = torch.tensor(self.reinforcement_vector) + delta

            # Normalize for stability
            norm = torch.norm(self.reinforcement_vector)
            if norm > 0:
                self.reinforcement_vector = self.reinforcement_vector / norm
        else:
            # Fallback for lists/arrays
            import math
            delta = [v * (similarity * self.lr) for v in intent_tensor]
            self.reinforcement_vector = [
                r + d for r, d in zip(self.reinforcement_vector, delta)
            ]
            # Normalize
            norm = math.sqrt(sum(x * x for x in self.reinforcement_vector))
            if norm > 0:
                self.reinforcement_vector = [r / norm for r in self.reinforcement_vector]

        return self.reinforcement_vector

    def apply_bias(self, candidate_vector):
        """
        Apply reinforcement to influence candidate selection.

        Args:
            candidate_vector: Thought candidate vector

        Returns:
            Biased candidate vector
        """
        if candidate_vector is None:
            return candidate_vector

        candidate_tensor = safe_tensor(candidate_vector)
        if candidate_tensor is None:
            return candidate_vector

        # Ensure dimensions match
        if TORCH_AVAILABLE:
            if isinstance(self.reinforcement_vector, torch.Tensor):
                if candidate_tensor.shape[0] != self.reinforcement_vector.shape[0]:
                    # If dimensions don't match, return original
                    return candidate_vector
                
                biased = candidate_tensor + (self.reinforcement_vector * self.lr)
                
                # Normalize output
                norm = torch.norm(biased)
                if norm > 0:
                    biased = biased / norm
                
                return biased
            else:
                # Reinforcement vector is a list, convert candidate to list
                if isinstance(candidate_tensor, torch.Tensor):
                    candidate_list = candidate_tensor.tolist()
                else:
                    candidate_list = list(candidate_tensor)
                
                if len(candidate_list) != len(self.reinforcement_vector):
                    return candidate_vector
                
                biased = [
                    c + (r * self.lr)
                    for c, r in zip(candidate_list, self.reinforcement_vector)
                ]
                
                # Normalize
                import math
                norm = math.sqrt(sum(x * x for x in biased))
                if norm > 0:
                    biased = [b / norm for b in biased]
                
                return biased
        else:
            # No torch available, work with lists
            candidate_list = list(candidate_vector) if hasattr(candidate_vector, '__iter__') else [candidate_vector]
            
            if len(candidate_list) != len(self.reinforcement_vector):
                return candidate_vector
            
            biased = [
                c + (r * self.lr)
                for c, r in zip(candidate_list, self.reinforcement_vector)
            ]
            
            # Normalize
            import math
            norm = math.sqrt(sum(x * x for x in biased))
            if norm > 0:
                biased = [b / norm for b in biased]
            
            return biased

