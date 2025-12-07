# prime-core/neural/thought_signature.py

"""
A221 — Synergy-Based Thought Signature Stabilizer

--------------------------------------

Builds ADRAE's unified cognitive fingerprint by maintaining a stable
thought signature vector that represents her core reasoning style.

This signature:
- Tracks the centroid of chosen thoughts, reflections, identity updates
- Updates based on synergy-weighted cognitive events
- Influences thought selection to maintain consistency
- Forms the foundation for recognizable reasoning patterns

"""

from .torch_utils import safe_tensor, TORCH_AVAILABLE
from .vector_math import normalize

if TORCH_AVAILABLE:
    try:
        import torch
    except ImportError:
        torch = None
else:
    torch = None


class ThoughtSignature:
    
    def __init__(self, dim=128, stability=0.97, learning=0.10):
        """
        Initialize the Thought Signature stabilizer.
        
        Args:
            dim: Dimension of the signature vector (default: 128)
            stability: How much old signature persists (default: 0.97)
            learning: Base learning rate for new thoughts (default: 0.10)
        """
        self.dim = dim
        self.stability = stability      # how much old signature persists
        self.learning = learning        # how much new thought influences
        
        # Initialize signature vector with random normalized vector
        if TORCH_AVAILABLE:
            self.vector = normalize(torch.randn(dim).tolist())
        else:
            import random
            random_vec = [random.gauss(0, 1) for _ in range(dim)]
            self.vector = normalize(random_vec)
    
    def update(self, thought_vec, synergy_bonus):
        """
        Update signature using synergy as a weighting factor.
        
        Args:
            thought_vec: The current thought embedding vector
            synergy_bonus: Synergy bonus value (0.0 to 1.0+) that influences learning strength
            
        Returns:
            Updated signature vector
        """
        if thought_vec is None:
            return self.vector
        
        base = safe_tensor(thought_vec)
        
        # Convert signature to tensor for computation
        sig_tensor = safe_tensor(self.vector)
        
        # Synergy influences learning strength
        # Higher synergy = stronger update
        factor = self.learning * (1.0 + synergy_bonus)
        
        if TORCH_AVAILABLE and isinstance(sig_tensor, torch.Tensor) and isinstance(base, torch.Tensor):
            # Ensure same dimensions
            if sig_tensor.shape != base.shape:
                # Reshape if possible, otherwise use base shape
                if base.numel() == sig_tensor.numel():
                    sig_tensor = sig_tensor.reshape(base.shape)
                else:
                    # Truncate or pad to match
                    if base.numel() < sig_tensor.numel():
                        sig_tensor = sig_tensor[:base.numel()]
                    else:
                        padding = torch.zeros(base.numel() - sig_tensor.numel())
                        sig_tensor = torch.cat([sig_tensor, padding])
            
            # Update formula: (stability * old) + (learning_factor * new)
            new_vec = (
                sig_tensor * self.stability +
                base * factor
            )
            
            # Normalize the result
            norm = torch.linalg.norm(new_vec)
            if norm > 0:
                new_vec = new_vec / norm
            
            # Store as list for consistency
            self.vector = new_vec.tolist()
        else:
            # Fallback for non-tensor case
            if isinstance(sig_tensor, list):
                sig_list = sig_tensor
            else:
                sig_list = list(sig_tensor) if hasattr(sig_tensor, '__iter__') else [sig_tensor]
            
            if isinstance(base, list):
                base_list = base
            else:
                base_list = list(base) if hasattr(base, '__iter__') else [base]
            
            # Ensure same length
            min_len = min(len(sig_list), len(base_list))
            sig_list = sig_list[:min_len]
            base_list = base_list[:min_len]
            
            # Update formula
            new_vec = [
                sig_list[i] * self.stability + base_list[i] * factor
                for i in range(min_len)
            ]
            
            # Normalize
            self.vector = normalize(new_vec)
        
        return self.vector
    
    def get(self):
        """
        Get the current signature vector.
        
        Returns:
            Current signature vector (list or tensor)
        """
        return self.vector
    
    def compute_drift(self, new_signature=None):
        """
        Compute drift from previous signature (optional diagnostic).
        
        Args:
            new_signature: Optional new signature to compare against current
            
        Returns:
            Drift value (0.0 = no drift, 1.0 = maximum drift)
        """
        if new_signature is None:
            return 0.0
        
        from .torch_utils import safe_cosine_similarity
        similarity = safe_cosine_similarity(self.vector, new_signature)
        return 1.0 - similarity if similarity is not None else 0.0

