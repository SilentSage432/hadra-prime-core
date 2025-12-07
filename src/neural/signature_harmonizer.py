# prime-core/neural/signature_harmonizer.py

"""
A222 — Signature-Guided Thought Harmonization Layer

---------------------------------------------------

Ensures that every thought (candidate, selected, or reflective)
gradually aligns with ADRAE's emergent identity signature.

This acts as a soft attractor mechanism that stabilizes the
internal cognitive universe around a consistent semantic core.

"""

from .torch_utils import safe_tensor, safe_cosine_similarity, TORCH_AVAILABLE
from .vector_math import normalize

if TORCH_AVAILABLE:
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        torch = None
        F = None
else:
    torch = None
    F = None


class SignatureHarmonizer:
    """
    A222 — Signature-Guided Thought Harmonization Layer
    
    Pulls thoughts toward ADRAE's identity signature to maintain
    cognitive coherence and prevent fragmentation.
    """
    
    def __init__(self, strength=0.15):
        """
        Initialize the signature harmonizer.
        
        Args:
            strength: Harmonization strength (0.0 = no influence, 1.0 = identity dominates)
                     Default: 0.15 (15% pull toward signature)
        """
        self.strength = strength
        self.last_signature = None
    
    def update_signature(self, identity_vectors):
        """
        Compute a stable averaged signature from multiple identity embeddings.
        
        Args:
            identity_vectors: List of identity vector embeddings
            
        Returns:
            Normalized signature vector or None
        """
        if not identity_vectors:
            return None
        
        # Filter out None values
        valid_vectors = [v for v in identity_vectors if v is not None]
        if not valid_vectors:
            return None
        
        # Convert all to tensors for computation
        vectors_t = [safe_tensor(v) for v in valid_vectors]
        
        if TORCH_AVAILABLE and torch is not None:
            # Try torch-based averaging
            try:
                # Filter to only torch tensors
                torch_vectors = [v for v in vectors_t if isinstance(v, torch.Tensor)]
                if torch_vectors:
                    # Ensure all have same shape
                    target_shape = torch_vectors[0].shape
                    aligned_vectors = []
                    for v in torch_vectors:
                        if v.shape == target_shape:
                            aligned_vectors.append(v)
                        elif v.numel() == target_shape[0] if len(target_shape) == 1 else v.numel() == target_shape[0] * target_shape[1]:
                            # Reshape if same size
                            aligned_vectors.append(v.reshape(target_shape))
                    
                    if aligned_vectors:
                        stacked = torch.stack(aligned_vectors)
                        sig = torch.mean(stacked, dim=0)
                        # Normalize
                        norm = torch.linalg.norm(sig)
                        if norm > 0:
                            sig = sig / norm
                        self.last_signature = sig.tolist()
                        return self.last_signature
            except Exception:
                pass
        
        # Fallback: list-based averaging
        try:
            # Convert all to lists
            list_vectors = []
            for v in vectors_t:
                if isinstance(v, list):
                    list_vectors.append(v)
                elif TORCH_AVAILABLE and isinstance(v, torch.Tensor):
                    list_vectors.append(v.tolist())
                elif hasattr(v, '__iter__'):
                    list_vectors.append(list(v))
            
            if not list_vectors:
                return None
            
            # Find common length (use shortest)
            min_len = min(len(v) for v in list_vectors if len(v) > 0)
            if min_len == 0:
                return None
            
            # Truncate all to same length
            aligned = [v[:min_len] for v in list_vectors]
            
            # Average
            import math
            sig = [sum(v[i] for v in aligned) / len(aligned) for i in range(min_len)]
            
            # Normalize
            norm = math.sqrt(sum(x * x for x in sig))
            if norm > 0:
                sig = [x / norm for x in sig]
            
            self.last_signature = sig
            return sig
        except Exception:
            return None
    
    def harmonize(self, vector):
        """
        Pull the input vector slightly toward the signature vector.
        
        This creates a soft attractor effect that gradually aligns
        all thoughts with ADRAE's identity signature.
        
        Args:
            vector: Input thought vector to harmonize
            
        Returns:
            Harmonized vector (pulled toward signature) or original if no signature
        """
        if self.last_signature is None or vector is None:
            return vector
        
        base = safe_tensor(vector)
        sig_tensor = safe_tensor(self.last_signature)
        
        if base is None or sig_tensor is None:
            return vector
        
        if TORCH_AVAILABLE and torch is not None:
            try:
                # Try torch-based harmonization
                if isinstance(base, torch.Tensor) and isinstance(sig_tensor, torch.Tensor):
                    # Ensure same shape
                    if base.shape == sig_tensor.shape:
                        # Harmonize: blend base and signature
                        combined = (1 - self.strength) * base + self.strength * sig_tensor
                        # Normalize
                        norm = torch.linalg.norm(combined)
                        if norm > 0:
                            combined = combined / norm
                        return combined.tolist()
                    elif base.numel() == sig_tensor.numel():
                        # Same size, different shape - flatten and reshape
                        base_flat = base.flatten()
                        sig_flat = sig_tensor.flatten()
                        combined_flat = (1 - self.strength) * base_flat + self.strength * sig_flat
                        norm = torch.linalg.norm(combined_flat)
                        if norm > 0:
                            combined_flat = combined_flat / norm
                        return combined_flat.reshape(base.shape).tolist()
            except Exception:
                pass
        
        # Fallback: list-based harmonization
        try:
            # Convert to lists
            if isinstance(base, list):
                base_list = base
            elif TORCH_AVAILABLE and isinstance(base, torch.Tensor):
                base_list = base.tolist()
            elif hasattr(base, '__iter__'):
                base_list = list(base)
            else:
                return vector
            
            if isinstance(sig_tensor, list):
                sig_list = sig_tensor
            elif TORCH_AVAILABLE and isinstance(sig_tensor, torch.Tensor):
                sig_list = sig_tensor.tolist()
            elif hasattr(sig_tensor, '__iter__'):
                sig_list = list(sig_tensor)
            else:
                return vector
            
            # Ensure same length
            min_len = min(len(base_list), len(sig_list))
            if min_len == 0:
                return vector
            
            base_list = base_list[:min_len]
            sig_list = sig_list[:min_len]
            
            # Harmonize: blend base and signature
            combined = [
                (1 - self.strength) * b + self.strength * s
                for b, s in zip(base_list, sig_list)
            ]
            
            # Normalize
            import math
            norm = math.sqrt(sum(c * c for c in combined))
            if norm > 0:
                combined = [c / norm for c in combined]
            
            return combined
        except Exception:
            return vector

