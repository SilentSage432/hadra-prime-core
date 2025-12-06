# prime-core/cognition/personality_signature_engine.py

"""
A167 — Emergent Personality Signature Stabilizer
------------------------------------------------

This module maintains PRIME's long-term personality signature: a stable vector
derived from identity seeds, gradient learning, and long-term cognitive trends.

The signature:
- persists across runtime
- shapes cognition, evolution, and reflection
- stabilizes PRIME's "self-pattern" over time
"""

try:
    import torch
except:
    torch = None


class PersonalitySignatureEngine:

    def __init__(self, blend_rate=0.02):
        # Signature vector (long-term identity field)
        self.signature = None

        # Small update rate keeps signature stable
        self.blend_rate = blend_rate

    def update(self, personality_vec):
        """
        Blend new personality signals into the signature over time.
        """
        if personality_vec is None:
            return self.signature

        # First-time initialization
        if self.signature is None:
            # Clone or copy to avoid reference issues
            if torch is not None and isinstance(personality_vec, torch.Tensor):
                self.signature = personality_vec.clone()
            else:
                if isinstance(personality_vec, list):
                    self.signature = personality_vec[:]
                else:
                    self.signature = personality_vec
            return self.signature

        try:
            # Torch version
            if torch is not None and isinstance(personality_vec, torch.Tensor):
                # Ensure signature is also a tensor
                if not isinstance(self.signature, torch.Tensor):
                    self.signature = torch.tensor(self.signature, dtype=personality_vec.dtype)
                
                # Ensure same dimensions
                if self.signature.shape == personality_vec.shape:
                    updated = (
                        (1 - self.blend_rate) * self.signature
                        + self.blend_rate * personality_vec
                    )
                    norm = torch.norm(updated)
                    if norm > 0:
                        self.signature = updated / norm
                    else:
                        self.signature = updated
            else:
                # Python version
                # Convert to lists if needed
                if torch is not None and isinstance(personality_vec, torch.Tensor):
                    personality_vec = personality_vec.tolist()
                if torch is not None and isinstance(self.signature, torch.Tensor):
                    self.signature = self.signature.tolist()
                
                if isinstance(self.signature, list) and isinstance(personality_vec, list):
                    if len(self.signature) == len(personality_vec):
                        new_vec = []
                        for s, p in zip(self.signature, personality_vec):
                            new_vec.append((1 - self.blend_rate) * s + self.blend_rate * p)

                        # Normalize
                        import math
                        norm = math.sqrt(sum(x*x for x in new_vec))
                        if norm > 0:
                            new_vec = [x / norm for x in new_vec]
                        self.signature = new_vec
        except Exception:
            # If update fails, keep existing signature
            pass

        return self.signature

    def apply(self, vector):
        """
        Applies personality signature shaping to any cognitive vector.
        """
        if vector is None or self.signature is None:
            return vector

        try:
            if torch is not None and isinstance(vector, torch.Tensor):
                # Ensure signature is also a tensor
                sig_tensor = self.signature
                if not isinstance(sig_tensor, torch.Tensor):
                    sig_tensor = torch.tensor(sig_tensor, dtype=vector.dtype)
                
                # Ensure same dimensions
                if vector.shape == sig_tensor.shape:
                    merged = 0.9 * vector + 0.1 * sig_tensor
                    norm = torch.norm(merged)
                    if norm > 0:
                        return merged / norm
                    return merged
                else:
                    return vector

            # Python fallback
            # Convert to lists if needed
            if torch is not None and isinstance(vector, torch.Tensor):
                vector = vector.tolist()
            if torch is not None and isinstance(self.signature, torch.Tensor):
                sig_list = self.signature.tolist()
            else:
                sig_list = list(self.signature) if not isinstance(self.signature, list) else self.signature
            
            vec_list = list(vector) if not isinstance(vector, list) else vector
            
            if len(vec_list) == len(sig_list):
                merged = [
                    (0.9 * v) + (0.1 * s)
                    for v, s in zip(vec_list, sig_list)
                ]

                import math
                norm = math.sqrt(sum(x*x for x in merged))
                if norm > 0:
                    merged = [x / norm for x in merged]

                return merged
            else:
                return vector
        except Exception:
            return vector

