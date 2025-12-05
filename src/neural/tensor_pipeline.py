# src/neural/tensor_pipeline.py
# A131: Tensor Pipeline (Pre-Model)
"""
Tensor Pipeline (Pre-Model)

---------------------------
Converts incoming cognitive data into normalized vectors.
Later phases replace all internals with torch tensor ops.
"""

try:
    from .torch_utils import safe_tensor
    from .vector_math import normalize
except ImportError:
    from torch_utils import safe_tensor
    from vector_math import normalize


class TensorPipeline:
    def __init__(self):
        self.last_vector = None

    def encode_text(self, text: str):
        """
        TEMPORARY ENCODER:
        Converts text → simple numeric vector (character ordinals).
        Replaced by PyTorch encoder in phases A136–A140.
        """
        raw = [ord(c) % 97 for c in text.lower() if c.isalpha()]
        vector = normalize(raw)
        self.last_vector = vector
        return safe_tensor(vector)

    def batch_vectors(self, vectors):
        """
        Prepares multiple vectors for future batching.
        """
        return [safe_tensor(v) for v in vectors]

    def stats(self):
        """
        Basic debug info.
        """
        if not self.last_vector:
            return {"status": "empty"}
        return {
            "length": len(self.last_vector),
            "preview": self.last_vector[:8],
        }

