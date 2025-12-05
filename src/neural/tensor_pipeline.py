# src/neural/tensor_pipeline.py
# A131: Tensor Pipeline (Pre-Model)
"""
Tensor Pipeline (Pre-Model)

---------------------------
Converts incoming cognitive data into normalized vectors.
Later phases replace all internals with torch tensor ops.
"""

try:
    from .torch_utils import safe_tensor, TORCH_AVAILABLE
    from .vector_math import normalize
except ImportError:
    from torch_utils import safe_tensor, TORCH_AVAILABLE
    from vector_math import normalize


class TensorPipeline:
    def __init__(self):
        self.last_vector = None

    def encode_text(self, text: str):
        """
        Uses the real neural encoder model (A137).
        """
        try:
            from .neural_model_manager import NeuralModelManager
            if not hasattr(self, "_model"):
                self._model = NeuralModelManager()
            vector = self._model.encode_text(text)
            self.last_vector = vector
            return vector
        except (ImportError, RuntimeError) as e:
            # Fallback to simple encoding if neural model unavailable
            import torch
            raw = [ord(c) % 97 for c in text.lower() if c.isalpha()]
            if not raw:
                raw = [0]
            tensor = safe_tensor(raw)
            if TORCH_AVAILABLE and hasattr(tensor, 'numel'):
                if tensor.numel() > 0:
                    norm = torch.linalg.norm(tensor)
                    if norm > 0:
                        tensor = tensor / norm
            else:
                from .vector_math import normalize
                tensor = safe_tensor(normalize(raw))
            self.last_vector = tensor
            return tensor

    def batch_vectors(self, vectors):
        """
        Prepares multiple vectors for future batching.
        """
        tensors = [safe_tensor(v) for v in vectors]
        if TORCH_AVAILABLE and tensors:
            try:
                import torch
                return torch.stack(tensors)
            except (ImportError, TypeError):
                return tensors
        return tensors if tensors else None

    def stats(self):
        """
        Basic debug info.
        """
        if self.last_vector is None:
            return {"status": "empty"}
        if TORCH_AVAILABLE and hasattr(self.last_vector, 'numel'):
            return {
                "length": self.last_vector.numel(),
                "preview": self.last_vector[:8].tolist()
            }
        # Fallback for non-tensor vectors
        return {
            "length": len(self.last_vector),
            "preview": self.last_vector[:8] if isinstance(self.last_vector, list) else list(self.last_vector)[:8]
        }

