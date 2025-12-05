# prime-core/neural/neural_model_manager.py

"""
Neural Model Manager

--------------------

Handles initialization, loading, and usage of PRIME's neural encoder.

"""

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from .neural_encoder import PrimeEmbeddingEncoder

class NeuralModelManager:

    def __init__(self):

        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for NeuralModelManager. Please install torch.")

        self.device = torch.device("cpu")  # CPU-only for now

        self.encoder = PrimeEmbeddingEncoder().to(self.device)

        self.encoder.eval()  # inference mode only

    def encode_text(self, text):

        """

        High-level interface for producing embeddings.

        """

        with torch.no_grad():

            return self.encoder.encode_text(text)

    def status(self):

        return {

            "model": "PrimeEmbeddingEncoder",

            "device": str(self.device),

            "embed_dim": self.encoder.embed_dim,

        }

