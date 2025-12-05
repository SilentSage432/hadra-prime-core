# prime-core/neural/neural_encoder.py

"""
Neural Embedding Encoder (A137)

-------------------------------

This is PRIME's first real neural encoder model.

It converts text tokens into dense vector embeddings.

Architecture:

- character-level tokenizer → indices

- embedding layer

- mean pooling

- projection layer

This will be upgraded in later phases.

"""

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

from .torch_utils import safe_tensor

class PrimeEmbeddingEncoder(nn.Module):

    def __init__(self, vocab_size=128, embed_dim=128, hidden_dim=128):

        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for PrimeEmbeddingEncoder. Please install torch.")

        super().__init__()

        self.embed_dim = embed_dim

        # Simple character-level embedding table

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Projection layer for smoothing representations

        self.proj = nn.Linear(embed_dim, hidden_dim)

        # Final normalization in vector space

        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, token_tensor):

        """

        token_tensor: a LongTensor of token indices.

        """

        embed = self.embedding(token_tensor)        # shape: [seq, embed_dim]

        pooled = embed.mean(dim=0)                  # mean pooling

        projected = self.proj(pooled)               # linear transformation

        normalized = self.layer_norm(projected)     # stable embedding

        return normalized  # final torch vector

    def encode_text(self, text: str):

        """

        Convert raw text → tokens → embedding vector.

        """

        # Convert characters to integer tokens

        tokens = [ord(c) % 128 for c in text.lower() if c.isalpha()]

        if len(tokens) == 0:

            tokens = [0]

        token_tensor = torch.tensor(tokens, dtype=torch.long)

        with torch.no_grad():

            vector = self.forward(token_tensor)

        # Return standard python list for storage/display

        return vector

