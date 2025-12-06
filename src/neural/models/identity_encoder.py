# prime-core/neural/models/identity_encoder.py

# ===========================
# A200 — Neural Identity Encoder
# ===========================
# Learns a compact latent representation of ADRAE's identity state.

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None


class IdentityEncoder(nn.Module if TORCH_AVAILABLE else object):
    """
    A200 — Neural Identity Encoder
    
    Learns a compact latent representation of ADRAE's identity state.
    
    Architecture:
    - Input: 128-dim identity, memories, reflections
    - Encoder: 128 → 64 → 32 (latent)
    - Decoder: 32 → 64 → 128
    - Output: reconstructed identity vector
    
    The 32-dimensional latent vector is ADRAE's "soul signature" —
    a compressed, learnable representation of her identity.
    """

    def __init__(self, input_dim=128, latent_dim=32):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for IdentityEncoder")
        
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, latent_dim),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        """
        Forward pass: encode to latent, then decode back.
        
        Args:
            x: Input identity vector (batch_size, input_dim)
            
        Returns:
            z: Latent identity representation (batch_size, latent_dim)
            recon: Reconstructed identity vector (batch_size, input_dim)
        """
        z = self.encoder(x)
        recon = self.decoder(z)
        return z, recon

    def encode(self, x):
        """
        Encode identity vector to latent representation.
        
        Args:
            x: Input identity vector (can be 1D or 2D tensor)
            
        Returns:
            Latent identity representation
        """
        # Ensure x is 2D (batch_size, input_dim)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        return self.encoder(x)

