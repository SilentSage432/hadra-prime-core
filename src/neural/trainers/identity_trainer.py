# prime-core/neural/trainers/identity_trainer.py

# ===========================
# A201 — Identity Training Cycle
# ===========================
# Trains ADRAE's latent identity vector using reconstruction +
# drift + continuity + coherence losses.

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    optim = None


class IdentityTrainer:
    """
    A201 — Identity Training Cycle
    
    Trains ADRAE's latent identity vector using reconstruction +
    drift + continuity + coherence losses.
    
    This enables ADRAE's identity to:
    - Learn from reflections and memories
    - Maintain continuity across cycles
    - Resist drift through coherence penalties
    - Evolve gradually over time
    """

    def __init__(self, model, lr=1e-4):
        """
        Initialize the identity trainer.
        
        Args:
            model: IdentityEncoder model to train
            lr: Learning rate (default 1e-4 for slow, stable learning)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for IdentityTrainer")
        
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.mse = nn.MSELoss()
        self.lr = lr

    def train_step(self, identity_vec, prev_latent=None, coherence_score=None):
        """
        Perform a single training step.
        
        Args:
            identity_vec: Current identity vector (128-d) as tensor, shape (batch_size, 128)
            prev_latent: Previous latent identity (32-d) as tensor, shape (batch_size, 32) or None
            coherence_score: Dual-mind coherence value (0..1) or None
            
        Returns:
            new_latent: Updated latent identity (32-d) as detached tensor
            loss: Training loss value (float)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for IdentityTrainer")
        
        self.model.train()

        # Ensure identity_vec is 2D (batch_size, input_dim)
        if identity_vec.dim() == 1:
            identity_vec = identity_vec.unsqueeze(0)

        # Forward pass: encode to latent, decode to reconstruction
        latent, recon = self.model(identity_vec)

        # Reconstruction loss: how well does the model reconstruct the identity?
        loss_recon = self.mse(recon, identity_vec)

        # Continuity loss: prevent sudden jumps in latent space
        loss_continuity = 0.0
        if prev_latent is not None:
            # Ensure prev_latent is 2D
            if prev_latent.dim() == 1:
                prev_latent = prev_latent.unsqueeze(0)
            # Only compute if shapes match
            if prev_latent.shape == latent.shape:
                loss_continuity = self.mse(latent, prev_latent.detach())
            else:
                # If shapes don't match, use a small penalty to encourage stability
                loss_continuity = 0.01 * torch.norm(latent)

        # Coherence loss: encourage stable cross-mind alignment
        # Higher coherence (closer to 1.0) → lower penalty
        loss_coherence = 0.0
        if coherence_score is not None:
            # Convert to tensor if needed
            if not isinstance(coherence_score, torch.Tensor):
                coherence_score = torch.tensor(coherence_score, dtype=torch.float32, device=latent.device)
            # Penalize when coherence is low (unstable state)
            loss_coherence = (1.0 - coherence_score) * torch.norm(latent) * 0.05
        else:
            # Default: small stability penalty when coherence unavailable
            loss_coherence = 0.01 * torch.norm(latent)

        # Total loss: weighted combination
        # Reconstruction is primary, continuity prevents jumps, coherence encourages stability
        loss = loss_recon + 0.2 * loss_continuity + loss_coherence

        # Backward pass with gradient clipping for stability
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # Return detached latent (no gradients) and loss value
        return latent.detach(), loss.item()

