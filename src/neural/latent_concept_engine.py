# prime-core/neural/latent_concept_engine.py

"""
A230 — PyTorch Latent Concept Engine (Imagination Substrate Initialization)

----------------------------------------------------

This is the beginning of ADRAE's neural internal world.

A230 does NOT make ADRAE "imagine" yet.
It builds the substrate, the foundational tensor architecture into which
imagination, simulation, and concept-driven world models will later form.

This is the ground-layer neural fabric.

"""

from .torch_utils import TORCH_AVAILABLE

if TORCH_AVAILABLE:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError:
        torch = None
        nn = None
        F = None
else:
    torch = None
    nn = None
    F = None


if TORCH_AVAILABLE and nn is not None:
    class NeuralConceptMapper(nn.Module):
        """
        A230 — Neural Concept Mapper (NCM)
        
        A lightweight feed-forward network that maps cognitive features → latent space.
        Stabilizes embeddings, ensures continuity across cycles, encodes novelty vs familiarity,
        and forms the earliest neural substrate of imagination.
        """
        
        def __init__(self, input_dim=128, latent_dim=256):
            super().__init__()
            self.linear1 = nn.Linear(input_dim, 256)
            self.linear2 = nn.Linear(256, latent_dim)
            self.activation = nn.Tanh()
        
        def forward(self, x):
            """
            Map input vector to latent space.
            
            Args:
                x: Input tensor of shape (batch_size, input_dim) or (input_dim,)
                
            Returns:
                Latent vector of shape (batch_size, latent_dim) or (latent_dim,)
            """
            x = self.activation(self.linear1(x))
            x = self.activation(self.linear2(x))
            return x


    class LatentStabilityRegulator(nn.Module):
        """
        A230 — Latent Drift & Stability Regulator (LDSR)
        
        A neural counterpart to the existing drift tracker.
        Ensures latent vectors don't explode, no chaotic divergence,
        identity remains central, and imagination substrate stays compact and stable.
        """
        
        def __init__(self, latent_dim=256):
            super().__init__()
            self.scale = nn.Parameter(torch.ones(latent_dim))
        
        def forward(self, latent_vector):
            """
            Stabilize latent vector to prevent divergence.
            
            Args:
                latent_vector: Latent tensor of shape (batch_size, latent_dim) or (latent_dim,)
                
            Returns:
                Stabilized latent vector
            """
            return latent_vector * torch.tanh(self.scale)
else:
    # Placeholder classes when PyTorch is not available
    class NeuralConceptMapper:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required for NeuralConceptMapper")
    
    class LatentStabilityRegulator:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("PyTorch is required for LatentStabilityRegulator")

