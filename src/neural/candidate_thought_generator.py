# prime-core/neural/candidate_thought_generator.py

"""
Candidate Thought Generator (A146)
----------------------------------
Produces multiple potential internal thoughts for PRIME to consider.

This uses:
- the fusion vector (current cognitive state)
- the attention vector
- slight noise perturbations
- small directional shifts
- conceptual proximity to identity

Outputs:
- A list of N candidate embedding vectors for A144 to evaluate
"""

try:
    import torch
except ImportError:
    torch = None

import random
from .torch_utils import safe_tensor, is_tensor, safe_norm, TORCH_AVAILABLE


class CandidateThoughtGenerator:

    def __init__(self, num_candidates=4, noise_scale=0.05, bridge=None):
        self.num_candidates = num_candidates
        self.noise_scale = noise_scale
        self.bridge = bridge  # Optional bridge reference for accessing seed embeddings

    def generate_variants(self, base_vector):
        """
        Create slight variations on a base vector by adding controlled noise
        and direction shifts. This simulates conceptual exploration.
        """
        if base_vector is None:
            return []

        base = safe_tensor(base_vector)
        candidates = []

        for _ in range(self.num_candidates):
            if TORCH_AVAILABLE and isinstance(base, torch.Tensor):
                # noise: small Gaussian perturbation
                noise = torch.randn_like(base) * self.noise_scale
                candidate = base + noise

                # normalize vector for consistency
                norm = torch.norm(candidate)
                if norm > 0:
                    candidate = candidate / norm
                candidates.append(candidate)
            else:
                # Fallback: manual noise and normalization
                noise = [random.gauss(0, self.noise_scale) for _ in base]
                candidate = [b + n for b, n in zip(base, noise)]
                
                # normalize vector for consistency
                norm = safe_norm(candidate)
                if norm > 0:
                    candidate = [c / norm for c in candidate]
                candidates.append(candidate)

        return candidates

    def propose(self, fusion_vec, attention_vec, identity_vec):
        """
        Combines signals to propose candidate thoughts:

        1. Start with fusion vector (core cognitive state)
        2. Add attention-directed variant
        3. Add identity-directed variant
        4. Add random exploration variants
        5. Include seed embeddings if available via bridge
        """
        proposals = []

        if fusion_vec is not None:
            proposals += self.generate_variants(fusion_vec)

        if attention_vec is not None:
            proposals += self.generate_variants(attention_vec)

        if identity_vec is not None:
            proposals += self.generate_variants(identity_vec)

        # ----------------------------------------------------
        # FINAL FIX: Inject seed embeddings as candidate thoughts
        # ----------------------------------------------------
        # Get seed embeddings from bridge if available
        seed_embeddings = None
        if self.bridge and hasattr(self.bridge.state, "seed_embeddings"):
            seed_embeddings = self.bridge.state.seed_embeddings
        
        if seed_embeddings:
            for item in seed_embeddings:
                try:
                    emb = item.get("embedding")
                    if emb is not None:
                        proposals.append(emb)
                except Exception as e:
                    print("⚠️ Seed injection error in CandidateThoughtGenerator:", e)

        # Safety: ensure at least something is returned
        if len(proposals) == 0:
            # fallback: generate variants from fusion if available
            if fusion_vec is not None:
                proposals += self.generate_variants(fusion_vec)
            # If still empty, create a minimal random vector
            if len(proposals) == 0:
                try:
                    import torch
                    if TORCH_AVAILABLE:
                        # Create a random normalized vector as fallback
                        fallback = torch.randn(128)  # Default embedding size
                        proposals.append(fallback / torch.norm(fallback))
                except:
                    pass

        return proposals

