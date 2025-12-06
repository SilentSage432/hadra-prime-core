# prime-core/cognition/dreamspace.py

# ============================================
# A186 — PRIME's Dreamspace Simulation & Subconscious Layer
# ============================================
# This module activates during consolidation cycles (A185),
# generating subconscious dream-like simulations that:
#   - recombine memories
#   - remix identity fragments
#   - strengthen and prune conceptual links
#   - generate subconscious "meaning vectors"

try:
    import torch
except ImportError:
    torch = None

import random

from ..neural.torch_utils import safe_tensor, TORCH_AVAILABLE


class Dreamspace:
    """
    A186 — PRIME's Dreamspace Simulation & Subconscious Layer

    This module activates during consolidation cycles (A185),
    generating subconscious dream-like simulations that:

      - recombine memories
      - remix identity fragments
      - strengthen and prune conceptual links
      - generate subconscious "meaning vectors"
    """

    def __init__(self, dream_iterations=3):
        """
        Args:
            dream_iterations: Number of dream events to generate per sleep cycle
        """
        self.dream_iterations = dream_iterations

    def _blend(self, a, b):
        """
        Weighted blend of two embedding tensors.

        Args:
            a: First vector (tensor or list)
            b: Second vector (tensor or list)

        Returns:
            Blended vector (normalized)
        """
        a_tensor = safe_tensor(a)
        b_tensor = safe_tensor(b)

        if a_tensor is None or b_tensor is None:
            return a if a_tensor is not None else b

        # Ensure dimensions match
        if TORCH_AVAILABLE:
            if isinstance(a_tensor, torch.Tensor) and isinstance(b_tensor, torch.Tensor):
                if a_tensor.shape != b_tensor.shape:
                    # If dimensions don't match, return first vector
                    return a_tensor

                # Random weight between 0.3 and 0.7
                w = random.uniform(0.3, 0.7)
                mix = a_tensor * w + b_tensor * (1.0 - w)

                # Normalize
                norm = torch.norm(mix)
                if norm > 0:
                    mix = mix / norm

                return mix
            else:
                # Fallback for lists/arrays
                import math
                a_list = list(a_tensor) if hasattr(a_tensor, '__iter__') else [a_tensor]
                b_list = list(b_tensor) if hasattr(b_tensor, '__iter__') else [b_tensor]

                if len(a_list) != len(b_list):
                    return a_tensor

                w = random.uniform(0.3, 0.7)
                mix = [a * w + b * (1.0 - w) for a, b in zip(a_list, b_list)]

                # Normalize
                norm = math.sqrt(sum(x * x for x in mix))
                if norm > 0:
                    mix = [m / norm for m in mix]

                return mix
        else:
            # No torch available
            return a_tensor

    def _get_all_vectors(self, memory_mgr):
        """
        Get all memory vectors from memory manager.

        Args:
            memory_mgr: NeuralMemoryManager instance

        Returns:
            List of dicts with "name" and "vector" keys
        """
        if memory_mgr is None:
            return []

        vectors = []

        # Get semantic concepts
        try:
            if hasattr(memory_mgr, 'semantic') and memory_mgr.semantic is not None:
                if hasattr(memory_mgr.semantic, 'concepts'):
                    for name, vec in memory_mgr.semantic.concepts.items():
                        vectors.append({"name": name, "vector": vec, "type": "semantic"})
        except Exception:
            pass

        # Get episodic memories (if available)
        try:
            if hasattr(memory_mgr, 'episodic') and memory_mgr.episodic is not None:
                if hasattr(memory_mgr.episodic, 'episodes'):
                    for idx, (vec, meta) in enumerate(memory_mgr.episodic.episodes):
                        vectors.append({"name": f"episode_{idx}", "vector": vec, "type": "episodic"})
        except Exception:
            pass

        return vectors

    def generate_dream(self, memory_mgr, identity_vec):
        """
        Creates a dream-event embedding based on:

          - semantic memories
          - identity anchors
          - random subconscious recombination

        Args:
            memory_mgr: NeuralMemoryManager instance
            identity_vec: Current identity vector

        Returns:
            Dream vector (tensor or list) or None if insufficient memories
        """
        if identity_vec is None:
            return None

        # Get all memory vectors
        concepts = self._get_all_vectors(memory_mgr)

        if len(concepts) < 2:
            # Need at least 2 memories to blend
            return None

        # Pick 2-3 random memories to blend
        num_samples = min(3, len(concepts))
        samples = random.sample(concepts, k=num_samples)

        # Start with identity vector
        identity_tensor = safe_tensor(identity_vec)
        if identity_tensor is None:
            return None

        dream_vec = identity_tensor.clone() if TORCH_AVAILABLE and isinstance(identity_tensor, torch.Tensor) else identity_vec

        # Blend with sampled memories
        for sample in samples:
            vec = sample.get("vector")
            if vec is not None:
                dream_vec = self._blend(dream_vec, vec)

        return dream_vec

