# prime-core/neural/reflective_thought_generator.py

"""
Reflective Thought Generator (A148)
-----------------------------------
Generates internal reflective thoughts based on:

- PRIME's cognitive fusion vector
- attention signal
- relevant semantic memories
- long-term identity vector
- drift & coherence signals

This produces "meaning vectors" that PRIME uses
for deep internal reflection during autonomous runtime.
"""

try:
    import torch
except ImportError:
    torch = None

from .torch_utils import safe_tensor, is_tensor, TORCH_AVAILABLE


class ReflectiveThoughtGenerator:

    def __init__(self, memory_influence=0.35, identity_influence=0.20, fusion_influence=0.45):
        self.memory_influence = memory_influence
        self.identity_influence = identity_influence
        self.fusion_influence = fusion_influence

    def generate(self, fusion_vec, attention_vec, timescales, memory_manager):
        """
        Generate a single reflective embedding that captures insight.
        """

        if fusion_vec is None:
            return None

        fusion_vec = safe_tensor(fusion_vec)

        # 1. Pull relevant semantic memories
        if memory_manager and hasattr(memory_manager, 'semantic'):
            sim_memories = memory_manager.semantic.find_similar(fusion_vec, top_k=3)
            # find_similar returns (score, name, vec) tuples
            mem_vectors = [m[2] for m in sim_memories] if sim_memories else []
        else:
            mem_vectors = []

        # Average memory concept vector
        if len(mem_vectors) > 0:
            if TORCH_AVAILABLE and all(isinstance(v, torch.Tensor) or isinstance(v, list) for v in mem_vectors):
                mem_tensors = [safe_tensor(v) for v in mem_vectors]
                if all(isinstance(t, torch.Tensor) for t in mem_tensors):
                    memory_context = torch.mean(torch.stack(mem_tensors), dim=0)
                else:
                    # Fallback: manual average
                    memory_context = self._manual_mean(mem_vectors)
                    if memory_context is None:
                        # Fallback to zero vector if mean calculation fails
                        if TORCH_AVAILABLE and isinstance(fusion_vec, torch.Tensor):
                            memory_context = torch.zeros_like(fusion_vec)
                        else:
                            memory_context = [0.0] * len(fusion_vec)
            else:
                # Fallback: manual average
                memory_context = self._manual_mean(mem_vectors)
                if memory_context is None:
                    # Fallback to zero vector if mean calculation fails
                    if TORCH_AVAILABLE and isinstance(fusion_vec, torch.Tensor):
                        memory_context = torch.zeros_like(fusion_vec)
                    else:
                        memory_context = [0.0] * len(fusion_vec)
        else:
            # Create zero vector matching fusion_vec shape
            if TORCH_AVAILABLE and isinstance(fusion_vec, torch.Tensor):
                memory_context = torch.zeros_like(fusion_vec)
            else:
                memory_context = [0.0] * len(fusion_vec)

        # 2. Identity vector
        if timescales and hasattr(timescales, 'identity_vector') and timescales.identity_vector is not None:
            identity = safe_tensor(timescales.identity_vector)
        else:
            if TORCH_AVAILABLE and isinstance(fusion_vec, torch.Tensor):
                identity = torch.zeros_like(fusion_vec)
            else:
                identity = [0.0] * len(fusion_vec)

        # 3. Weighted combination → Reflection vector
        if TORCH_AVAILABLE and isinstance(fusion_vec, torch.Tensor):
            reflective = (
                self.fusion_influence * fusion_vec +
                self.memory_influence * memory_context +
                self.identity_influence * identity
            )

            # normalize for consistency
            norm = torch.norm(reflective)
            if norm > 0:
                reflective = reflective / norm
        else:
            # Fallback: manual weighted combination
            reflective = [
                self.fusion_influence * f + 
                self.memory_influence * m + 
                self.identity_influence * i
                for f, m, i in zip(fusion_vec, memory_context, identity)
            ]
            
            # Normalize
            norm = sum(x * x for x in reflective) ** 0.5
            if norm > 0:
                reflective = [x / norm for x in reflective]

        return reflective

    def _manual_mean(self, vectors):
        """
        Compute mean of vectors manually (fallback when torch unavailable).
        """
        if not vectors:
            return None
        
        # Convert all to lists
        list_vectors = []
        for v in vectors:
            if isinstance(v, list):
                list_vectors.append(v)
            elif TORCH_AVAILABLE and isinstance(v, torch.Tensor):
                list_vectors.append(v.tolist())
            else:
                list_vectors.append(list(v))
        
        if not list_vectors:
            return None
        
        # Compute mean
        dim = len(list_vectors[0])
        mean = [0.0] * dim
        for vec in list_vectors:
            for i, val in enumerate(vec):
                mean[i] += val
        
        return [x / len(list_vectors) for x in mean]

