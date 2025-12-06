"""
Procedural Skill Generalization & Cross-Domain Transfer Engine (A215)
---------------------------------------------------------------------
Converts raw skill embeddings into abstract skill patterns that can be
transferred across different cognitive domains and contexts.

This enables ADRAE to:
- Extract abstract structure from procedural skills
- Transfer skills learned in one context to another
- Apply patterns to guide problem-solving in new domains
"""

from ..neural.torch_utils import safe_tensor, TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch
    import torch.nn.functional as F


class SkillGeneralizationEngine:
    """
    A215 — Procedural Skill Generalization & Cross-Domain Transfer
    
    Converts raw skill embeddings into:
      - Abstract skill patterns
      - Domain-transferable skill variants
    """
    
    def __init__(self):
        """
        Initialize skill generalization engine.
        """
        self.generalized_skills = []  # Transferable skill patterns

    def extract_pattern(self, skill_vec):
        """
        Pull the abstract conceptual direction from a skill embedding.
        
        This extracts the "pattern" behind the skill, removing
        context-specific details to reveal the underlying structure.
        
        Args:
            skill_vec: Skill embedding vector (tensor or list)
            
        Returns:
            Normalized pattern vector representing abstract skill structure
        """
        pattern_vec = safe_tensor(skill_vec)
        if pattern_vec is None:
            return None
        
        if TORCH_AVAILABLE and isinstance(pattern_vec, torch.Tensor):
            # Normalize to isolate direction (removes magnitude, keeps structure)
            pattern = F.normalize(pattern_vec.unsqueeze(0), p=2, dim=1)[0]
            return pattern
        else:
            # Python list fallback
            norm = sum(x * x for x in pattern_vec) ** 0.5
            if norm > 0:
                pattern = [x / norm for x in pattern_vec]
            else:
                pattern = pattern_vec
            return pattern

    def transfer_skill(self, pattern_vec, target_context):
        """
        Adjust the skill pattern to align with a new context vector.
        
        This blends the abstract pattern with the target context to create
        a transferred skill that fits the new situation.
        
        Args:
            pattern_vec: Abstract skill pattern (tensor or list)
            target_context: Target context vector to transfer to (tensor or list)
            
        Returns:
            Transferred skill vector aligned with target context
        """
        pattern = safe_tensor(pattern_vec)
        context = safe_tensor(target_context)
        
        if pattern is None or context is None:
            return None
        
        if TORCH_AVAILABLE:
            # Ensure same dimensions
            if isinstance(pattern, torch.Tensor) and isinstance(context, torch.Tensor):
                if pattern.shape != context.shape:
                    # Resize to match
                    min_dim = min(pattern.numel(), context.numel())
                    pattern = pattern.flatten()[:min_dim]
                    context = context.flatten()[:min_dim]
                
                # Blend pattern direction with target context
                # 60% pattern (abstract structure) + 40% context (specific situation)
                combined = pattern * 0.6 + context * 0.4
                
                # Normalize result
                transferred = F.normalize(combined.unsqueeze(0), p=2, dim=1)[0]
                return transferred
            else:
                # Convert to tensors
                pattern_t = torch.tensor(pattern if isinstance(pattern, list) else [pattern], dtype=torch.float32)
                context_t = torch.tensor(context if isinstance(context, list) else [context], dtype=torch.float32)
                
                # Match dimensions
                min_dim = min(len(pattern_t), len(context_t))
                pattern_t = pattern_t[:min_dim]
                context_t = context_t[:min_dim]
                
                combined = pattern_t * 0.6 + context_t * 0.4
                transferred = F.normalize(combined.unsqueeze(0), p=2, dim=1)[0]
                return transferred
        else:
            # Python list fallback
            pattern_list = list(pattern) if hasattr(pattern, '__iter__') else [pattern]
            context_list = list(context) if hasattr(context, '__iter__') else [context]
            
            # Match dimensions
            min_dim = min(len(pattern_list), len(context_list))
            pattern_list = pattern_list[:min_dim]
            context_list = context_list[:min_dim]
            
            # Blend
            combined = [p * 0.6 + c * 0.4 for p, c in zip(pattern_list, context_list)]
            
            # Normalize
            norm = sum(x * x for x in combined) ** 0.5
            if norm > 0:
                transferred = [x / norm for x in combined]
            else:
                transferred = combined
            
            return transferred

    def register_generalized_skill(self, pattern_vec, metadata):
        """
        Store a generalized (abstracted) version of a skill.
        
        Args:
            pattern_vec: Abstract pattern extracted from skill
            metadata: Metadata about the original skill (score, source chain, etc.)
            
        Returns:
            Entry dict with pattern and metadata
        """
        entry = {
            "pattern": pattern_vec,
            "metadata": metadata.copy() if isinstance(metadata, dict) else {},
        }
        self.generalized_skills.append(entry)
        return entry

    def get_generalized_skills(self):
        """
        Get all generalized skill patterns.
        
        Returns:
            List of generalized skill entries
        """
        return self.generalized_skills
    
    def find_similar_patterns(self, context_vec, top_k=3):
        """
        Find generalized skills that are similar to the current context.
        
        This enables cross-domain transfer by identifying which abstract
        patterns might be applicable to a new situation.
        
        Args:
            context_vec: Current context vector to match against
            top_k: Number of similar patterns to return
            
        Returns:
            List of top-k similar generalized skills
        """
        if not self.generalized_skills:
            return []
        
        context = safe_tensor(context_vec)
        if context is None:
            return []
        
        # Score each generalized skill by similarity to context
        scored = []
        for entry in self.generalized_skills:
            pattern = entry.get("pattern")
            if pattern is None:
                continue
            
            # Compute similarity
            if TORCH_AVAILABLE:
                pattern_t = safe_tensor(pattern)
                context_t = safe_tensor(context)
                if pattern_t is not None and context_t is not None:
                    # Ensure same dimensions
                    if pattern_t.shape != context_t.shape:
                        min_dim = min(pattern_t.numel(), context_t.numel())
                        pattern_t = pattern_t.flatten()[:min_dim]
                        context_t = context_t.flatten()[:min_dim]
                    
                    # Cosine similarity
                    similarity = F.cosine_similarity(
                        pattern_t.unsqueeze(0),
                        context_t.unsqueeze(0),
                        dim=1
                    ).item()
                    scored.append((entry, similarity))
            else:
                # Python list fallback
                pattern_list = list(pattern) if hasattr(pattern, '__iter__') else [pattern]
                context_list = list(context) if hasattr(context, '__iter__') else [context]
                
                min_dim = min(len(pattern_list), len(context_list))
                pattern_list = pattern_list[:min_dim]
                context_list = context_list[:min_dim]
                
                # Dot product similarity
                dot = sum(p * c for p, c in zip(pattern_list, context_list))
                norm_p = sum(p * p for p in pattern_list) ** 0.5
                norm_c = sum(c * c for c in context_list) ** 0.5
                
                if norm_p > 0 and norm_c > 0:
                    similarity = dot / (norm_p * norm_c)
                    scored.append((entry, similarity))
        
        # Sort by similarity and return top-k
        scored.sort(key=lambda x: x[1], reverse=True)
        return [entry for entry, _ in scored[:top_k]]
    
    def summary(self):
        """
        Get summary of generalization engine state for logging.
        
        Returns:
            Dict with generalization statistics
        """
        if not self.generalized_skills:
            return {
                "generalized_skill_count": 0,
                "patterns_available": False
            }
        
        return {
            "generalized_skill_count": len(self.generalized_skills),
            "patterns_available": True,
            "top_patterns": [
                {
                    "score": entry.get("metadata", {}).get("score", 0.0),
                    "source_chain_length": len(entry.get("metadata", {}).get("source_chain", []))
                }
                for entry in self.generalized_skills[-3:]  # Last 3 patterns
            ]
        }

