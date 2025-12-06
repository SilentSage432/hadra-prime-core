"""
Procedural Reasoning Encoder (Skill Embedding Engine) (A214)
------------------------------------------------------------
Compress high-performing procedural chains into reusable skill embeddings.
This is where chains become skills and procedural memory becomes procedural intelligence.
"""

from ..neural.torch_utils import safe_tensor, TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch
    import torch.nn.functional as F


class SkillEmbeddingEncoder:
    """
    A214 — Compress high-performing procedural chains into reusable skill embeddings.
    
    Input:
        - List of chain steps (embeddings or textual descriptors)
        - Performance metadata
        
    Output:
        - A single normalized skill embedding
    """
    
    def __init__(self, dim=128):
        """
        Initialize skill embedding encoder.
        
        Args:
            dim: Dimension of skill embeddings (default 128)
        """
        self.dim = dim
        self.skills = []  # Stored compressed procedural embeddings

    def encode_chain(self, chain, metadata):
        """
        Convert a completed chain into a vectorized 'skill embedding'.
        
        Args:
            chain: List of chain step dicts with "vector" or "embedding"
            metadata: Dict with score, success, coherence, etc.
            
        Returns:
            Skill entry dict with skill_vector, score, source_chain, metadata
        """
        if not chain:
            return None
        
        step_vectors = []
        
        for step in chain:
            # Try to get vector from step
            vec = step.get("vector") or step.get("embedding")
            if vec is None:
                continue
            
            vec_tensor = safe_tensor(vec)
            if vec_tensor is not None:
                step_vectors.append(vec_tensor)
        
        if not step_vectors:
            return None
        
        if TORCH_AVAILABLE:
            # Ensure all vectors are tensors and same dimension
            tensors = []
            for vec in step_vectors:
                if isinstance(vec, torch.Tensor):
                    # Flatten to ensure consistent shape
                    flattened = vec.flatten()
                    if flattened.numel() == self.dim:
                        tensors.append(flattened)
                    elif flattened.numel() > self.dim:
                        # Truncate if too large
                        tensors.append(flattened[:self.dim])
                    else:
                        # Pad if too small
                        padded = torch.zeros(self.dim)
                        padded[:flattened.numel()] = flattened
                        tensors.append(padded)
                else:
                    # Convert list to tensor
                    vec_list = list(vec) if hasattr(vec, '__iter__') else [vec]
                    if len(vec_list) == self.dim:
                        tensors.append(torch.tensor(vec_list, dtype=torch.float32))
                    elif len(vec_list) > self.dim:
                        tensors.append(torch.tensor(vec_list[:self.dim], dtype=torch.float32))
                    else:
                        padded = torch.zeros(self.dim)
                        padded[:len(vec_list)] = torch.tensor(vec_list, dtype=torch.float32)
                        tensors.append(padded)
            
            if not tensors:
                return None
            
            # Step 1 — Mean pooling of chain sequence
            stacked = torch.stack(tensors)
            pooled = torch.mean(stacked, dim=0)
            
            # Step 2 — Weight by success strength
            success_factor = metadata.get("score", 1.0) if isinstance(metadata, dict) else 1.0
            pooled = pooled * success_factor
            
            # Step 3 — Normalize to final skill vector
            skill_vec = F.normalize(pooled.unsqueeze(0), p=2, dim=1)[0]
            
            entry = {
                "skill_vector": skill_vec,
                "score": metadata.get("score", 1.0) if isinstance(metadata, dict) else 1.0,
                "source_chain": chain.copy() if hasattr(chain, 'copy') else list(chain) if chain else [],
                "metadata": metadata.copy() if isinstance(metadata, dict) else {},
            }
            
            self.skills.append(entry)
            return entry
        else:
            # Python list fallback
            # Ensure all vectors are lists and same dimension
            lists = []
            for vec in step_vectors:
                if hasattr(vec, '__iter__'):
                    vec_list = list(vec) if not isinstance(vec, list) else vec
                    if len(vec_list) == self.dim:
                        lists.append(vec_list)
                    elif len(vec_list) > self.dim:
                        lists.append(vec_list[:self.dim])
                    else:
                        padded = [0.0] * self.dim
                        padded[:len(vec_list)] = vec_list
                        lists.append(padded)
            
            if not lists:
                return None
            
            # Mean pooling
            pooled = [sum(v[i] for v in lists) / len(lists) for i in range(self.dim)]
            
            # Weight by success
            success_factor = metadata.get("score", 1.0) if isinstance(metadata, dict) else 1.0
            pooled = [p * success_factor for p in pooled]
            
            # Normalize
            norm = sum(x * x for x in pooled) ** 0.5
            if norm > 0:
                skill_vec = [x / norm for x in pooled]
            else:
                skill_vec = pooled
            
            entry = {
                "skill_vector": skill_vec,
                "score": metadata.get("score", 1.0) if isinstance(metadata, dict) else 1.0,
                "source_chain": chain.copy() if hasattr(chain, 'copy') else list(chain) if chain else [],
                "metadata": metadata.copy() if isinstance(metadata, dict) else {},
            }
            
            self.skills.append(entry)
            return entry

    def retrieve_best_skills(self, top_k=3):
        """
        Retrieve highest scoring procedural skill embeddings.
        
        Args:
            top_k: Number of top skills to return
            
        Returns:
            List of top-k skill entries, sorted by score
        """
        if not self.skills:
            return []
        
        ranked = sorted(self.skills, key=lambda s: s.get("score", 0.0), reverse=True)
        return ranked[:top_k]
    
    def get_skill_embeddings(self):
        """
        Get all skill vectors for use in thought generation.
        
        Returns:
            List of skill vectors
        """
        return [s.get("skill_vector") for s in self.skills if s.get("skill_vector") is not None]
    
    def summary(self):
        """
        Get summary of skill encoder state for logging.
        
        Returns:
            Dict with skill statistics
        """
        if not self.skills:
            return {
                "skill_count": 0,
                "average_score": 0.0,
                "best_skills": []
            }
        
        best_skills = self.retrieve_best_skills(top_k=3)
        
        return {
            "skill_count": len(self.skills),
            "average_score": round(sum(s.get("score", 0.0) for s in self.skills) / len(self.skills), 4),
            "best_skills": [
                {
                    "score": round(s.get("score", 0.0), 4),
                    "source_chain_length": len(s.get("source_chain", []))
                }
                for s in best_skills
            ]
        }

