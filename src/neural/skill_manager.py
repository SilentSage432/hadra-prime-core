"""
Skill Manager (A217)
-------------------
Manages ADRAE's self-generated skill vectors that help reduce uncertainty.

Skills are embeddings created automatically when uncertainty is detected,
representing capabilities ADRAE needs to develop.
"""

from .torch_utils import safe_tensor, TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch


class SkillManager:
    """
    A217 — Manages self-generated skill vectors that help reduce uncertainty.
    
    Skills are created automatically when uncertainty is high, and they
    influence future thought selection, action selection, and reflection paths.
    """
    
    def __init__(self):
        """
        Initialize skill manager with empty skill bank.
        """
        self.skills = []  # List of { "name": str, "vec": tensor/list, "metadata": dict }

    def add_skill(self, name, vec, metadata=None):
        """
        Add a new skill vector to the skill bank.
        
        Args:
            name: Skill name/identifier
            vec: Skill vector (tensor or list)
            metadata: Optional metadata about the skill (uncertainty level, context, etc.)
        """
        if vec is None:
            return None
        
        vec_tensor = safe_tensor(vec)
        if vec_tensor is None:
            return None
        
        # Clone and detach if tensor to avoid reference issues
        if TORCH_AVAILABLE and isinstance(vec_tensor, torch.Tensor):
            vec_tensor = vec_tensor.clone().detach()
        
        skill_entry = {
            "name": name,
            "vec": vec_tensor,
            "metadata": metadata or {}
        }
        
        self.skills.append(skill_entry)
        return skill_entry

    def get_all_skill_vectors(self):
        """
        Get all skill vectors for use in thought selection and action biasing.
        
        Returns:
            List of skill vectors
        """
        return [s["vec"] for s in self.skills if s.get("vec") is not None]
    
    def get_skills_by_name(self, name_pattern):
        """
        Get skills matching a name pattern.
        
        Args:
            name_pattern: String pattern to match (e.g., "skill_auto_")
            
        Returns:
            List of matching skill entries
        """
        return [s for s in self.skills if name_pattern in s.get("name", "")]
    
    def get_skill_count(self):
        """
        Get total number of skills in the bank.
        
        Returns:
            int: Number of skills
        """
        return len(self.skills)
    
    def status(self):
        """
        Get status summary of skill bank.
        
        Returns:
            Dict with skill count and names
        """
        return {
            "count": len(self.skills),
            "skills": [s["name"] for s in self.skills],
            "skill_details": [
                {
                    "name": s["name"],
                    "metadata": s.get("metadata", {})
                }
                for s in self.skills
            ]
        }
    
    def clear_skills(self):
        """
        Clear all skills from the bank (for testing/reset).
        """
        self.skills = []
    
    def remove_skill(self, name):
        """
        Remove a skill by name.
        
        Args:
            name: Name of skill to remove
            
        Returns:
            bool: True if skill was removed, False if not found
        """
        initial_count = len(self.skills)
        self.skills = [s for s in self.skills if s.get("name") != name]
        return len(self.skills) < initial_count

