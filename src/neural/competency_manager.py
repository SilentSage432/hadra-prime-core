"""
Competency Manager (A218)
--------------------------
Organizes skill vectors into competency clusters that represent
functional domains of expertise within ADRAE's cognitive architecture.

Each cluster becomes a "competency node" - a proto-intelligence module
that influences thought selection, goals, memory recall, and stability.
"""

from .torch_utils import safe_tensor, safe_cosine_similarity, TORCH_AVAILABLE

if TORCH_AVAILABLE:
    import torch
    import torch.nn.functional as F


class CompetencyManager:
    """
    A218 — Manages competency clusters formed from skill vectors.
    
    Clusters represent functional domains like:
    - Memory Competency Cluster
    - Attention Stability Cluster
    - Identity Integrity Cluster
    - Goal Selection Cluster
    - Uncertainty Reduction Cluster
    - Reflection Optimization Cluster
    """
    
    def __init__(self, similarity_threshold=0.65):
        """
        Initialize competency manager.
        
        Args:
            similarity_threshold: Cosine similarity threshold for clustering (default 0.65)
        """
        self.clusters = {}  # name -> { "centroid": tensor, "members": [tensors], "metadata": dict }
        self.similarity_threshold = similarity_threshold

    def cluster_skills(self, skills):
        """
        Cluster skill vectors by latent similarity.
        
        Uses cosine similarity to group similar skills into competency clusters.
        Each cluster becomes a persistent module with a centroid.
        
        Args:
            skills: List of (name, vec) tuples representing skills to cluster
        """
        if not skills:
            return
        
        for name, vec in skills:
            if vec is None:
                continue
            
            vec_tensor = safe_tensor(vec)
            if vec_tensor is None:
                continue
            
            assigned = False
            
            # Try to assign to existing cluster
            for cname, cluster in self.clusters.items():
                centroid = cluster.get("centroid")
                if centroid is None:
                    continue
                
                # Compute similarity to cluster centroid
                sim = safe_cosine_similarity(vec_tensor, centroid)
                if sim is not None and sim > self.similarity_threshold:
                    # Add to existing cluster
                    cluster["members"].append(vec_tensor.clone() if TORCH_AVAILABLE and isinstance(vec_tensor, torch.Tensor) else vec_tensor)
                    
                    # Update centroid (mean of all members)
                    if TORCH_AVAILABLE and isinstance(vec_tensor, torch.Tensor):
                        members_tensor = torch.stack(cluster["members"])
                        cluster["centroid"] = torch.mean(members_tensor, dim=0)
                    else:
                        # Python list fallback
                        if cluster["members"]:
                            dim = len(cluster["members"][0])
                            centroid_list = [sum(m[i] for m in cluster["members"]) / len(cluster["members"]) 
                                           for i in range(dim)]
                            cluster["centroid"] = centroid_list
                    
                    # Update metadata
                    if "skill_names" not in cluster.get("metadata", {}):
                        cluster["metadata"] = cluster.get("metadata", {})
                        cluster["metadata"]["skill_names"] = []
                    cluster["metadata"]["skill_names"].append(name)
                    
                    assigned = True
                    break
            
            # Create new cluster if no match found
            if not assigned:
                cname = f"competency_{len(self.clusters)}"
                vec_clone = vec_tensor.clone() if TORCH_AVAILABLE and isinstance(vec_tensor, torch.Tensor) else vec_tensor
                self.clusters[cname] = {
                    "centroid": vec_clone,
                    "members": [vec_clone],
                    "metadata": {
                        "skill_names": [name],
                        "created_at": len(self.clusters)
                    }
                }

    def get_centroids(self):
        """
        Get all cluster centroids for use in thought selection and biasing.
        
        Returns:
            List of centroid vectors
        """
        return [c.get("centroid") for c in self.clusters.values() if c.get("centroid") is not None]
    
    def get_cluster_by_name(self, name):
        """
        Get a specific cluster by name.
        
        Args:
            name: Cluster name
            
        Returns:
            Cluster dict or None
        """
        return self.clusters.get(name)
    
    def get_cluster_count(self):
        """
        Get total number of competency clusters.
        
        Returns:
            int: Number of clusters
        """
        return len(self.clusters)
    
    def status(self):
        """
        Get status summary of competency clusters.
        
        Returns:
            Dict mapping cluster names to member counts
        """
        return {
            k: {
                "member_count": len(v.get("members", [])),
                "skill_names": v.get("metadata", {}).get("skill_names", [])
            }
            for k, v in self.clusters.items()
        }
    
    def clear_clusters(self):
        """
        Clear all clusters (for testing/reset).
        """
        self.clusters = {}
    
    def get_competency_influence(self, target_vec):
        """
        Compute how much each competency cluster influences a target vector.
        
        This is used to determine which competencies should activate
        based on the current cognitive state.
        
        Args:
            target_vec: Target vector to compute influence for
            
        Returns:
            Dict mapping cluster names to influence scores
        """
        target = safe_tensor(target_vec)
        if target is None:
            return {}
        
        influences = {}
        for cname, cluster in self.clusters.items():
            centroid = cluster.get("centroid")
            if centroid is not None:
                sim = safe_cosine_similarity(target, centroid)
                if sim is not None:
                    influences[cname] = sim
        
        return influences

