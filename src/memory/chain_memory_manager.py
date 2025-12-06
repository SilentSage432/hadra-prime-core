"""
Multi-Step Chain Memory Imprinting & Optimization Engine (A213)
---------------------------------------------------------------
Stores, ranks, and optimizes multi-step execution chains.
Chains are treated as procedural knowledge that ADRAE learns and reuses.
"""

import time


class ChainMemoryManager:
    """
    A213 — Stores, ranks, and optimizes multi-step execution chains.
    Chains are treated as procedural knowledge that can be learned and reused.
    """
    
    def __init__(self):
        self.chains = []  # Full historical archive of execution chains

    def store_chain(self, chain, result):
        """
        Store a completed chain with metadata from the execution.
        
        Args:
            chain: The execution chain (list of subgoal dicts)
            result: Execution result dict with status, metrics, etc.
            
        Returns:
            Stored chain entry dict
        """
        entry = {
            "timestamp": time.time(),
            "chain": chain.copy() if hasattr(chain, 'copy') else list(chain) if chain else [],
            "success": result.get("status") == "chain_complete",
            "reroutes": result.get("reroutes", 0),
            "avg_drift": result.get("avg_drift", 0.0),
            "coherence": result.get("coherence", 1.0),
            "score": self._score_chain(result),
            "metadata": result.copy() if isinstance(result, dict) else {}
        }
        self.chains.append(entry)
        return entry

    def _score_chain(self, result):
        """
        Compute a reinforcement score for ranking procedural chains.
        
        Higher scores indicate more successful, stable, coherent chains.
        
        Args:
            result: Execution result dict
            
        Returns:
            Score (float) between 0.0 and 1.0
        """
        success = 1.0 if result.get("status") == "chain_complete" else 0.0
        drift_penalty = max(0.0, 1.0 - abs(result.get("avg_drift", 0.0)) * 10)
        coherence = result.get("coherence", 1.0)
        
        # Score: 60% success, 30% coherence, 10% drift stability
        score = (success * 0.6) + (coherence * 0.3) + (drift_penalty * 0.1)
        
        return score

    def retrieve_best_chains(self, top_k=3):
        """
        Returns the highest scoring procedural chains.
        
        Args:
            top_k: Number of top chains to return
            
        Returns:
            List of top-k chain entries, sorted by score
        """
        if not self.chains:
            return []
        
        ranked = sorted(self.chains, key=lambda c: c.get("score", 0.0), reverse=True)
        return ranked[:top_k]

    def optimize(self):
        """
        Prune low scoring chains & reinforce successful structural patterns.
        
        Removes chains with scores below 50% of average, keeping only
        effective procedural knowledge.
        """
        if len(self.chains) < 10:
            return
        
        avg_score = sum(c.get("score", 0.0) for c in self.chains) / len(self.chains)
        threshold = avg_score * 0.5
        
        # Remove ineffective procedural chains
        initial_count = len(self.chains)
        self.chains = [c for c in self.chains if c.get("score", 0.0) >= threshold]
        pruned_count = initial_count - len(self.chains)
        
        return {
            "pruned": pruned_count,
            "remaining": len(self.chains),
            "threshold": threshold
        }
    
    def get_chain_count(self):
        """Get total number of stored chains."""
        return len(self.chains)
    
    def get_average_score(self):
        """Get average score of all stored chains."""
        if not self.chains:
            return 0.0
        return sum(c.get("score", 0.0) for c in self.chains) / len(self.chains)
    
    def summary(self):
        """
        Get summary of chain memory state for logging.
        
        Returns:
            Dict with chain statistics
        """
        if not self.chains:
            return {
                "chain_count": 0,
                "average_score": 0.0,
                "best_chains": []
            }
        
        best_chains = self.retrieve_best_chains(top_k=3)
        
        return {
            "chain_count": len(self.chains),
            "average_score": round(self.get_average_score(), 4),
            "best_chains": [
                {
                    "score": round(c.get("score", 0.0), 4),
                    "success": c.get("success", False),
                    "chain_length": len(c.get("chain", [])),
                    "reroutes": c.get("reroutes", 0)
                }
                for c in best_chains
            ]
        }

