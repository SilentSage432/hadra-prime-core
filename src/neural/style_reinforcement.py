# prime-core/neural/style_reinforcement.py

"""
A225 — Cognitive Style Reinforcement Layer

------------------------------------------

Strengthens ADRAE's emergent cognitive style by reinforcing
patterns that increase coherence, clarity, and identity alignment.

This turns cognitive style from a static decorator (A224) into
a self-strengthening pattern that evolves over time.

"""

from .torch_utils import safe_tensor, safe_cosine_similarity, TORCH_AVAILABLE


class CognitiveStyleReinforcer:
    """
    A225 — Cognitive Style Reinforcement Layer
    
    Reinforces ADRAE's cognitive style traits based on runtime performance:
    - Positive reinforcement when coherence/identity alignment is high
    - Negative reinforcement (suppression) when coherence drops
    - Allows style to evolve and stabilize over time
    """
    
    def __init__(self):
        """
        Initialize the style reinforcer with learning rates.
        
        Running estimates of useful style traits are maintained
        and used to guide reinforcement.
        """
        # Running estimates of the useful style traits
        self.trend_tempo = 1.0
        self.trend_curvature = 0.15
        self.trend_resonance = 0.4
        self.trend_depth = 0.5
        self.trend_stability = 0.8
        
        # Learning rates
        self.alpha = 0.05  # slow positive reinforcement
        self.beta = 0.03   # slow negative reinforcement (suppression)
    
    def reinforce(self, style, coherence=None, identity_align=None):
        """
        Update the style parameters based on runtime signals.
        
        Args:
            style: CognitiveStyleArchitect instance to reinforce
            coherence: Current coherence value (0.0 to 1.0, optional)
            identity_align: Identity alignment value (0.0 to 1.0, optional)
            
        Returns:
            Updated style object (modified in place, returned for convenience)
        """
        if style is None:
            return style
        
        try:
            # Positive reinforcement - high coherence strengthens style
            if coherence is not None and coherence > 0.95:
                # Reinforce tempo (speed of transitions)
                style.tempo += self.alpha * (1.0 - style.tempo)
                
                # Reinforce curvature toward trend
                style.curvature += self.alpha * (self.trend_curvature - style.curvature)
                
                # Reinforce resonance gain toward trend
                style.resonance_gain += self.alpha * (self.trend_resonance - style.resonance_gain)
                
                # Reinforce depth bias toward trend
                style.depth_bias += self.alpha * (self.trend_depth - style.depth_bias)
            
            # Identity alignment reinforcement
            if identity_align is not None and identity_align > 0.90:
                # Strong identity alignment strengthens resonance
                style.resonance_gain += self.alpha * min(0.1, (identity_align - 0.9))
                
                # Also reinforce stability weight
                style.stability_weight += self.alpha * min(0.1, (identity_align - 0.9))
            
            # Negative reinforcement (suppression) - low coherence weakens unstable traits
            if coherence is not None and coherence < 0.85:
                # Suppress excessive curvature when coherence drops
                style.curvature -= self.beta * style.curvature
                
                # Suppress excessive novelty pull when coherence drops
                style.novelty_pull -= self.beta * style.novelty_pull
            
            # Clamp values to valid ranges
            style.tempo = max(0.2, min(2.0, style.tempo))
            style.curvature = max(0.0, min(0.5, style.curvature))
            style.resonance_gain = max(0.0, min(1.0, style.resonance_gain))
            style.depth_bias = max(0.0, min(1.0, style.depth_bias))
            style.novelty_pull = max(0.0, min(0.5, style.novelty_pull))
            style.stability_weight = max(0.0, min(1.0, style.stability_weight))
        
        except Exception as e:
            # If reinforcement fails, continue with current style
            pass
        
        return style
    
    def status(self, style):
        """
        Get status of style parameters.
        
        Args:
            style: CognitiveStyleArchitect instance
            
        Returns:
            Dict with current style parameters
        """
        if style is None:
            return {"active": False}
        
        try:
            return {
                "tempo": style.tempo,
                "curvature": style.curvature,
                "resonance_gain": style.resonance_gain,
                "depth_bias": style.depth_bias,
                "novelty_pull": style.novelty_pull,
                "stability_weight": style.stability_weight,
                "reinforcement_active": True
            }
        except Exception:
            return {"active": False}

