# src/cognition/inference_cooldown.py

"""
Inference Cooling Window

Prevents premature certainty by requiring multiple observations
before inferences can solidify. Sovereign learning principle:
meaning emerges through repeated exposure, not single events.
"""

import os

# Default cooldown window: N observations required before solidifying
DEFAULT_INFERENCE_COOLDOWN = 7

def get_cooldown_threshold() -> int:
    """
    Get inference cooldown threshold from environment variable.
    Defaults to 7 if not set.
    """
    try:
        cooldown_str = os.getenv("ADRAE_INFERENCE_COOLDOWN", str(DEFAULT_INFERENCE_COOLDOWN))
        return int(cooldown_str)
    except (ValueError, TypeError):
        return DEFAULT_INFERENCE_COOLDOWN

class InferenceTracker:
    """
    Tracks observation counts for inferences.
    Prevents solidification until cooldown threshold is met.
    """
    
    def __init__(self):
        self.observation_counts = {}  # inference_id -> count
        self.cooldown_threshold = get_cooldown_threshold()
    
    def record_observation(self, inference_id: str) -> int:
        """
        Record an observation for an inference.
        Returns current count.
        """
        if inference_id not in self.observation_counts:
            self.observation_counts[inference_id] = 0
        self.observation_counts[inference_id] += 1
        return self.observation_counts[inference_id]
    
    def is_ready(self, inference_id: str) -> bool:
        """
        Check if inference has passed cooldown threshold.
        """
        count = self.observation_counts.get(inference_id, 0)
        return count >= self.cooldown_threshold
    
    def get_count(self, inference_id: str) -> int:
        """Get current observation count for an inference."""
        return self.observation_counts.get(inference_id, 0)

