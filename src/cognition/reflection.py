# prime-core/cognition/reflection.py

"""
Reflection Module (Python)

--------------------------

PRIME's reflection layer with neural routing integration.

This module processes internal thoughts and routes them through the neural subsystem.

"""

try:
    from ..neural.neural_kernel_adapter import NeuralKernelAdapter
    _neural_adapter = NeuralKernelAdapter()
except ImportError:
    _neural_adapter = None


class Reflection:

    def __init__(self):

        # Add inside Reflection.__init__()
        self.neural = _neural_adapter

    def neural_reflect(self, thought):

        """

        Pass reflection into the neural subsystem.

        """

        if self.neural:

            return self.neural.reflect(thought)

        return None

    def reflect(self, cognitive_state=None, sel_state=None):

        """

        Main reflection processing method.

        Processes internal thoughts and generates neural embeddings.

        """

        import time

        # Generate reflection text from cognitive state
        reflection_text = self._generate_reflection_text(cognitive_state, sel_state)

        result = {

            "timestamp": time.time(),

            "reflection_text": reflection_text,

            "neural_embedding": None

        }

        # Integrate into the main reflection cycle
        neural_vector = self.neural_reflect(reflection_text)

        result["neural_embedding"] = neural_vector

        return result

    def _generate_reflection_text(self, cognitive_state, sel_state):

        """

        Generate reflection text from cognitive state.

        """

        if not cognitive_state:

            return "No cognitive state available for reflection."

        goal = cognitive_state.get("topGoal", {}).get("type", "none") if isinstance(cognitive_state, dict) else getattr(cognitive_state, "topGoal", {}).get("type", "none") if hasattr(cognitive_state, "topGoal") else "none"

        motivation = cognitive_state.get("motivation", {}) if isinstance(cognitive_state, dict) else getattr(cognitive_state, "motivation", {}) if hasattr(cognitive_state, "motivation") else {}

        consolidation = motivation.get("consolidation", 0) if isinstance(motivation, dict) else getattr(motivation, "consolidation", 0) if hasattr(motivation, "consolidation") else 0

        curiosity = motivation.get("curiosity", 0) if isinstance(motivation, dict) else getattr(motivation, "curiosity", 0) if hasattr(motivation, "curiosity") else 0

        clarity_seeking = motivation.get("claritySeeking", 0) if isinstance(motivation, dict) else getattr(motivation, "claritySeeking", 0) if hasattr(motivation, "claritySeeking") else 0

        return f"Reflecting: PRIME prioritized '{goal}' due to consolidation={consolidation:.3f}, curiosity={curiosity:.3f}, claritySeeking={clarity_seeking:.3f}."

