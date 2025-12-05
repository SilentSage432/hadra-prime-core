# prime-core/neural/neural_bridge.py

"""
Neural Bridge

-------------

This layer connects PRIME's cognitive systems to the neural subsystem.

It does NOT perform heavy neural reasoning yet — it ensures PRIME

starts routing thoughts, perceptions, and narratives through embeddings.

"""

from .neural_event_hooks import NeuralEventHooks

class NeuralBridge:

    def __init__(self):

        self.hooks = NeuralEventHooks()

    def process_perception(self, text):

        """

        Entry point for PRIME's perception to flow into neural processing.

        """

        return self.hooks.on_perception(text)

    def process_reflection(self, thought):

        """

        Entry point for PRIME's reflective cognition.

        """

        return self.hooks.on_reflection(thought)

    def compare(self, a, b):

        """

        Compare embeddings using cosine similarity.

        """

        return self.hooks.similarity(a, b)

    def status(self):

        return self.hooks.debug_status()

