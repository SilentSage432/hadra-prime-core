# prime-core/cognition/perception.py

"""
Perception Module (Python)

--------------------------

PRIME's perception layer with neural routing integration.

This module processes external input and routes it through the neural subsystem.

"""

try:
    from ..neural.neural_kernel_adapter import NeuralKernelAdapter
    _neural_adapter = NeuralKernelAdapter()
except ImportError:
    _neural_adapter = None


class Perception:

    def __init__(self):

        # ADD THIS INSIDE Perception.__init__()
        self.neural = _neural_adapter

    def neural_perceive(self, text):

        """

        Pass perception into the neural subsystem.

        """

        if self.neural:

            return self.neural.perceive(text)

        return None

    def process(self, text):

        """

        Main perception processing method.

        Processes text input and generates neural embeddings.

        """

        import time

        result = {

            "text": text,

            "timestamp": time.time(),

            "neural_embedding": None

        }

        # INSIDE process(text)
        neural_vector = self.neural_perceive(text)

        result["neural_embedding"] = neural_vector

        return result

