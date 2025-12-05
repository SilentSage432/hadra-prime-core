# prime-core/neural/neural_kernel_adapter.py

"""
Neural Kernel Adapter

---------------------

This component links PRIME's internal cognition cycle with the neural subsystem.

"""

from .neural_bridge import NeuralBridge

class NeuralKernelAdapter:

    def __init__(self):

        self.bridge = NeuralBridge()

    def perceive(self, text):

        """

        Called whenever PRIME receives external input.

        """

        return self.bridge.process_perception(text)

    def reflect(self, thought):

        """

        Called when PRIME generates internal thoughts.

        """

        return self.bridge.process_reflection(thought)

    def compare_embeddings(self, a, b):

        """

        Neural similarity primitive.

        """

        return self.bridge.compare(a, b)

    def status(self):

        return self.bridge.status()

