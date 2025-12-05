# prime-core/neural/neural_event_hooks.py

"""
Neural Event Hooks

------------------

This subsystem attaches neural processing to PRIME's cognitive phases.

No PyTorch is used yet — everything routes through safe tensor utilities.

"""

from .tensor_pipeline import TensorPipeline

from .torch_utils import safe_cosine_similarity

class NeuralEventHooks:

    def __init__(self):

        self.pipeline = TensorPipeline()

        self.enabled = True

        self.last_embedding = None

    def on_perception(self, text: str):

        """

        When PRIME perceives new input.

        We generate a lightweight embedding using the temporary encoder.

        """

        if not self.enabled:

            return None

        embedding = self.pipeline.encode_text(text)

        self.last_embedding = embedding

        return embedding

    def on_reflection(self, thought: str):

        """

        When PRIME reflects internally.

        Also produces an embedding.

        """

        if not self.enabled:

            return None

        emb = self.pipeline.encode_text(thought)

        self.last_embedding = emb

        return emb

    def similarity(self, a, b):

        """

        Computes similarity between two embeddings.

        """

        return safe_cosine_similarity(a, b)

    def debug_status(self):

        return {

            "enabled": self.enabled,

            "last_vector": self.pipeline.stats(),

        }

