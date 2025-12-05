# prime-core/memory/neural/episodic_neural_memory.py

"""
Episodic Neural Memory

----------------------

Stores embeddings of real experiences across time.

"""

import time

# Import from neural utilities
try:
    from ...neural.torch_utils import safe_cosine_similarity
except (ImportError, ValueError):
    # Fallback: direct import when running as script or different package structure
    import sys
    import os
    _neural_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../neural'))
    if _neural_path not in sys.path:
        sys.path.insert(0, _neural_path)
    from torch_utils import safe_cosine_similarity

class EpisodicNeuralMemory:

    def __init__(self):

        self.entries = []

    def store(self, embedding, metadata=None):

        """

        Store an embedding with a timestamp and optional metadata.

        """

        entry = {

            "timestamp": time.time(),

            "vector": embedding,

            "meta": metadata or {}

        }

        self.entries.append(entry)

        return entry

    def retrieve_similar(self, embedding, top_k=5):

        """

        Find the most similar stored episodes.

        """

        scored = []

        for e in self.entries:

            score = safe_cosine_similarity(embedding, e["vector"])

            scored.append((score, e))

        scored.sort(key=lambda x: x[0], reverse=True)

        return scored[:top_k]

    def stats(self):

        return {

            "count": len(self.entries),

            "latest": self.entries[-1]["timestamp"] if self.entries else None

        }

