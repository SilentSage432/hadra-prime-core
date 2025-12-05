# prime-core/memory/neural/semantic_neural_memory.py

"""
Semantic Neural Memory

----------------------

Stores conceptual / meaning-level embeddings.

Represents PRIME's internal concepts and understanding.

"""

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

class SemanticNeuralMemory:

    def __init__(self):

        self.concepts = {}  # name → vector

    def store(self, name, embedding):

        """

        Save or update a semantic concept vector.

        """

        self.concepts[name] = embedding

    def retrieve(self, name):

        return self.concepts.get(name)

    def find_similar(self, embedding, top_k=5):

        scored = []

        for name, vec in self.concepts.items():

            score = safe_cosine_similarity(embedding, vec)

            scored.append((score, name, vec))

        scored.sort(key=lambda x: x[0], reverse=True)

        return scored[:top_k]

    def stats(self):

        return {

            "concepts": list(self.concepts.keys()),

            "count": len(self.concepts)

        }

