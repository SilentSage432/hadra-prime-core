# prime-core/memory/neural/neural_memory_manager.py

"""
Neural Memory Manager

---------------------

Unifies episodic & semantic neural memory.

Acts as PRIME's high-level interface to neural storage.

"""

from .episodic_neural_memory import EpisodicNeuralMemory

from .semantic_neural_memory import SemanticNeuralMemory

class NeuralMemoryManager:

    def __init__(self):

        self.episodic = EpisodicNeuralMemory()

        self.semantic = SemanticNeuralMemory()

    def store_episode(self, embedding, meta=None):

        return self.episodic.store(embedding, meta)

    def store_concept(self, name, embedding):

        return self.semantic.store(name, embedding)

    def recall_similar_episodes(self, embedding, top_k=5):

        return self.episodic.retrieve_similar(embedding, top_k)

    def recall_similar_concepts(self, embedding, top_k=5):

        return self.semantic.find_similar(embedding, top_k)

    def stats(self):

        return {

            "episodic": self.episodic.stats(),

            "semantic": self.semantic.stats()

        }

