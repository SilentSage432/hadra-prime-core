# prime-core/neural/neural_bridge.py

"""
Neural Bridge

-------------

This layer connects PRIME's cognitive systems to the neural subsystem.

It does NOT perform heavy neural reasoning yet — it ensures PRIME

starts routing thoughts, perceptions, and narratives through embeddings.

"""

from .neural_event_hooks import NeuralEventHooks
from .neural_state_tracker import NeuralStateTracker
from .dual_mind_sync import DualMindSync
from .neural_coherence_engine import NeuralCoherenceEngine
from .neural_attention_engine import NeuralAttentionEngine

class NeuralBridge:

    def __init__(self):

        self.hooks = NeuralEventHooks()
        self.state = NeuralStateTracker()
        self.dual = DualMindSync()
        self.coherence = NeuralCoherenceEngine()
        self.attention = NeuralAttentionEngine()

    def process_perception(self, text):

        """

        Entry point for PRIME's perception to flow into neural processing.

        """

        embedding = self.hooks.on_perception(text)
        
        # Stabilize neural signal before committing to state
        identity = self.state.timescales.identity_vector
        stable = self.coherence.stabilize(embedding, identity)
        
        self.state.update(stable)
        
        # Update attention focus vector using new updated timescales
        focus = self.attention.compute_attention_vector(self.state.timescales)
        
        # Update dual-mind shared vectors using LT identity vector + ST summary
        lt_vec = self.state.timescales.identity_vector
        st_summary = self.state.timescales.ST.summary_vector()
        
        self.dual.update_prime_vectors(lt_vec, st_summary)
        
        return stable

    def process_reflection(self, thought):

        """

        Entry point for PRIME's reflective cognition.

        """

        embedding = self.hooks.on_reflection(thought)
        
        # Stabilize neural signal before committing to state
        identity = self.state.timescales.identity_vector
        stable = self.coherence.stabilize(embedding, identity)
        
        self.state.update(stable)
        
        # Update attention focus vector using new updated timescales
        focus = self.attention.compute_attention_vector(self.state.timescales)
        
        # Update dual-mind shared vectors using LT identity vector + ST summary
        lt_vec = self.state.timescales.identity_vector
        st_summary = self.state.timescales.ST.summary_vector()
        
        self.dual.update_prime_vectors(lt_vec, st_summary)
        
        return stable

    def compare(self, a, b):

        """

        Compare embeddings using cosine similarity.

        """

        return self.hooks.similarity(a, b)

    def status(self):

        return self.hooks.debug_status()

    def neural_state(self):

        return self.state.summary()

    def dual_mind_status(self):

        return self.dual.status()

    def coherence_status(self):

        return {

            "dual_mind": self.dual.status(),

            "state": self.state.summary(),

        }

    def attention_status(self):

        return self.attention.status()

