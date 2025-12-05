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
from .neural_context_fusion import NeuralContextFusion
from .neural_thought_selector import NeuralThoughtSelector
from .candidate_thought_generator import CandidateThoughtGenerator
from ..cognition.cognitive_action_engine import CognitiveActionEngine

class NeuralBridge:

    def __init__(self):

        self.hooks = NeuralEventHooks()
        self.state = NeuralStateTracker()
        self.dual = DualMindSync()
        self.coherence = NeuralCoherenceEngine()
        self.attention = NeuralAttentionEngine()
        self.fusion = NeuralContextFusion()
        self.selector = NeuralThoughtSelector()
        self.generator = CandidateThoughtGenerator()
        self.action_engine = CognitiveActionEngine()

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
        
        # Create fusion vector
        fusion = self.fusion.fuse(
            self.attention.last_focus_vector,
            self.state.timescales
        )
        
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
        
        # Create fusion vector
        fusion = self.fusion.fuse(
            self.attention.last_focus_vector,
            self.state.timescales
        )
        
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

    def fusion_status(self):

        return self.fusion.status()

    def select_thought(self, candidate_embeddings):

        fusion = self.fusion.last_fusion_vector

        return self.selector.select(

            candidate_embeddings,

            fusion,

            self.attention,

            self.state.memory_manager if hasattr(self.state, "memory_manager") else None

        )

    def choose_cognitive_action(self):

        return self.action_engine.choose_action()

    def perform_action(self, action):

        return self.action_engine.execute(action, self)

    def propose_thoughts(self):
        """
        Generate candidate thought embeddings for A144 to evaluate.
        """
        fusion = self.fusion.last_fusion_vector
        attention = self.attention.last_focus_vector
        identity = self.state.timescales.identity_vector

        return self.generator.propose(fusion, attention, identity)

