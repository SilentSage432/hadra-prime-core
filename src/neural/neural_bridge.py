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
from .reflective_thought_generator import ReflectiveThoughtGenerator
from .self_stability_engine import SelfStabilityEngine
from .adaptive_evolution_engine import AdaptiveEvolutionEngine
from .evolution_memory_consolidator import EvolutionMemoryConsolidator
from .evolution_trajectory_predictor import EvolutionaryTrajectoryPredictor
from ..memory.memory_interaction_engine import MemoryInteractionEngine
from ..memory.autobiographical_memory import AutobiographicalMemory
from ..self.self_model_engine import SelfModelEngine
from ..cognition.global_workspace import ConsciousWorkspace
from ..cognition.cognitive_action_engine import CognitiveActionEngine
from ..cognition.cognitive_loop_orchestrator import CognitiveLoopOrchestrator
from ..cognition.cognitive_growth_scheduler import CognitiveGrowthScheduler
from ..cognition.personality_drift_regulator import PersonalityDriftRegulator
from ..cognition.personality_gradient_engine import PersonalityGradientEngine
from ..cognition.personality_signature_engine import PersonalitySignatureEngine
from ..cognition.personality_continuity_engine import LifelongPersonalityContinuity

# Import persistence layer (from project root)
import sys
import os
_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if _root_path not in sys.path:
    sys.path.insert(0, _root_path)
from persistence.memory_store import MemoryStore
from persistence.log_writer import LogWriter
from perception.perception_manager import PerceptionManager
from tasks.task_queue import TaskQueue

class NeuralBridge:

    def __init__(self):

        self.hooks = NeuralEventHooks()
        self.state = NeuralStateTracker()
        self.dual = DualMindSync()
        self.coherence = NeuralCoherenceEngine()
        self.attention = NeuralAttentionEngine()
        self.fusion = NeuralContextFusion()
        self.selector = NeuralThoughtSelector()
        self.generator = CandidateThoughtGenerator(bridge=self)
        self.reflector = ReflectiveThoughtGenerator()
        self.memory_engine = MemoryInteractionEngine()
        self.action_engine = CognitiveActionEngine()
        self.memory_store = MemoryStore()
        self.logger = LogWriter()
        self.perception = PerceptionManager(self)
        self.tasks = TaskQueue()
        self.orchestrator = CognitiveLoopOrchestrator(self)
        self.stability = SelfStabilityEngine()
        self.evolution = AdaptiveEvolutionEngine()
        self.evo_consolidator = EvolutionMemoryConsolidator()
        self.evo_predictor = EvolutionaryTrajectoryPredictor()
        self.growth = CognitiveGrowthScheduler()
        self.personality = PersonalityDriftRegulator()
        self.personality_gradient = PersonalityGradientEngine()
        self.personality_signature = PersonalitySignatureEngine()
        self.personality_continuity = LifelongPersonalityContinuity()
        # A170 Autobiographical Memory Matrix
        self.autobio = AutobiographicalMemory()
        # A171: Emergent Self-Model Engine
        self.self_model = SelfModelEngine()
        # A172: Conscious Workspace Buffer (Global Workspace Core)
        self.workspace = ConsciousWorkspace(self.state)
        self.ready_for_adaptive_evolution = False
        self.cycle_count = 0
        self.stability_report = None
        
        # -----------------------------------------
        # Seed Embeddings Activation (A152 Fix)
        # -----------------------------------------
        try:
            self.state.seed_embeddings = self.embed_seed_memory()
            print("🔥 SEED EMBEDDINGS LOADED:", len(self.state.seed_embeddings))
        except Exception as e:
            print("SEED EMBEDDING ERROR:", e)
            self.state.seed_embeddings = []
        
        # Seed neural memory with initial concepts on first run
        self.seed_neural_memory()
        
        # A157: Register rewrite paths for evolution engine
        self.evolution.register_mutation_point(
            "attention",
            self.attention.get_scaling,
            self.attention.set_scaling
        )
        self.evolution.register_mutation_point(
            "fusion",
            self.fusion.get_identity_weight,
            self.fusion.set_identity_weight
        )

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

        return self.action_engine.choose_action(self)

    def perform_action(self, action):

        return self.action_engine.execute(action, self)

    def propose_thoughts(self):
        """
        Generate candidate thought embeddings for A144 to evaluate.
        """
        fusion = self.fusion.last_fusion_vector
        attention = self.attention.last_focus_vector
        identity = self.state.timescales.identity_vector

        # Generate base candidates (generator now handles seed injection internally)
        candidates = self.generator.propose(fusion, attention, identity)
        
        # Include last perception embedding as thought seed
        if self.state.last_perception and self.state.last_perception.get("embedding") is not None:
            perception_vec = self.state.last_perception["embedding"]
            candidates.append(perception_vec)
        
        # Include task embeddings
        task_embeddings = getattr(self.state, "task_embeddings", [])
        for item in task_embeddings:
            if item.get("embedding") is not None:
                candidates.append(item["embedding"])
        
        return candidates

    def memory_cycle(self):
        """
        Perform a full memory metabolism cycle:
        - Recall contextually relevant memories
        - Reinforce accessed memories
        - Decay unused memories
        """
        fusion = self.fusion.last_fusion_vector
        attention = self.attention.last_focus_vector
        mm = self.state.memory_manager if hasattr(self.state, "memory_manager") else None

        if mm is None:
            return []

        return self.memory_engine.maintenance_cycle(fusion, attention, mm)

    def generate_reflection(self):
        """
        Generate a meaningful internal reflection embedding based on:
        - Current fusion state
        - Attention signals
        - Relevant semantic memories
        - Long-term identity
        """
        fusion = self.fusion.last_fusion_vector
        attention = self.attention.last_focus_vector
        timescales = self.state.timescales
        mm = self.state.memory_manager if hasattr(self.state, "memory_manager") else None

        return self.reflector.generate(fusion, attention, timescales, mm)

    def cognitive_step(self):
        """
        Perform a single complete cognitive cycle:
        - Propose candidate thoughts
        - Select best thought
        - Execute cognitive action
        - Perform memory metabolism
        - Update attention & fusion
        - Monitor drift & coherence
        """
        return self.orchestrator.step()

    def cognitive_status(self):
        """
        Get the status of the last cognitive step.
        """
        return self.orchestrator.status()

    def embed_seed_memory(self):
        """
        Load seed memories from the memory store and embed them.
        Ensures compatibility across all stored seed formats.
        """
        seeds = self.memory_store.data.get("thought_events", [])
        embedded = []
        
        for seed in seeds:
            # Each seed may be stored as:
            # { "timestamp": "...", "data": { "content": "...", ... } }
            # or directly as { "content": "...", ... }
            data = seed.get("data", seed)
            
            # Check if this is a seed (timestamp == "seed" or has seed-like structure)
            is_seed = seed.get("timestamp") == "seed" or "content" in data
            
            if not is_seed:
                continue
            
            text = data.get("content") or data.get("text")
            if not text:
                continue
            
            try:
                # Use hooks to encode the seed text
                vec = self.hooks.on_perception(text)
                if vec is not None:
                    # Keep as tensor (don't convert to list)
                    embedded.append({
                        "text": text,
                        "type": data.get("type", "concept"),
                        "embedding": vec
                    })
            except Exception as e:
                print("ERROR embedding seed:", text, e)
        
        print("🔥 FINAL SEED EMBEDDING COUNT:", len(embedded))
        return embedded

    def seed_neural_memory(self):
        """
        Seed the neural memory manager with initial concepts if it exists.
        This ensures semantic memory has initial anchors for thought generation.
        """
        # Check if memory_manager exists, if not, initialize it
        if not hasattr(self.state, "memory_manager") or self.state.memory_manager is None:
            from ..memory.neural.neural_memory_manager import NeuralMemoryManager
            self.state.memory_manager = NeuralMemoryManager()
        
        # Get seed embeddings
        seed_embeddings = self.embed_seed_memory()
        
        # Store seeds in semantic memory
        for seed in seed_embeddings:
            seed_type = seed.get("type", "concept")
            seed_text = seed.get("text", "")
            seed_vec = seed.get("embedding")
            
            if seed_vec is not None:
                # Store in semantic memory with a meaningful name
                if seed_type == "identity":
                    name = f"identity_{seed_text[:20].replace(' ', '_')}"
                else:
                    name = f"concept_{seed_text}"
                
                self.state.memory_manager.store_concept(name, seed_vec)

    def inject_perception(self, text: str):
        """
        Main operator-facing API endpoint for injecting perceptions into PRIME.
        Encodes the text, updates neural state, and logs the perception.
        """
        perception = self.perception.perceive(text)
        self.memory_store.log_perception(perception)
        
        # Also process through the full perception pipeline
        self.process_perception(text)
        
        return perception

    def add_task(self, text, priority=5):
        """
        Add a task to PRIME's task queue with priority.
        Tasks influence thought generation and cognitive direction.
        """
        task = self.tasks.add_task(text, priority)
        
        # Encode task text into embedding
        embedding = self.hooks.on_perception(text)
        
        # Convert to list if tensor
        embedding_list = embedding
        try:
            import torch
            if isinstance(embedding, torch.Tensor):
                embedding_list = embedding.tolist()
        except:
            pass
        
        # Store in memory
        self.memory_store.log_thought_event({
            "type": "task_added",
            "content": text,
            "priority": priority
        })
        
        # Store embedding for future use
        self.state.task_embeddings.append({
            "text": text,
            "embedding": embedding,
            "embedding_list": embedding_list,
            "priority": priority
        })
        
        return {"task": text, "priority": priority}

    def autobiographical_summary(self):
        """
        A170: Get summary of autobiographical memory state.
        """
        return self.autobio.summarize()

    def autobiographical_recent(self, n=10):
        """
        A170: Get the most recent N autobiographical entries.
        """
        return self.autobio.get_recent(n)
    
    def self_model_status(self):
        """
        A171: Get status of the emergent self-model.
        """
        return self.self_model.summary()

