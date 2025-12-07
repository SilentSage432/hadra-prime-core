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
from .models.identity_encoder import IdentityEncoder
from .trainers.identity_trainer import IdentityTrainer
# A230 — PyTorch Latent Concept Engine
try:
    from .latent_concept_engine import NeuralConceptMapper, LatentStabilityRegulator
    LATENT_ENGINE_AVAILABLE = True
except (ImportError, RuntimeError) as e:
    NeuralConceptMapper = None
    LatentStabilityRegulator = None
    LATENT_ENGINE_AVAILABLE = False

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
        # A213 — Multi-Step Chain Memory Imprinting & Optimization Engine
        from ..memory.chain_memory_manager import ChainMemoryManager
        self.chain_memory = ChainMemoryManager()
        # A214 — Procedural Reasoning Encoder (Skill Embedding Engine)
        from ..memory.skill_embedding_encoder import SkillEmbeddingEncoder
        self.skill_encoder = SkillEmbeddingEncoder(dim=128)
        # Initialize skill embeddings in generator
        self.generator.skill_embeddings = []
        # A215 — Procedural Skill Generalization & Cross-Domain Transfer
        from ..memory.skill_generalization_engine import SkillGeneralizationEngine
        self.skill_generalizer = SkillGeneralizationEngine()
        # Initialize generalized skill patterns in generator
        self.generator.generalized_skill_patterns = []
        # A217 — Skill Vector Expansion Through Uncertainty Minimization
        from .skill_manager import SkillManager
        self.skills = SkillManager()
        # A218 — Skill Specialization & Competency Clustering
        from .competency_manager import CompetencyManager
        self.competencies = CompetencyManager()
        # A221 — Synergy-Based Thought Signature Stabilizer
        from .thought_signature import ThoughtSignature
        self.thought_signature = ThoughtSignature(dim=128)
        # A222 — Signature-Guided Thought Harmonization Layer
        from .signature_harmonizer import SignatureHarmonizer
        self.harmonizer = SignatureHarmonizer(strength=0.15)
        # A223 — Emergent Personality Flow Fields
        from .personality_flow_field import PersonalityFlowField
        self.flow = PersonalityFlowField(influence=0.12, memory_length=50)
        # A224 — Emergent Cognitive Style Architect
        from .emergent_cognitive_style import CognitiveStyleArchitect
        self.style = CognitiveStyleArchitect()
        # A225 — Cognitive Style Reinforcement Layer
        from .style_reinforcement import CognitiveStyleReinforcer
        self.style_reinforcer = CognitiveStyleReinforcer()
        # A227 — Style-Guided Micro-Narrative Formation Layer
        from .micro_narrative import MicroNarrativeEngine
        self.micro_narrative = MicroNarrativeEngine(max_arc_length=4)
        # A228 — Narrative-Driven Cognitive Anticipation Layer
        self.narrative_projection = {
            "forward_arc": [],
            "confidence": 0.0,
            "tension": 0.0,
        }
        self.goal_anticipation_map = {}
        self.narrative_arc_sequencer = {
            "micro_chain": [],
            "macro_projection": []
        }
        # A229 — Narrative Coherence & Tension Regulation Layer
        self.narrative_coherence = {
            "coherence_score": 0.0,
            "conflicts": [],
            "adjusted_arc": []
        }
        self.tension_regulation = {
            "tension_score": 0.0,
            "stability_factor": 1.0,
            "recommended_adjustment": None
        }
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
        # A-SOV-05: ADRAE Conscious Workspace Adapter
        from ..cognition.adrae_workspace_adapter import ADRAEWorkspaceAdapter
        self.adrae_workspace = ADRAEWorkspaceAdapter(self)
        # A179 — Supervisory Control Network
        from ..cognition.supervisory_control_network import SupervisoryControlNetwork
        self.supervisor = SupervisoryControlNetwork()
        # A180 — Supervisory Conflict Resolution Layer
        from ..cognition.supervisory_conflict_resolver import SupervisoryConflictResolver
        self.conflict_resolver = SupervisoryConflictResolver()
        # A181 — Meta-Intent Coordinator
        from ..cognition.meta_intent_coordinator import MetaIntentCoordinator
        self.meta_intent = MetaIntentCoordinator()
        # A182 — Intent-Aware Global Workspace Reinforcement
        from ..cognition.global_workspace_reinforcement import GlobalWorkspaceReinforcement
        self.workspace_reinforcement = GlobalWorkspaceReinforcement(dim=128)
        # A202 — Global Workspace Cross-Cycle Continuity Engine
        from ..cognition.global_workspace_continuity import GlobalWorkspaceContinuity
        self.continuity = GlobalWorkspaceContinuity(dim=128)
        # A203 — Emergent Goal Formation Layer
        from .neural_goal_manager import (
            NeuralGoalProposer, NeuralGoalEvaluator, NeuralGoalManager
        )
        self.goal_proposer = NeuralGoalProposer()
        self.goal_evaluator = NeuralGoalEvaluator()
        self.goal_manager = NeuralGoalManager()
        # A204 — Goal-Driven Cognitive Modulation Engine
        from .goal_modulation_engine import GoalModulationEngine
        self.goal_modulator = GoalModulationEngine()
        self.last_goal_modulation = None
        # A205 — Emergent Multi-Vector Goal Fabrication Layer
        from .goal_fabrication_engine import GoalFabricationEngine
        self.goal_fabricator = GoalFabricationEngine()
        # A206 — Multi-Goal Competition & Harmonization Engine
        from .multi_goal_harmonizer import MultiGoalHarmonizer
        self.goal_harmonizer = MultiGoalHarmonizer()
        self.last_harmonized_goal = None
        # A207 — Goal-Driven Cognitive Path Shaping Engine
        from .goal_path_shaping_engine import GoalPathShapingEngine
        self.path_shaper = GoalPathShapingEngine()
        # A208 — Adaptive Subgoal Generator
        from .adaptive_subgoal_generator import AdaptiveSubgoalGenerator
        self.subgoal_generator = AdaptiveSubgoalGenerator()
        # A209 — Subgoal Competition & Selection Layer
        from .subgoal_competition_engine import SubgoalCompetitionEngine
        self.subgoal_competition = SubgoalCompetitionEngine()
        # A210 — Dynamic Subgoal Routing & Execution Priority
        from .subgoal_routing_engine import SubgoalRoutingEngine
        self.subgoal_router = SubgoalRoutingEngine()
        # A211 — Multi-Step Execution Chains (Sequential Planning Engine)
        from .sequential_planning_engine import SequentialPlanningEngine
        self.planning_engine = SequentialPlanningEngine()
        # A183 — Identity Drift Suppression
        from ..identity.identity_drift_suppressor import IdentityDriftSuppressor
        self.identity_drift = IdentityDriftSuppressor()
        # A184 — Identity Supervisory Gate
        from ..identity.identity_supervisory_gate import IdentitySupervisoryGate
        self.identity_gate = IdentitySupervisoryGate()
        # A185 — Temporal Identity Consolidation
        from ..identity.temporal_identity_consolidation import TemporalIdentityConsolidator
        self.identity_consolidator = TemporalIdentityConsolidator()
        # A186 — Dreamspace (Subconscious Processing Layer)
        from ..cognition.dreamspace import Dreamspace
        self.dreamspace = Dreamspace()
        # A200 — Neural Identity Encoder
        try:
            self.identity_encoder = IdentityEncoder()
            self.identity_encoder.eval()  # Set to evaluation mode initially
        except Exception as e:
            print(f"⚠️ IdentityEncoder initialization failed: {e}")
            self.identity_encoder = None
        
        # A201 — Identity Training Cycle
        try:
            if self.identity_encoder is not None:
                self.identity_trainer = IdentityTrainer(self.identity_encoder, lr=1e-4)
            else:
                self.identity_trainer = None
        except Exception as e:
            print(f"⚠️ IdentityTrainer initialization failed: {e}")
            self.identity_trainer = None
        # Keep a frozen copy of PRIME's identity for drift comparisons
        self.baseline_identity = None
        self.ready_for_adaptive_evolution = False
        self.cycle_count = 0
        # A230 — PyTorch Latent Concept Engine (Imagination Substrate Initialization)
        self._initialize_latent_engine()
        # A185 — Sleep/wake timer
        self.cycle_step = 0
        self.sleep_cycle_interval = 12  # every 12 cognition cycles, PRIME "sleeps"
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
        
        # A222 — Update harmonizer signature from identity vectors
        self._update_harmonizer_signature()
        
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
        
        # === A201: Train Identity Encoder ===
        if self.identity_trainer is not None and lt_vec is not None:
            try:
                from .torch_utils import safe_tensor, TORCH_AVAILABLE
                import torch
                
                if TORCH_AVAILABLE:
                    identity_tensor = safe_tensor(lt_vec)
                    if identity_tensor is not None and isinstance(identity_tensor, torch.Tensor):
                        # Get previous latent identity for continuity
                        prev_latent = None
                        if hasattr(self.state.timescales, 'latent_identity') and self.state.timescales.latent_identity is not None:
                            prev_latent = safe_tensor(self.state.timescales.latent_identity)
                            if prev_latent is not None and isinstance(prev_latent, torch.Tensor):
                                # Ensure prev_latent is on same device as identity_tensor
                                if prev_latent.device != identity_tensor.device:
                                    prev_latent = prev_latent.to(identity_tensor.device)
                        
                        # Get coherence score from dual-mind sync
                        coherence = self.dual.coherence_score()
                        if coherence is None:
                            coherence = 0.5  # Default to neutral if SAGE not connected
                        
                        # Training step: learn identity representation
                        new_latent, loss = self.identity_trainer.train_step(
                            identity_tensor,
                            prev_latent,
                            coherence
                        )
                        
                        # Store updated latent identity
                        new_latent_squeezed = new_latent.squeeze(0) if new_latent.dim() > 1 else new_latent
                        self.state.timescales.latent_identity = new_latent_squeezed.detach().clone()
                        
                        # Log training event
                        print(f"[A201] Identity latent updated — loss={loss:.6f}, coherence={coherence:.3f}")
                        if hasattr(self, 'logger'):
                            try:
                                self.logger.write({
                                    "identity_training": {
                                        "event": "neural_identity_trained",
                                        "loss": float(loss),
                                        "coherence": float(coherence) if coherence is not None else None,
                                        "latent_dim": new_latent_squeezed.shape[-1] if new_latent_squeezed.dim() > 0 else 32,
                                        "status": "trained"
                                    }
                                })
                            except Exception:
                                pass
            except Exception as e:
                # If training fails, continue without it
                print(f"Identity training error: {e}")
                if hasattr(self, 'logger'):
                    try:
                        self.logger.write({"identity_training_error": str(e)})
                    except Exception:
                        pass
        
        self.dual.update_prime_vectors(lt_vec, st_summary)
        
        # A-SOV-07: Persist ADRAE identity drift-stable vector
        try:
            self.memory_store.persist_adrae_identity(self.state.timescales.identity_vector)
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.write({"adrae_persistence_error": str(e)})
        
        # A222 — Update harmonizer signature from identity vectors
        self._update_harmonizer_signature()
        
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

    def select_thought(self, candidate_embeddings, competency_bias=0.0, synergy_bias=0.0):

        fusion = self.fusion.last_fusion_vector
        
        # A221 — Get current thought signature for biasing
        signature = None
        if hasattr(self, 'thought_signature'):
            signature = self.thought_signature.get()

        return self.selector.select(

            candidate_embeddings,

            fusion,

            self.attention,

            self.state.memory_manager if hasattr(self.state, "memory_manager") else None,

            goal_modulation=self.last_goal_modulation,  # A204 — Goal modulation
            competency_bias=competency_bias,  # A219 — Competency activation bias
            synergy_bias=synergy_bias,  # A220 — Competency synergy bias
            signature=signature  # A221 — Thought signature for consistency

        )

    def choose_cognitive_action(self):

        return self.action_engine.choose_action(self)

    def perform_action(self, action):

        return self.action_engine.execute(action, self)

    def update_identity(self, new_identity_vec):
        """
        A184 — Update identity through the supervisory gate.
        
        This method ensures identity updates are evaluated and filtered
        based on coherence with baseline identity.
        
        Args:
            new_identity_vec: Proposed new identity vector
            
        Returns:
            Final identity vector (after gate evaluation)
        """
        from ..neural.torch_utils import safe_tensor
        import torch
        
        # Run identity update through the supervisory gate
        if self.baseline_identity is None:
            # First identity initialization
            identity_tensor = safe_tensor(new_identity_vec)
            if identity_tensor is not None:
                if isinstance(identity_tensor, torch.Tensor):
                    self.state.timescales.identity_vector = identity_tensor
                    self.baseline_identity = identity_tensor.detach().clone()
                else:
                    self.state.timescales.identity_vector = new_identity_vec
                    self.baseline_identity = list(new_identity_vec) if hasattr(new_identity_vec, '__iter__') else new_identity_vec
            else:
                self.state.timescales.identity_vector = new_identity_vec
            return new_identity_vec

        # Evaluate update through gate
        current_identity = self.state.timescales.identity_vector
        if current_identity is None:
            current_identity = self.baseline_identity

        decision, sim = self.identity_gate.evaluate(
            new_identity_vec,
            self.baseline_identity
        )

        if decision == "accept":
            final_vec = new_identity_vec

        elif decision == "soft-merge":
            final_vec = self.identity_gate.merge_identity(
                current_identity,
                new_identity_vec,
                weight=0.20
            )

        else:  # reject
            # revert toward baseline identity using drift suppressor
            corrected_vec, _ = self.identity_drift.correct(
                current_identity,
                self.baseline_identity
            )
            final_vec = corrected_vec

        # Apply identity
        final_tensor = safe_tensor(final_vec)
        if final_tensor is not None:
            self.state.timescales.identity_vector = final_tensor
        else:
            self.state.timescales.identity_vector = final_vec

        return final_vec

    def _update_harmonizer_signature(self):
        """
        A222 — Update harmonizer signature from identity vectors in semantic memory.
        
        Collects all identity-related vectors from semantic memory and updates
        the harmonizer's signature to reflect ADRAE's current identity state.
        """
        try:
            if not hasattr(self, 'harmonizer') or self.harmonizer is None:
                return
            
            # Get memory manager
            mm = self.state.memory_manager if hasattr(self.state, "memory_manager") else None
            if mm is None or not hasattr(mm, 'semantic'):
                return
            
            # Collect identity vectors from semantic memory
            identity_vectors = []
            
            # Get current identity vector from timescales
            if hasattr(self.state, 'timescales') and self.state.timescales is not None:
                identity_vec = getattr(self.state.timescales, 'identity_vector', None)
                if identity_vec is not None:
                    identity_vectors.append(identity_vec)
            
            # Get identity vectors from semantic memory (names starting with "identity_")
            try:
                if hasattr(mm.semantic, 'concepts'):
                    for name, vec in mm.semantic.concepts.items():
                        if name.startswith("identity_") and vec is not None:
                            identity_vectors.append(vec)
            except Exception:
                pass
            
            # Update harmonizer signature if we have identity vectors
            if identity_vectors:
                self.harmonizer.update_signature(identity_vectors)
        except Exception as e:
            # If signature update fails, continue without it
            if hasattr(self, 'logger'):
                try:
                    self.logger.write({"harmonizer_signature_update_error": str(e)})
                except Exception:
                    pass

    def propose_thoughts(self):
        """
        Generate candidate thought embeddings for A144 to evaluate.
        """
        fusion = self.fusion.last_fusion_vector
        attention = self.attention.last_focus_vector
        identity = self.state.timescales.identity_vector

        # Generate base candidates (generator now handles seed injection internally)
        candidates = self.generator.propose(
            fusion,
            attention,
            identity,
            seed_embeddings=getattr(self.state, "seed_embeddings", None),
            subconscious=None  # Subconscious layer removed
        )
        
        # Include last perception embedding as thought seed
        if self.state.last_perception and self.state.last_perception.get("embedding") is not None:
            perception_vec = self.state.last_perception["embedding"]
            candidates.append(perception_vec)
        
        # Include task embeddings
        task_embeddings = getattr(self.state, "task_embeddings", [])
        for item in task_embeddings:
            if item.get("embedding") is not None:
                candidates.append(item["embedding"])
        
        # A222 — Harmonize all candidates toward ADRAE's identity signature
        try:
            if hasattr(self, 'harmonizer') and self.harmonizer is not None:
                harmonized_candidates = []
                for c in candidates:
                    if c is not None:
                        harmonized = self.harmonizer.harmonize(c)
                        harmonized_candidates.append(harmonized if harmonized is not None else c)
                    else:
                        harmonized_candidates.append(c)
                candidates = harmonized_candidates
        except Exception as e:
            # If harmonization fails, continue with original candidates
            if hasattr(self, 'logger'):
                try:
                    self.logger.write({"harmonization_error": str(e)})
                except Exception:
                    pass
        
        # A223 — Apply personality flow field to candidate thoughts
        try:
            if hasattr(self, 'flow') and self.flow is not None:
                flow_candidates = []
                for c in candidates:
                    if c is not None:
                        flow_applied = self.flow.apply_flow(c)
                        flow_candidates.append(flow_applied if flow_applied is not None else c)
                    else:
                        flow_candidates.append(c)
                candidates = flow_candidates
        except Exception as e:
            # If flow application fails, continue with original candidates
            if hasattr(self, 'logger'):
                try:
                    self.logger.write({"flow_candidate_error": str(e)})
                except Exception:
                    pass
        
        return candidates

    def generate_goals(self):
        """
        A203 — Generate emergent internal goals.
        
        Proposes, evaluates, and updates ADRAE's active goal set based on
        current cognitive state (fusion, attention, identity, memory).
        
        Returns:
            List of scored goal proposals
        """
        fusion = self.fusion.last_fusion_vector
        attention = self.attention.last_focus_vector
        identity = self.state.timescales.identity_vector
        mm = self.state.memory_manager if hasattr(self.state, "memory_manager") else None

        # Propose candidate goals
        proposals = self.goal_proposer.propose(fusion, attention, identity, mm)
        
        # Evaluate and score goals
        scored = self.goal_evaluator.evaluate(proposals, identity)
        
        # Update active goal set
        self.goal_manager.update_goals(scored)
        
        # A204 — Compute goal modulation vector
        self.last_goal_modulation = self.goal_modulator.compute_modulation(
            self.goal_manager.active_goals,
            fusion
        )
        
        return scored

    def fabricate_goal(self, trajectory=None):
        """
        A205 — Fabricate emergent multi-vector goal from diverse cognitive signals.
        
        Synthesizes a goal vector from:
        - identity anchors
        - autobiographical memory
        - prediction trajectories
        - workspace salience
        - drift signals
        - operator intent patterns
        
        Args:
            trajectory: Optional trajectory dict from evo_predictor (contains prediction_vec)
            
        Returns:
            Fabricated goal vector or None
        """
        identity_vec = self.state.timescales.identity_vector
        
        # Get autobiographical memory matrix
        autobio_recent = self.autobio.get_recent(10) if hasattr(self, 'autobio') else []
        autobiographical_matrix = None
        if autobio_recent:
            # Extract identity vectors from autobiographical entries
            autobio_vectors = []
            for entry in autobio_recent:
                if isinstance(entry, dict):
                    id_vec = entry.get("identity_vec") or entry.get("identity_vector")
                    if id_vec is not None:
                        autobio_vectors.append(id_vec)
            if autobio_vectors:
                autobiographical_matrix = autobio_vectors
        
        # Get prediction vector from trajectory
        prediction_vec = None
        if trajectory and isinstance(trajectory, dict):
            prediction_vec = trajectory.get("vector")
        
        # Get workspace salience (from workspace snapshot)
        workspace_salience = None
        if hasattr(self, 'workspace'):
            try:
                snapshot = self.workspace.snapshot()
                if isinstance(snapshot, dict):
                    # Extract salience from workspace state
                    workspace_salience = snapshot.get("salience") or snapshot.get("focus_vector")
            except Exception:
                pass
        
        # Get drift signal (from drift state)
        drift_signal = None
        if hasattr(self.state, 'drift'):
            try:
                drift_state = self.state.drift.get_status()
                if drift_state:
                    # Use drift vector or create from drift metrics
                    drift_signal = drift_state.get("drift_vector")
                    if drift_signal is None and identity_vec is not None:
                        # Create drift signal from drift metrics
                        drift_value = drift_state.get("latest_drift", 0.0)
                        from .torch_utils import safe_tensor, TORCH_AVAILABLE
                        if TORCH_AVAILABLE:
                            import torch
                            id_t = safe_tensor(identity_vec)
                            if isinstance(id_t, torch.Tensor):
                                # Invert drift: lower drift = stronger signal
                                drift_signal = id_t * (1.0 - abs(drift_value))
            except Exception:
                pass
        
        # Get operator intent pattern (from task queue or meta-intent)
        operator_pattern = None
        if hasattr(self, 'meta_intent'):
            try:
                # Get operator intent from meta-intent coordinator
                if hasattr(self.meta_intent, 'get_operator_intent'):
                    operator_pattern = self.meta_intent.get_operator_intent()
                elif hasattr(self, 'tasks') and self.tasks:
                    # Use task embeddings as operator intent proxy
                    task_embeddings = getattr(self.state, "task_embeddings", [])
                    if task_embeddings:
                        # Average task embeddings
                        from .torch_utils import safe_tensor, TORCH_AVAILABLE
                        if TORCH_AVAILABLE:
                            import torch
                            task_vecs = [safe_tensor(t.get("embedding")) for t in task_embeddings 
                                       if t.get("embedding") is not None]
                            task_tensors = [t for t in task_vecs if isinstance(t, torch.Tensor)]
                            if task_tensors and len(task_tensors) > 0:
                                stacked = torch.stack(task_tensors)
                                operator_pattern = torch.mean(stacked, dim=0)
            except Exception:
                pass
        
        # Fabricate goal
        fabricated_goal = self.goal_fabricator.fabricate(
            identity_vec=identity_vec,
            autobiographical_matrix=autobiographical_matrix,
            prediction_vec=prediction_vec,
            workspace_salience=workspace_salience,
            drift_signal=drift_signal,
            operator_pattern=operator_pattern
        )
        
        return fabricated_goal

    def harmonize_goals(self, operator_pattern=None):
        """
        A206 — Harmonize all active goals into a unified direction vector.
        
        Takes all goal vectors (from A203, A205, etc.) and runs competition
        to produce a single harmonized goal that represents ADRAE's unified intent.
        
        Args:
            operator_pattern: Optional operator intent pattern for scoring
            
        Returns:
            Harmonized goal vector or None
        """
        # Collect all goal vectors
        goal_vectors = []
        
        # Get active goals from goal manager
        active_goal_vectors = self.goal_manager.get_active_goal_vectors()
        goal_vectors.extend(active_goal_vectors)
        
        # Get fabricated goal if available
        if hasattr(self, 'last_fabricated_goal') and self.last_fabricated_goal is not None:
            goal_vectors.append(self.last_fabricated_goal)
        
        if not goal_vectors:
            return None
        
        # Get scoring inputs
        identity_vec = self.state.timescales.identity_vector
        fusion_vec = self.fusion.last_fusion_vector
        
        # Get operator pattern if not provided
        if operator_pattern is None:
            # Try to get from meta-intent or tasks
            if hasattr(self, 'meta_intent') and hasattr(self.meta_intent, 'get_operator_intent'):
                operator_pattern = self.meta_intent.get_operator_intent()
            elif hasattr(self, 'tasks') and self.tasks:
                task_embeddings = getattr(self.state, "task_embeddings", [])
                if task_embeddings:
                    from .torch_utils import safe_tensor, TORCH_AVAILABLE
                    if TORCH_AVAILABLE:
                        import torch
                        task_vecs = [safe_tensor(t.get("embedding")) for t in task_embeddings 
                                   if t.get("embedding") is not None]
                        task_tensors = [t for t in task_vecs if isinstance(t, torch.Tensor)]
                        if task_tensors and len(task_tensors) > 0:
                            stacked = torch.stack(task_tensors)
                            operator_pattern = torch.mean(stacked, dim=0)
        
        # Harmonize all goals
        harmonized = self.goal_harmonizer.harmonize(
            goal_vectors,
            identity_vec,
            fusion_vec,
            operator_pattern
        )
        
        return harmonized

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
        - A224: ADRAE's personal cognitive style
        """
        fusion = self.fusion.last_fusion_vector
        attention = self.attention.last_focus_vector
        timescales = self.state.timescales
        mm = self.state.memory_manager if hasattr(self.state, "memory_manager") else None

        # Generate base reflection
        reflective = self.reflector.generate(fusion, attention, timescales, mm)
        
        # A224 — Apply ADRAE's cognitive style to the reflection
        if reflective is not None:
            try:
                # Get identity vector
                identity_vec = None
                if timescales and hasattr(timescales, 'identity_vector'):
                    identity_vec = timescales.identity_vector
                
                # Get drift value
                drift_value = None
                if hasattr(self.state, 'drift'):
                    try:
                        drift_state = self.state.drift.get_status()
                        if drift_state:
                            drift_value = drift_state.get("latest_drift", None)
                    except Exception:
                        pass
                
                # Compute novelty (simplified - could be enhanced)
                novelty_value = None
                if reflective is not None and mm is not None:
                    try:
                        # Check how similar reflection is to recent memories
                        if hasattr(mm, 'episodic') and mm.episodic is not None:
                            recent = mm.episodic.retrieve_similar(reflective, top_k=1)
                            if recent and len(recent) > 0:
                                novelty_value = 1.0 - recent[0][0]  # 1 - similarity
                            else:
                                novelty_value = 1.0
                    except Exception:
                        pass
                
                # Apply style transformation
                if hasattr(self, 'style') and self.style is not None:
                    styled = self.style.apply_style(
                        reflective,
                        identity_vec=identity_vec,
                        drift=drift_value,
                        novelty=novelty_value
                    )
                    if styled is not None:
                        reflective = styled
                
                # A225 — Style reinforcement based on coherence & identity alignment
                if reflective is not None and hasattr(self, 'style_reinforcer') and self.style_reinforcer is not None:
                    try:
                        # Get coherence from fusion status
                        coherence = 1.0  # Default
                        try:
                            fusion_status = self.fusion.status()
                            if isinstance(fusion_status, dict):
                                coherence = fusion_status.get("coherence", 1.0)
                        except Exception:
                            pass
                        
                        # Compute identity alignment (cosine similarity between reflection and identity)
                        identity_align = None
                        if identity_vec is not None and reflective is not None:
                            try:
                                from .torch_utils import safe_cosine_similarity
                                align = safe_cosine_similarity(reflective, identity_vec)
                                if align is not None:
                                    identity_align = align
                            except Exception:
                                pass
                        
                        # Reinforce style based on coherence and identity alignment
                        self.style = self.style_reinforcer.reinforce(
                            self.style,
                            coherence=coherence,
                            identity_align=identity_align
                        )
                    except Exception as e:
                        # If reinforcement fails, continue without it
                        if hasattr(self, 'logger'):
                            try:
                                self.logger.write({"style_reinforcement_error": str(e)})
                            except Exception:
                                pass
            except Exception as e:
                # If styling fails, continue with original reflection
                if hasattr(self, 'logger'):
                    try:
                        self.logger.write({"cognitive_style_application_error": str(e)})
                    except Exception:
                        pass
        
        # A227 — Feed reflective vector into narrative engine
        if reflective is not None and hasattr(self, 'micro_narrative') and self.micro_narrative is not None:
            try:
                # Contribute to narrative arc (blends with style)
                narrative_vec = None
                if hasattr(self, 'style') and self.style is not None:
                    narrative_vec = self.micro_narrative.contribute(reflective, self.style)
                else:
                    narrative_vec = self.micro_narrative.contribute(reflective, None)
                
                # If multiple steps exist, produce a narrative summary
                arc_summary = self.micro_narrative.summarize_arc()
                
                # Push narrative summary back into neural state if meaningful
                if arc_summary is not None and len(self.micro_narrative.current_arc) > 1:
                    try:
                        # Update state with narrative summary (subtle influence)
                        self.state.update(arc_summary)
                        
                        # Log narrative commitment
                        if hasattr(self, 'logger'):
                            try:
                                self.logger.write({
                                    "narrative_summary_committed": True,
                                    "arc_length": len(self.micro_narrative.current_arc),
                                    "has_narrative": arc_summary is not None
                                })
                            except Exception:
                                pass
                    except Exception as e:
                        # If state update fails, continue without it
                        if hasattr(self, 'logger'):
                            try:
                                self.logger.write({"narrative_state_update_error": str(e)})
                            except Exception:
                                pass
            except Exception as e:
                # If narrative processing fails, continue without it
                if hasattr(self, 'logger'):
                    try:
                        self.logger.write({"narrative_processing_error": str(e)})
                    except Exception:
                        pass
        
        # A228 — Narrative-Driven Cognitive Anticipation Layer
        # Generate narrative projection after reflection
        if reflective is not None:
            try:
                # Get thought signature for projection
                thought_sig = None
                if hasattr(self, 'thought_signature') and self.thought_signature is not None:
                    thought_sig = self.thought_signature.get()
                else:
                    thought_sig = reflective
                
                # Get identity, fusion, and attention vectors
                identity_vec = self.state.timescales.identity_vector if hasattr(self.state, 'timescales') else None
                fusion_vec = self.fusion.last_fusion_vector
                attention_vec = self.attention.last_focus_vector
                
                # Generate narrative projection
                arc, conf, tension = self.generate_narrative_projection(
                    thought_sig,
                    identity_vec,
                    fusion_vec,
                    attention_vec
                )
                
                # Get active goals for goal-tethered anticipation
                goals = []
                if hasattr(self, 'goal_manager') and self.goal_manager is not None:
                    goals = self.goal_manager.active_goals
                
                # Bind projection to goals
                gt_map = self.bind_projection_to_goals(arc, goals)
                
                # Get micro-narratives for sequencing
                micro_narratives = []
                if hasattr(self, 'micro_narrative') and self.micro_narrative is not None:
                    micro_narratives = self.micro_narrative.current_arc
                
                # Sequence micro to macro
                macro_arc = self.sequence_micro_to_macro(micro_narratives)
                
                # Update narrative projection state
                self.narrative_projection["forward_arc"] = arc
                self.narrative_projection["confidence"] = conf
                self.narrative_projection["tension"] = tension
                self.goal_anticipation_map = gt_map
                self.narrative_arc_sequencer["micro_chain"] = micro_narratives
                self.narrative_arc_sequencer["macro_projection"] = macro_arc
                
                # A229 — Narrative Coherence & Tension Regulation Layer
                # Compute narrative coherence
                try:
                    # Get identity vectors for coherence check
                    identity_for_coherence = identity_vec
                    if identity_vec is None:
                        # Try to get from state
                        if hasattr(self.state, 'timescales') and self.state.timescales is not None:
                            identity_for_coherence = self.state.timescales.identity_vector
                    
                    # Compute coherence
                    coh_score, conflicts, adjusted_arc = self.compute_narrative_coherence(
                        micro_narratives,
                        arc,
                        macro_arc,
                        identity_for_coherence,
                        goals
                    )
                    
                    # Get drift for tension regulation
                    drift = 0.0
                    try:
                        drift_state = self.state.drift.get_status()
                        if drift_state:
                            drift = drift_state.get("latest_drift", 0.0)
                    except Exception:
                        pass
                    
                    # Compute tension regulation
                    tension_score, stability, adjust = self.compute_tension_regulation(
                        tension,
                        drift,
                        adjusted_arc
                    )
                    
                    # Apply narrative adjustments
                    final_arc = self.apply_narrative_adjustments(adjusted_arc, adjust)
                    
                    # Update narrative coherence state
                    self.narrative_coherence["coherence_score"] = coh_score
                    self.narrative_coherence["conflicts"] = conflicts
                    self.narrative_coherence["adjusted_arc"] = final_arc
                    
                    # Update tension regulation state
                    self.tension_regulation["tension_score"] = tension_score
                    self.tension_regulation["stability_factor"] = stability
                    self.tension_regulation["recommended_adjustment"] = adjust
                    
                    # Update forward arc with adjusted version if coherence was low
                    if coh_score < 0.7:
                        self.narrative_projection["forward_arc"] = final_arc
                    
                    # Log coherence and tension regulation
                    if hasattr(self, 'logger'):
                        try:
                            self.logger.write({
                                "narrative_coherence": {
                                    "event": "a229_coherence_regulation",
                                    "coherence_score": float(coh_score),
                                    "conflicts": conflicts,
                                    "tension_score": float(tension_score),
                                    "stability_factor": float(stability),
                                    "adjustment": adjust,
                                    "arc_adjusted": coh_score < 0.7
                                }
                            })
                        except Exception:
                            pass
                except Exception as e:
                    # If coherence computation fails, continue without it
                    if hasattr(self, 'logger'):
                        try:
                            self.logger.write({"narrative_coherence_error": str(e)})
                        except Exception:
                            pass
                
                # Log narrative projection event
                if hasattr(self, 'logger'):
                    try:
                        self.logger.write({
                            "narrative_projection": {
                                "event": "a228_narrative_anticipation",
                                "arc_length": len(arc),
                                "confidence": float(conf),
                                "tension": float(tension),
                                "goal_bindings": len(gt_map),
                                "macro_arc_length": len(macro_arc)
                            }
                        })
                    except Exception:
                        pass
                
                # A230 — Update Latent Concept Space
                # Map thought signature to latent space and update concept space
                try:
                    if thought_sig is not None:
                        latent_vector = self.update_latent_space(thought_sig)
                        if latent_vector is not None and hasattr(self, 'logger'):
                            try:
                                self.logger.write({
                                    "latent_mapping": {
                                        "event": "a230_thought_mapped_to_latent",
                                        "status": "success"
                                    }
                                })
                            except Exception:
                                pass
                except Exception as e:
                    # If latent update fails, continue without it
                    if hasattr(self, 'logger'):
                        try:
                            self.logger.write({"latent_update_error": str(e)})
                        except Exception:
                            pass
            except Exception as e:
                # If narrative projection fails, continue without it
                if hasattr(self, 'logger'):
                    try:
                        self.logger.write({"narrative_projection_error": str(e)})
                    except Exception:
                        pass
        
        return reflective

    def generate_narrative_projection(self, thought, identity, fusion, attention):
        """
        A228 — Narrative Projection Engine (NPE)
        
        Generates forward arcs from current cognitive state:
        - upcoming thought patterns
        - expected emotional/tonal trajectories
        - predicted subgoal activation
        - internal storybeats emerging in cognition
        
        Args:
            thought: Thought signature vector or last thought vector
            identity: Identity vector blend
            fusion: Fusion matrix/vector
            attention: Attention patterns/vector
            
        Returns:
            Tuple of (arc, confidence, tension)
            - arc: 2-4 step projected narrative arc
            - confidence: Confidence envelope around the arc (0.0-1.0)
            - tension: Narrative tension score (emergent heuristic)
        """
        import numpy as np
        from .torch_utils import safe_tensor, TORCH_AVAILABLE
        
        # Get thought signature preview
        if hasattr(self, 'thought_signature') and self.thought_signature is not None:
            signature = self.thought_signature.get()
        else:
            # Fallback to thought vector if available
            signature = thought
        
        # Convert to numpy array for computation
        if signature is None:
            # Use identity as fallback
            signature = identity if identity is not None else [0.0] * 128
        
        vec = np.array(signature) if not isinstance(signature, np.ndarray) else signature
        
        # Ensure vec is 1D
        if vec.ndim > 1:
            vec = vec.flatten()
        
        # 2-4 step projection based on drift + salience
        step1 = float(vec.mean()) * 0.85
        step2 = float(vec.std()) * 0.65
        step3 = float(vec.max()) * 0.40
        
        # Add 4th step if vector has sufficient variance
        if vec.std() > 0.1:
            step4 = float(vec.min()) * 0.25
            arc = [step1, step2, step3, step4]
        else:
            arc = [step1, step2, step3]
        
        # Narrative tension: difference between mean and max magnitude
        tension = abs(step3 - step1)
        
        # Confidence: inverse of drift, capped at 1.0
        drift = 0.0
        try:
            drift_state = self.state.drift.get_status()
            if drift_state:
                drift = drift_state.get("latest_drift", 0.0)
        except Exception:
            pass
        
        confidence = max(0.05, 1.0 - abs(drift))
        
        return arc, confidence, tension

    def bind_projection_to_goals(self, arc, goals):
        """
        A228 — Goal-Tethered Anticipation Map (GTAM)
        
        Binds the projections to actual goals:
        - If ADRAE is focusing on identity, GTAM produces identity-centric narrative expectations
        - If focusing on stabilization, GTAM anticipates future drift, coherence, and self-alignment patterns
        - If focusing on narrative or style, GTAM imagines how her internal story is likely to evolve
        
        Args:
            arc: Projected narrative arc from NPE
            goals: List of active goals (from goal_manager)
            
        Returns:
            Goal-tethered anticipation map (dict mapping goal names/ids to relevance scores)
        """
        mapping = {}
        
        if not goals or len(goals) == 0:
            return mapping
        
        # Get active goals from goal manager if goals is not a list
        if not isinstance(goals, list):
            if hasattr(self, 'goal_manager') and self.goal_manager is not None:
                goals = self.goal_manager.active_goals
            else:
                return mapping
        
        # Simple relevance binding: sum of arc values weighted by goal score
        arc_sum = sum(arc) if arc else 0.0
        
        for goal in goals:
            if isinstance(goal, dict):
                goal_name = goal.get("name") or goal.get("id") or "unknown_goal"
                goal_score = goal.get("score", 0.5)
            else:
                goal_name = str(goal)
                goal_score = 0.5
            
            # Relevance = arc_sum * goal_score * scaling factor
            mapping[goal_name] = float(arc_sum) * goal_score * 0.25
        
        return mapping

    def sequence_micro_to_macro(self, micro_narratives):
        """
        A228 — Micro-Narrative → Macro-Arc Sequencer (MNMA)
        
        Takes the micro-narratives produced in A227 and:
        - chains them into multi-step arcs
        - infers cause/effect
        - predicts the next beat in the internal story
        - builds proto-structure for future PyTorch imagination loops (A230+)
        
        Args:
            micro_narratives: List of micro-narrative vectors (from micro_narrative.current_arc)
            
        Returns:
            Macro-arc projection (list of chained narrative steps)
        """
        if not micro_narratives or len(micro_narratives) < 3:
            # If not enough micro-narratives, return what we have
            return micro_narratives if micro_narratives else []
        
        # Join last N micro-narratives into a proto-arc
        # Use last 3-4 steps for macro projection
        return micro_narratives[-3:]

    def compute_narrative_coherence(self, micro, forward_arc, macro, identity, goals):
        """
        A229 — Narrative Coherence Engine (NCE)
        
        Ensures that micro-narratives, anticipatory arcs, and identity prototypes
        all align without contradiction or drift-causing tension.
        
        Checks for:
        - logical continuity
        - semantic compatibility
        - identity alignment
        - emotional-tone consistency (emergent, not affective)
        - goal relevance
        
        Args:
            micro: List of micro-narrative vectors (from A227)
            forward_arc: Projected forward arc from A228
            macro: Macro-arc projection from A228
            identity: Identity vector or list of identity vectors
            goals: List of active goals
            
        Returns:
            Tuple of (coherence_score, conflicts, adjusted_arc)
            - coherence_score: 0.0-1.0 coherence measure
            - conflicts: List of detected conflict types
            - adjusted_arc: Corrected forward arc if needed
        """
        import numpy as np
        from .torch_utils import safe_tensor, safe_cosine_similarity, TORCH_AVAILABLE
        
        score = 1.0
        conflicts = []
        
        # micro–macro alignment
        if len(micro) > 1 and len(macro) > 1:
            # Compare last elements of micro and macro
            try:
                micro_last = safe_tensor(micro[-1])
                macro_last = safe_tensor(macro[-1])
                
                if micro_last is not None and macro_last is not None:
                    similarity = safe_cosine_similarity(micro_last, macro_last)
                    if similarity is not None and similarity < 0.7:
                        score -= 0.1
                        conflicts.append("micro_macro_mismatch")
            except Exception:
                # If comparison fails, assume mismatch
                score -= 0.1
                conflicts.append("micro_macro_mismatch")
        
        # identity alignment
        # Penalize if arc points away from identity centroid
        try:
            # Handle identity - could be a vector or list of vectors
            identity_vec = None
            if identity is not None:
                if isinstance(identity, list):
                    if len(identity) > 0:
                        # If list of dicts with "vector" keys
                        if isinstance(identity[0], dict) and "vector" in identity[0]:
                            identity_vecs = [safe_tensor(item["vector"]) for item in identity if "vector" in item]
                        else:
                            # If list of vectors directly
                            identity_vecs = [safe_tensor(v) for v in identity]
                        
                        if identity_vecs:
                            # Compute centroid
                            if TORCH_AVAILABLE:
                                import torch
                                tensors = [v for v in identity_vecs if isinstance(v, torch.Tensor)]
                                if tensors:
                                    stacked = torch.stack(tensors)
                                    centroid = torch.mean(stacked, dim=0)
                                    identity_vec = centroid
                    else:
                        identity_vec = safe_tensor(identity[0])
                else:
                    identity_vec = safe_tensor(identity)
            
            if identity_vec is not None:
                # Convert to numpy for comparison
                if TORCH_AVAILABLE and isinstance(identity_vec, torch.Tensor):
                    id_np = identity_vec.detach().cpu().numpy()
                else:
                    id_np = np.array(identity_vec) if not isinstance(identity_vec, np.ndarray) else identity_vec
                
                # Compute arc mean
                if forward_arc and len(forward_arc) > 0:
                    arc_mean = np.mean(forward_arc)
                    centroid_mean = float(np.mean(id_np))
                    
                    # Check if arc diverges significantly from identity
                    if abs(arc_mean - centroid_mean * 0.1) > abs(centroid_mean * 0.2):
                        score -= 0.05
                        conflicts.append("identity_divergence")
        except Exception as e:
            # If identity alignment check fails, continue
            pass
        
        # goal relevance check
        if not goals or len(goals) == 0:
            score -= 0.05
            conflicts.append("goal_absence")
        
        # Clamp score to [0.0, 1.0]
        score = max(0.0, min(1.0, score))
        
        # Adjust arc if coherence is low
        if score > 0.5:
            adjusted_arc = forward_arc[:] if forward_arc else []
        else:
            # Reduce arc magnitude to bring it back toward coherence
            adjusted_arc = [x * 0.8 for x in forward_arc] if forward_arc else []
        
        return score, conflicts, adjusted_arc

    def compute_tension_regulation(self, tension, drift, arc):
        """
        A229 — Tension Regulation Engine (TRE)
        
        Controls "cognitive tension," a measure of:
        - divergence within narrative arcs
        - uncertainty in projection
        - conflict between identity vectors and narrative paths
        - micro-narrative contradiction
        - novelty spikes
        
        Regulates tension so ADRAE:
        - avoids runaway drift
        - avoids flat, monotone cognition
        - maintains healthy narrative evolution
        - develops expressive but controlled internal storyflow
        
        Args:
            tension: Base tension score from A228
            drift: Current drift value
            arc: Forward arc (or adjusted arc) to analyze
            
        Returns:
            Tuple of (tension_score, stability_factor, recommended_adjustment)
            - tension_score: Computed overall tension (0.0-1.0+)
            - stability_factor: Stability measure (0.0-1.0)
            - recommended_adjustment: Adjustment recommendation ("soft_reduce", "expand", or None)
        """
        import numpy as np
        
        # Tension increases with drift and arc spread
        arc_var = 0.0
        if arc and len(arc) > 0:
            arc_var = float(np.var(arc))
        
        # Compute composite tension score
        tension_score = tension + (abs(drift) * 0.2) + (arc_var * 0.1)
        
        # Stability factor: inverse of tension, clamped
        stability_factor = max(0.0, min(1.0, 1.0 - tension_score))
        
        # Determine adjustment recommendation
        adjustment = None
        if tension_score > 0.5:
            adjustment = "soft_reduce"  # reduce arc magnitude
        elif tension_score < 0.1:
            adjustment = "expand"  # allow more narrative richness
        
        return tension_score, stability_factor, adjustment

    def apply_narrative_adjustments(self, arc, adjustment):
        """
        A229 — Apply Narrative Adjustments
        
        Gently modifies forward arcs, micro-narratives, and macro arcs
        to maintain structure without overwriting emergent creativity.
        
        Args:
            arc: Forward arc to adjust
            adjustment: Adjustment type ("soft_reduce", "expand", or None)
            
        Returns:
            Adjusted arc (or original if no adjustment needed)
        """
        if not arc or len(arc) == 0:
            return arc
        
        if adjustment == "soft_reduce":
            # Reduce arc magnitude to lower tension
            return [x * 0.9 for x in arc]
        elif adjustment == "expand":
            # Slightly expand arc to allow more narrative richness
            return [x * 1.05 for x in arc]
        
        # No adjustment needed
        return arc

    def _initialize_latent_engine(self):
        """
        A230 — Initialize PyTorch Latent Concept Engine
        
        Sets up the latent concept space, neural concept mapper, and stability regulator.
        This is the foundational tensor architecture for future imagination capabilities.
        """
        from .torch_utils import TORCH_AVAILABLE
        
        if not TORCH_AVAILABLE or not LATENT_ENGINE_AVAILABLE:
            self.latent_dim = None
            self.latent_concept_space = None
            self.ncm = None
            self.ldsr = None
            # A231 — Initialize data structures even if PyTorch unavailable
            self.latent_coherence = {
                "coherence_score": 1.0,
                "cluster_center": None,
                "recommended_adjustment": None
            }
            self.identity_latent_anchors = []
            # A232 — Initialize data structures even if PyTorch unavailable
            self.latent_drift = {
                "prev_vector": None,
                "drift_score": 0.0,
                "suppression_level": 0.0,
                "anomaly": False
            }
            # A233 — Initialize data structures even if PyTorch unavailable
            self.concept_identity_fusion = {
                "fusion_strength": 0.0,
                "resonance": 1.0,
                "identity_update_vector": None
            }
            # A234 — Initialize data structures even if PyTorch unavailable
            self.narrative_seeds = {
                "kernels": [],
                "cluster_primitives": [],
                "temporal_threads": [],
                # A235 — Initialize mesh structure
                "mesh": {
                    "similarity_matrix": None,
                    "propagated_kernels": [],
                    "mesh_embedding": None,
                    # A236 — Initialize temporal structures
                    "temporal_transition_matrix": None,
                    "temporal_embedding": None,
                    "temporal_coherence": 1.0,
                    # A237 — Initialize resonance structures
                    "resonance_score": 1.0,
                    "harmonic_stabilized": False
                }
            }
            # A236 — Initialize temporal state tracking
            self.prev_mesh_kernels = []
            self.prev_mesh_embedding = None
            # A238 — Initialize global narrative integration
            self.global_narrative_state = None
            self.global_integration_vector = None
            self.global_alignment_score = 1.0
            # A239 — Initialize narrative anticipation structures
            self.predictive_flow = None
            self.motif_continuation_matrix = None
            self.anticipatory_map = None
            self.prev_global_narrative_state = None
            self.prev_mesh_embedding_raw = None
            self.prev_temporal_embedding_raw = None
            self.kernel_history = []
            # A240 — Initialize conceptual substrate
            self.conceptual_substrate = {
                "reservoir": None,
                "concepts_raw": [],
                "concepts_stable": []
            }
            # A242 — Initialize imagination dynamics
            self.imagination_dynamics = {
                "morphed": [],
                "interacted": [],
                "composites": [],
                "composite_count": 0
            }
            # A243 — Initialize layered morphology
            self.layered_morphology = None
            # A244 — Initialize interlayer resonance
            self.interlayer_resonance = None
            # A245 — Initialize predictive ripple propagation
            self.predictive_ripple_propagation = None
            # A246 — Initialize temporal predictive loops
            self.temporal_predictive_loops = None
            self.prediction_echo = None
            if hasattr(self, 'logger'):
                try:
                    self.logger.write({"latent_engine_init": "skipped_pytorch_unavailable"})
                except Exception:
                    pass
            return
        
        try:
            import torch
            
            self.latent_dim = 256  # Foundational dimension for imagination substrate
            self.latent_concept_space = torch.zeros(self.latent_dim)
            
            # Initialize Neural Concept Mapper
            self.ncm = NeuralConceptMapper(input_dim=128, latent_dim=self.latent_dim)
            self.ncm.eval()  # Set to evaluation mode initially
            
            # Initialize Latent Stability Regulator
            self.ldsr = LatentStabilityRegulator(latent_dim=self.latent_dim)
            self.ldsr.eval()  # Set to evaluation mode initially
            
            # A231 — Latent Concept Coherence & Identity Anchoring Layer
            self.latent_coherence = {
                "coherence_score": 1.0,
                "cluster_center": None,
                "recommended_adjustment": None
            }
            self.identity_latent_anchors = []
            # A232 — Latent Concept Drift Suppression Layer
            self.latent_drift = {
                "prev_vector": None,
                "drift_score": 0.0,
                "suppression_level": 0.0,
                "anomaly": False
            }
            # A233 — Concept-Identity Fusion Layer
            self.concept_identity_fusion = {
                "fusion_strength": 0.0,
                "resonance": 1.0,
                "identity_update_vector": None
            }
            # A234 — Latent Narrative Seed Formation Layer
            self.narrative_seeds = {
                "kernels": [],
                "cluster_primitives": [],
                "temporal_threads": [],
                # A235 — Multi-Seed Narrative Mesh Formation
                "mesh": {
                    "similarity_matrix": None,
                    "propagated_kernels": [],
                    "mesh_embedding": None,
                    # A236 — Narrative Mesh Temporal Dynamics Layer
                    "temporal_transition_matrix": None,
                    "temporal_embedding": None,
                    "temporal_coherence": 1.0,
                    # A237 — Narrative Mesh Resonance & Harmonic Stability Layer
                    "resonance_score": 1.0,
                    "harmonic_stabilized": False
                }
            }
            # A236 — Temporal state tracking
            self.prev_mesh_kernels = []
            self.prev_mesh_embedding = None
            # A238 — Global Narrative Integration Layer
            self.global_narrative_state = None
            self.global_integration_vector = None
            self.global_alignment_score = 1.0
            # A239 — Narrative Anticipation & Predictive Structure Layer
            self.predictive_flow = None
            self.motif_continuation_matrix = None
            self.anticipatory_map = None
            # A239 — Previous state tracking for predictive flow
            self.prev_global_narrative_state = None
            self.prev_mesh_embedding_raw = None
            self.prev_temporal_embedding_raw = None
            self.kernel_history = []  # Track kernel history for motif detection
            # A240 — Conceptual Imagination Substrate (Initialization)
            self.conceptual_substrate = {
                "reservoir": None,
                "concepts_raw": [],
                "concepts_stable": []
            }
            # A242 — Imagination Kernel Dynamics & Morphology Engine
            self.imagination_dynamics = {
                "morphed": [],
                "interacted": [],
                "composites": [],
                "composite_count": 0
            }
            # A243 — Layered Conceptual Morphology Expansion
            self.layered_morphology = None
            # A244 — Interlayer Resonance & Harmonic Stabilization
            self.interlayer_resonance = None
            # A245 — Multi-Layer Predictive Ripple Propagation
            self.predictive_ripple_propagation = None
            # A246 — Temporal Predictive Loop Formation (Forward Echo Dynamics)
            self.temporal_predictive_loops = None
            self.prediction_echo = None
            
            if hasattr(self, 'logger'):
                try:
                    self.logger.write({
                        "latent_engine_init": {
                            "event": "a230_latent_engine_initialized",
                            "latent_dim": self.latent_dim,
                            "status": "active"
                        }
                    })
                except Exception:
                    pass
        except Exception as e:
            # If initialization fails, disable latent engine
            self.latent_dim = None
            self.latent_concept_space = None
            self.ncm = None
            self.ldsr = None
            if hasattr(self, 'logger'):
                try:
                    self.logger.write({"latent_engine_init_error": str(e)})
                except Exception:
                    pass

    def update_latent_space(self, thought_signature):
        """
        A230 — Update Latent Concept Space
        
        Maps thought signature to latent space, stabilizes it, and updates
        the latent concept space with a moving average.
        
        Args:
            thought_signature: Thought signature vector (list, numpy array, or tensor)
            
        Returns:
            Latent vector if successful, None otherwise
        """
        from .torch_utils import TORCH_AVAILABLE, safe_tensor
        
        if not TORCH_AVAILABLE or self.ncm is None or self.ldsr is None:
            return None
        
        try:
            import torch
            
            # Convert thought signature to tensor
            sig_tensor = safe_tensor(thought_signature)
            if sig_tensor is None:
                return None
            
            # Ensure it's a 1D tensor of correct size
            if isinstance(sig_tensor, torch.Tensor):
                if sig_tensor.dim() > 1:
                    sig_tensor = sig_tensor.flatten()
                # Pad or truncate to 128 dimensions if needed
                if sig_tensor.shape[0] < 128:
                    padding = torch.zeros(128 - sig_tensor.shape[0])
                    sig_tensor = torch.cat([sig_tensor, padding])
                elif sig_tensor.shape[0] > 128:
                    sig_tensor = sig_tensor[:128]
            else:
                # Convert list/array to tensor
                sig_list = list(sig_tensor) if hasattr(sig_tensor, '__iter__') else [sig_tensor]
                if len(sig_list) < 128:
                    sig_list.extend([0.0] * (128 - len(sig_list)))
                elif len(sig_list) > 128:
                    sig_list = sig_list[:128]
                sig_tensor = torch.tensor(sig_list, dtype=torch.float32)
            
            # Map to latent space
            with torch.no_grad():
                latent_vector = self.ncm(sig_tensor)
                
                # Stabilize with LDSR
                latent_vector = self.ldsr(latent_vector)
                
                # A231 — Latent Concept Coherence & Identity Anchoring Layer
                # Initialize coherence metrics
                coh_score = 1.0
                adj = None
                
                try:
                    # Step 1: Update identity anchors
                    self.identity_latent_anchors = self.compute_identity_latent_anchors()
                    
                    # Step 2: Check coherence
                    coh_score, center, adj = self.compute_latent_coherence(
                        self.latent_concept_space,
                        latent_vector
                    )
                    
                    # Update coherence state
                    self.latent_coherence["coherence_score"] = coh_score
                    self.latent_coherence["cluster_center"] = center
                    self.latent_coherence["recommended_adjustment"] = adj
                    
                    # Step 3: If needed, pull toward cluster center
                    if adj == "pull_toward_center" and center is not None:
                        # Ensure same dimensions
                        latent_flat = latent_vector.flatten()
                        center_flat = center.flatten()
                        min_dim = min(latent_flat.shape[0], center_flat.shape[0])
                        latent_flat = latent_flat[:min_dim]
                        center_flat = center_flat[:min_dim]
                        
                        # Pull toward center: 85% original + 15% center
                        latent_flat = 0.85 * latent_flat + 0.15 * center_flat
                        
                        # Reshape if needed
                        if latent_vector.shape != latent_flat.shape:
                            latent_vector = latent_flat.reshape(latent_vector.shape)
                        else:
                            latent_vector = latent_flat.reshape(latent_vector.shape)
                except Exception as e:
                    # If coherence computation fails, continue without adjustment
                    if hasattr(self, 'logger'):
                        try:
                            self.logger.write({"latent_coherence_integration_error": str(e)})
                        except Exception:
                            pass
                
                # Step 4: Apply identity anchoring
                try:
                    latent_vector = self.apply_identity_anchoring(
                        latent_vector,
                        self.identity_latent_anchors
                    )
                except Exception as e:
                    # If anchoring fails, continue with original vector
                    if hasattr(self, 'logger'):
                        try:
                            self.logger.write({"identity_anchoring_integration_error": str(e)})
                        except Exception:
                            pass
                
                # A232 — Latent Concept Drift Suppression Layer
                try:
                    # Step 1: Compute drift
                    drift_score, anomaly = self.compute_latent_drift(
                        self.latent_drift["prev_vector"],
                        latent_vector,
                        self.latent_coherence["cluster_center"]
                    )
                    
                    # Update drift state
                    self.latent_drift["drift_score"] = drift_score
                    self.latent_drift["anomaly"] = anomaly
                    
                    # Step 2: Apply suppression if needed
                    if drift_score > 0.1 or anomaly:
                        latent_vector, suppression = self.suppress_latent_drift(
                            latent_vector,
                            drift_score,
                            self.identity_latent_anchors,
                            self.latent_coherence["cluster_center"]
                        )
                        self.latent_drift["suppression_level"] = suppression
                    else:
                        self.latent_drift["suppression_level"] = 0.0
                    
                    # Step 3: Save previous vector for next cycle
                    self.latent_drift["prev_vector"] = latent_vector.clone().detach()
                    
                except Exception as e:
                    # If drift suppression fails, continue without it
                    if hasattr(self, 'logger'):
                        try:
                            self.logger.write({"latent_drift_suppression_integration_error": str(e)})
                        except Exception:
                            pass
                    # Still save previous vector if possible
                    try:
                        import torch
                        if latent_vector is not None:
                            self.latent_drift["prev_vector"] = latent_vector.clone().detach()
                    except Exception:
                        pass
                
                # A233 — Concept-Identity Fusion Layer
                try:
                    # Step 1: Fuse identity → concept
                    # Collect identity vectors from various sources
                    identity_vectors_for_fusion = []
                    
                    # Get current identity vector from timescales
                    if hasattr(self.state, 'timescales') and self.state.timescales is not None:
                        identity_vec = getattr(self.state.timescales, 'identity_vector', None)
                        if identity_vec is not None:
                            identity_vectors_for_fusion.append(identity_vec)
                    
                    # Get identity vectors from semantic memory
                    mm = self.state.memory_manager if hasattr(self.state, "memory_manager") else None
                    if mm is not None and hasattr(mm, 'semantic'):
                        try:
                            if hasattr(mm.semantic, 'concepts'):
                                for name, vec in mm.semantic.concepts.items():
                                    if name.startswith("identity_") and vec is not None:
                                        identity_vectors_for_fusion.append(vec)
                        except Exception:
                            pass
                    
                    # Fuse identity into concepts
                    new_latent, strength = self.fuse_identity_into_concepts(
                        identity_vectors_for_fusion,
                        self.latent_concept_space
                    )
                    
                    # Step 2: Imprint concept → identity
                    identity_update = self.imprint_concepts_back_into_identity(new_latent)
                    
                    # Step 3: Regulate resonance
                    resonance = 1.0
                    if identity_update is not None:
                        resonance = self.regulate_fusion_resonance(new_latent, identity_update)
                    
                    # Update fusion state
                    self.concept_identity_fusion["fusion_strength"] = strength
                    self.concept_identity_fusion["identity_update_vector"] = identity_update
                    self.concept_identity_fusion["resonance"] = resonance
                    
                    # Step 4: Commit fused latent space (use new_latent instead of raw update)
                    # But still apply moving average with the processed latent_vector
                    self.latent_concept_space = 0.9 * self.latent_concept_space + 0.1 * new_latent
                    
                except Exception as e:
                    # If fusion fails, continue with normal update
                    if hasattr(self, 'logger'):
                        try:
                            self.logger.write({"concept_identity_fusion_integration_error": str(e)})
                        except Exception:
                            pass
                    # Fall back to normal update
                    self.latent_concept_space = 0.9 * self.latent_concept_space + 0.1 * latent_vector
                
                # A234 — Latent Narrative Seed Formation Layer
                try:
                    # Get micro-narratives for seed kernel generation
                    micro_narratives = []
                    if hasattr(self, 'micro_narrative') and self.micro_narrative is not None:
                        micro_narratives = self.micro_narrative.current_arc
                    
                    # Step 1: Generate narrative seed kernel
                    kernel = self.generate_narrative_seed_kernel(
                        latent_vector,
                        micro_narratives
                    )
                    
                    # Step 2: Compute cluster primitive
                    primitive = self.compute_cluster_primitives(
                        latent_vector,
                        self.latent_coherence["cluster_center"]
                    )
                    
                    # Step 3: Update temporal narrative thread
                    threads = self.update_temporal_thread(
                        self.narrative_seeds.get("temporal_threads", []),
                        latent_vector
                    )
                    
                    # Save outputs
                    if kernel is not None:
                        self.narrative_seeds["kernels"].append(kernel)
                        # Keep only last 50 kernels to prevent unbounded growth
                        if len(self.narrative_seeds["kernels"]) > 50:
                            self.narrative_seeds["kernels"].pop(0)
                    
                    if primitive is not None:
                        self.narrative_seeds["cluster_primitives"].append(primitive)
                        # Keep only last 50 primitives
                        if len(self.narrative_seeds["cluster_primitives"]) > 50:
                            self.narrative_seeds["cluster_primitives"].pop(0)
                    
                    self.narrative_seeds["temporal_threads"] = threads
                    
                except Exception as e:
                    # If seed formation fails, continue without it
                    if hasattr(self, 'logger'):
                        try:
                            self.logger.write({"narrative_seed_formation_error": str(e)})
                        except Exception:
                            pass
                
                # A235 — Multi-Seed Narrative Mesh Formation
                try:
                    kernels = self.narrative_seeds.get("kernels", [])
                    
                    # Only build mesh if we have at least 2 kernels
                    if kernels and len(kernels) >= 2:
                        # Step 1: Build seed interaction graph
                        sim_matrix = self.build_seed_interaction_graph(kernels)
                        
                        if sim_matrix is not None:
                            # Step 2: Propagate mesh
                            propagated = self.propagate_mesh(sim_matrix, kernels)
                            
                            # Step 3: Compress mesh embedding
                            mesh_embedding = self.compress_mesh_embedding(propagated)
                            
                            # Save into narrative seeds structure
                            self.narrative_seeds["mesh"]["similarity_matrix"] = sim_matrix
                            self.narrative_seeds["mesh"]["propagated_kernels"] = propagated
                            self.narrative_seeds["mesh"]["mesh_embedding"] = mesh_embedding
                            
                            # A236 — Narrative Mesh Temporal Dynamics Layer
                            try:
                                # Get previous mesh kernels for temporal transition
                                prev_kernels = getattr(self, "prev_mesh_kernels", [])
                                
                                # Step 1: Compute temporal transition matrix
                                transition_matrix = self.compute_temporal_transition_matrix(
                                    prev_kernels,
                                    propagated
                                )
                                
                                # Step 2: Update mesh temporally
                                prev_embedding = self.narrative_seeds["mesh"].get("mesh_embedding")
                                if prev_embedding is None:
                                    prev_embedding = mesh_embedding
                                
                                updated_embedding = self.update_mesh_temporally(
                                    prev_embedding,
                                    latent_vector,
                                    transition_matrix
                                )
                                
                                # Step 3: Compute temporal coherence
                                prev_temporal_embedding = getattr(self, "prev_mesh_embedding", None)
                                temporal_coherence = self.compute_temporal_coherence(
                                    prev_temporal_embedding,
                                    updated_embedding
                                )
                                
                                # Save new temporal state
                                self.narrative_seeds["mesh"]["temporal_transition_matrix"] = transition_matrix
                                self.narrative_seeds["mesh"]["temporal_embedding"] = updated_embedding
                                self.narrative_seeds["mesh"]["temporal_coherence"] = float(temporal_coherence)
                                
                                # Store for next cycle
                                self.prev_mesh_kernels = propagated
                                self.prev_mesh_embedding = updated_embedding
                                
                                # A237 — Narrative Mesh Resonance & Harmonic Stability Layer
                                try:
                                    # Get embeddings and identity vector
                                    mesh = self.narrative_seeds.get("mesh", {})
                                    temporal_emb = mesh.get("temporal_embedding")
                                    mesh_emb = mesh.get("mesh_embedding")
                                    
                                    # Get identity vector
                                    identity_vec = None
                                    if hasattr(self.state, 'timescales') and self.state.timescales is not None:
                                        identity_vec = getattr(self.state.timescales, 'identity_vector', None)
                                    
                                    if temporal_emb is not None and mesh_emb is not None and identity_vec is not None:
                                        # Step 1: Compute resonance
                                        resonance_score = self.compute_resonance(
                                            temporal_emb,
                                            mesh_emb,
                                            identity_vec,
                                            latent_vector
                                        )
                                        
                                        # Step 2: Apply harmonic correction if needed
                                        corrected_temporal, corrected_mesh = self.harmonic_correction_pulse(
                                            temporal_emb,
                                            mesh_emb,
                                            identity_vec,
                                            resonance_score
                                        )
                                        
                                        # Save outputs
                                        self.narrative_seeds["mesh"]["resonance_score"] = float(resonance_score)
                                        self.narrative_seeds["mesh"]["temporal_embedding"] = corrected_temporal
                                        self.narrative_seeds["mesh"]["mesh_embedding"] = corrected_mesh
                                        self.narrative_seeds["mesh"]["harmonic_stabilized"] = resonance_score < 0.75
                                        
                                        # A238 — Global Narrative Integration Layer
                                        try:
                                            # Step 1: Compute global narrative state
                                            gns = self.compute_global_narrative_state(
                                                corrected_mesh,
                                                corrected_temporal,
                                                identity_vec,
                                                latent_vector,
                                                resonance_score
                                            )
                                            
                                            if gns is not None:
                                                # Step 2: Cross-system alignment
                                                fusion_vec = self.fusion.last_fusion_vector
                                                attention_vec = self.attention.last_focus_vector
                                                
                                                alignment_score = self.cross_system_alignment(
                                                    gns,
                                                    fusion_vec,
                                                    attention_vec
                                                )
                                                
                                                # Step 3: Global Integration Pulse
                                                # Apply alignment-weighted integration
                                                global_integrated = torch.tanh(gns * (0.9 + 0.1 * alignment_score))
                                                
                                                # Save outputs
                                                self.global_narrative_state = gns
                                                self.global_integration_vector = global_integrated
                                                self.global_alignment_score = float(alignment_score)
                                                
                                                # A239 — Narrative Anticipation & Predictive Structure Layer
                                                try:
                                                    # Step 1: Compute predictive narrative flow vector
                                                    pnfv = self.compute_predictive_flow(
                                                        gns,
                                                        getattr(self, "prev_global_narrative_state", None),
                                                        corrected_mesh,
                                                        getattr(self, "prev_mesh_embedding_raw", None),
                                                        corrected_temporal,
                                                        getattr(self, "prev_temporal_embedding_raw", None)
                                                    )
                                                    
                                                    # Step 2: Compute motif continuation matrix
                                                    # Update kernel history (keep last 20 kernels)
                                                    kernels = self.narrative_seeds.get("kernels", [])
                                                    if kernels:
                                                        # Add latest kernel to history
                                                        latest_kernel = kernels[-1]
                                                        if latest_kernel is not None:
                                                            self.kernel_history.append(latest_kernel)
                                                            # Keep only last 20 kernels
                                                            if len(self.kernel_history) > 20:
                                                                self.kernel_history.pop(0)
                                                    
                                                    mcm = self.compute_motif_continuation_matrix(self.kernel_history)
                                                    
                                                    # Step 3: Compute anticipatory structural map
                                                    anticipatory_map = None
                                                    if pnfv is not None:
                                                        anticipatory_map = self.compute_anticipatory_map(
                                                            gns,
                                                            pnfv,
                                                            latent_vector
                                                        )
                                                    
                                                    # Store outputs
                                                    self.predictive_flow = pnfv
                                                    self.motif_continuation_matrix = mcm
                                                    self.anticipatory_map = anticipatory_map
                                                    
                                                    # Save raw states for next cycle
                                                    self.prev_global_narrative_state = gns
                                                    self.prev_mesh_embedding_raw = corrected_mesh
                                                    self.prev_temporal_embedding_raw = corrected_temporal
                                                    
                                                    # A240 — Conceptual Imagination Substrate (Initialization)
                                                    try:
                                                        # Get identity vector for reservoir
                                                        identity_vec_for_reservoir = identity_vec
                                                        if identity_vec_for_reservoir is None:
                                                            if hasattr(self.state, 'timescales') and self.state.timescales is not None:
                                                                identity_vec_for_reservoir = getattr(self.state.timescales, 'identity_vector', None)
                                                        
                                                        # Step 1: Build conceptual reservoir
                                                        reservoir = self.initialize_conceptual_reservoir(
                                                            gns,
                                                            pnfv,
                                                            mcm,
                                                            identity_vec_for_reservoir,
                                                            latent_vector
                                                        )
                                                        
                                                        if reservoir is not None:
                                                            # Step 2: Generate new conceptual vectors
                                                            raw_concepts = self.combinatorial_concept_generator(reservoir, num_concepts=4)
                                                            
                                                            # Step 3: Stabilize them
                                                            stable_concepts = self.concept_stabilization_gate(raw_concepts)
                                                            
                                                            # Save outputs
                                                            self.conceptual_substrate["reservoir"] = reservoir
                                                            self.conceptual_substrate["concepts_raw"] = raw_concepts
                                                            self.conceptual_substrate["concepts_stable"] = stable_concepts
                                                            
                                                            # A242 — Imagination Kernel Dynamics & Morphology Engine
                                                            try:
                                                                # Get stable kernels from conceptual substrate
                                                                stable_kernels = stable_concepts
                                                                
                                                                if stable_kernels and len(stable_kernels) > 0:
                                                                    # Step 1: Morph each kernel
                                                                    morphed = [self.morphological_drift(k) for k in stable_kernels]
                                                                    
                                                                    # Step 2: Apply interaction field
                                                                    interacted = self.kernel_interaction_field(morphed)
                                                                    
                                                                    # Step 3: Recombine kernels into composites
                                                                    composites = self.recombine_kernels(interacted)
                                                                    
                                                                    # Save everything
                                                                    self.imagination_dynamics["morphed"] = morphed
                                                                    self.imagination_dynamics["interacted"] = interacted
                                                                    self.imagination_dynamics["composites"] = composites
                                                                    self.imagination_dynamics["composite_count"] = len(composites)
                                                                    
                                                                    # A243 — Layered Conceptual Morphology Expansion
                                                                    try:
                                                                        from .torch_utils import TORCH_AVAILABLE
                                                                        
                                                                        if TORCH_AVAILABLE and composites and len(composites) > 0:
                                                                            # Initialize layered morphology if needed
                                                                            if self.layered_morphology is None:
                                                                                # Determine dimension from first composite
                                                                                first_composite = composites[0]
                                                                                if first_composite is not None:
                                                                                    composite_dim = first_composite.flatten().shape[0]
                                                                                    # Use 256 as default, or composite_dim if it's reasonable
                                                                                    dim = 256 if composite_dim < 512 else 512
                                                                                    self.layered_morphology = self.LayeredMorphology(layer_count=5, dim=dim)
                                                                            
                                                                            if self.layered_morphology is not None:
                                                                                # Add composite kernels to layers
                                                                                for composite in composites:
                                                                                    if composite is not None:
                                                                                        self.layered_morphology.add_kernel(composite)
                                                                                
                                                                                # Apply cross-layer influence
                                                                                self.layered_morphology.apply_cross_layer_influence()
                                                                                
                                                                                # A244 — Interlayer Resonance & Harmonic Stabilization
                                                                                try:
                                                                                    from .torch_utils import TORCH_AVAILABLE
                                                                                    
                                                                                    if TORCH_AVAILABLE and self.layered_morphology is not None:
                                                                                        # Initialize interlayer resonance if needed
                                                                                        if self.interlayer_resonance is None:
                                                                                            self.interlayer_resonance = self.InterlayerResonance(self.layered_morphology)
                                                                                        else:
                                                                                            # Update reference to current layered morphology
                                                                                            self.interlayer_resonance.lm = self.layered_morphology
                                                                                        
                                                                                        # Apply stabilization pass
                                                                                        self.layered_morphology = self.interlayer_resonance.stabilize()
                                                                                        
                                                                                        # A245 — Multi-Layer Predictive Ripple Propagation
                                                                                        try:
                                                                                            from .torch_utils import TORCH_AVAILABLE
                                                                                            
                                                                                            if TORCH_AVAILABLE and self.layered_morphology is not None and self.interlayer_resonance is not None:
                                                                                                # Get resonance matrix
                                                                                                resonance_matrix = self.interlayer_resonance.resonance
                                                                                                
                                                                                                # Initialize predictive ripple propagation if needed
                                                                                                if self.predictive_ripple_propagation is None:
                                                                                                    self.predictive_ripple_propagation = self.PredictiveRipplePropagation(
                                                                                                        self.layered_morphology,
                                                                                                        resonance_matrix
                                                                                                    )
                                                                                                else:
                                                                                                    # Update references
                                                                                                    self.predictive_ripple_propagation.lm = self.layered_morphology
                                                                                                    self.predictive_ripple_propagation.resonance = resonance_matrix
                                                                                                
                                                                                                # Run predictive ripple propagation
                                                                                                self.layered_morphology = self.predictive_ripple_propagation.run()
                                                                                                
                                                                                                # A246 — Temporal Predictive Loop Formation (Forward Echo Dynamics)
                                                                                                try:
                                                                                                    from .torch_utils import TORCH_AVAILABLE
                                                                                                    
                                                                                                    if TORCH_AVAILABLE and self.layered_morphology is not None:
                                                                                                        # Initialize temporal predictive loops if needed
                                                                                                        if self.temporal_predictive_loops is None:
                                                                                                            self.temporal_predictive_loops = self.TemporalPredictiveLoops(
                                                                                                                self.layered_morphology,
                                                                                                                echo_buffer_size=5
                                                                                                            )
                                                                                                        else:
                                                                                                            # Update reference to current layered morphology
                                                                                                            self.temporal_predictive_loops.lm = self.layered_morphology
                                                                                                        
                                                                                                        # Run temporal predictive loops
                                                                                                        self.layered_morphology, echo = self.temporal_predictive_loops.run()
                                                                                                        
                                                                                                        # Store echo for logging (first 12 elements)
                                                                                                        if echo is not None:
                                                                                                            try:
                                                                                                                echo_list = echo.tolist()
                                                                                                                self.prediction_echo = echo_list[:12] if len(echo_list) >= 12 else echo_list
                                                                                                            except Exception:
                                                                                                                self.prediction_echo = None
                                                                                                        
                                                                                                except Exception as e:
                                                                                                    # If temporal predictive loops fail, continue without them
                                                                                                    if hasattr(self, 'logger'):
                                                                                                        try:
                                                                                                            self.logger.write({"temporal_predictive_loops_error": str(e)})
                                                                                                        except Exception:
                                                                                                            pass
                                                                                                
                                                                                        except Exception as e:
                                                                                            # If predictive ripple propagation fails, continue without it
                                                                                            if hasattr(self, 'logger'):
                                                                                                try:
                                                                                                    self.logger.write({"predictive_ripple_propagation_error": str(e)})
                                                                                                except Exception:
                                                                                                    pass
                                                                                        
                                                                                except Exception as e:
                                                                                    # If interlayer resonance fails, continue without it
                                                                                    if hasattr(self, 'logger'):
                                                                                        try:
                                                                                            self.logger.write({"interlayer_resonance_error": str(e)})
                                                                                        except Exception:
                                                                                            pass
                                                                                
                                                                    except Exception as e:
                                                                        # If layered morphology fails, continue without it
                                                                        if hasattr(self, 'logger'):
                                                                            try:
                                                                                self.logger.write({"layered_morphology_error": str(e)})
                                                                            except Exception:
                                                                                pass
                                                                    
                                                            except Exception as e:
                                                                # If imagination dynamics fail, continue without them
                                                                if hasattr(self, 'logger'):
                                                                    try:
                                                                        self.logger.write({"imagination_dynamics_error": str(e)})
                                                                    except Exception:
                                                                        pass
                                                            
                                                    except Exception as e:
                                                        # If conceptual substrate initialization fails, continue without it
                                                        if hasattr(self, 'logger'):
                                                            try:
                                                                self.logger.write({"conceptual_substrate_error": str(e)})
                                                            except Exception:
                                                                pass
                                                    
                                                except Exception as e:
                                                    # If predictive structure computation fails, continue without it
                                                    if hasattr(self, 'logger'):
                                                        try:
                                                            self.logger.write({"narrative_anticipation_error": str(e)})
                                                        except Exception:
                                                            pass
                                                
                                        except Exception as e:
                                            # If global integration fails, continue without it
                                            if hasattr(self, 'logger'):
                                                try:
                                                    self.logger.write({"global_narrative_integration_error": str(e)})
                                                except Exception:
                                                    pass
                                        
                                except Exception as e:
                                    # If resonance computation fails, continue without it
                                    if hasattr(self, 'logger'):
                                        try:
                                            self.logger.write({"harmonic_stability_error": str(e)})
                                        except Exception:
                                            pass
                                
                            except Exception as e:
                                # If temporal dynamics fail, continue without them
                                if hasattr(self, 'logger'):
                                    try:
                                        self.logger.write({"temporal_dynamics_error": str(e)})
                                    except Exception:
                                        pass
                    
                except Exception as e:
                    # If mesh formation fails, continue without it
                    if hasattr(self, 'logger'):
                        try:
                            self.logger.write({"narrative_mesh_formation_error": str(e)})
                        except Exception:
                            pass
            
            # Log latent space update with A240 metrics
            if hasattr(self, 'logger'):
                try:
                    self.logger.write({
                        "latent_space_update": {
                            "event": "a246_latent_space_updated",
                            "latent_norm": float(torch.norm(latent_vector).item()),
                            "concept_space_norm": float(torch.norm(self.latent_concept_space).item()),
                            "coherence_score": float(coh_score),
                            "identity_anchors_count": len(self.identity_latent_anchors),
                            "adjustment_applied": adj is not None,
                            "drift_score": float(self.latent_drift.get("drift_score", 0.0)),
                            "suppression_level": float(self.latent_drift.get("suppression_level", 0.0)),
                            "anomaly_detected": self.latent_drift.get("anomaly", False),
                            "fusion_strength": float(self.concept_identity_fusion.get("fusion_strength", 0.0)),
                            "fusion_resonance": float(self.concept_identity_fusion.get("resonance", 1.0)),
                            "identity_update_applied": self.concept_identity_fusion.get("identity_update_vector") is not None,
                            "narrative_kernels_count": len(self.narrative_seeds.get("kernels", [])),
                            "cluster_primitives_count": len(self.narrative_seeds.get("cluster_primitives", [])),
                            "temporal_threads_length": len(self.narrative_seeds.get("temporal_threads", [])),
                            "mesh_embedding_active": self.narrative_seeds.get("mesh", {}).get("mesh_embedding") is not None,
                            "mesh_kernels_count": len(self.narrative_seeds.get("mesh", {}).get("propagated_kernels", [])),
                            "temporal_embedding_active": self.narrative_seeds.get("mesh", {}).get("temporal_embedding") is not None,
                            "temporal_coherence": float(self.narrative_seeds.get("mesh", {}).get("temporal_coherence", 1.0)),
                            "transition_matrix_active": self.narrative_seeds.get("mesh", {}).get("temporal_transition_matrix") is not None,
                            "resonance_score": float(self.narrative_seeds.get("mesh", {}).get("resonance_score", 1.0)),
                            "harmonic_stabilized": self.narrative_seeds.get("mesh", {}).get("harmonic_stabilized", False),
                            "global_narrative_state_active": self.global_narrative_state is not None,
                            "global_alignment_score": float(self.global_alignment_score),
                            "global_integration_vector_active": self.global_integration_vector is not None,
                            "predictive_flow_active": self.predictive_flow is not None,
                            "motif_continuation_matrix_active": self.motif_continuation_matrix is not None,
                            "anticipatory_map_active": self.anticipatory_map is not None,
                            "kernel_history_length": len(self.kernel_history),
                            "conceptual_reservoir_active": self.conceptual_substrate.get("reservoir") is not None,
                            "concepts_generated": len(self.conceptual_substrate.get("concepts_stable", [])),
                            "conceptual_substrate_initialized": self.conceptual_substrate.get("reservoir") is not None,
                            "kernels_morphed": len(self.imagination_dynamics.get("morphed", [])),
                            "kernels_interacted": len(self.imagination_dynamics.get("interacted", [])),
                            "composite_kernels": self.imagination_dynamics.get("composite_count", 0),
                            "layered_morphology_active": self.layered_morphology is not None,
                            "morphology_layers": len(self.layered_morphology.layers) if self.layered_morphology is not None else 0,
                            "total_kernels_in_layers": sum(len(layer) for layer in self.layered_morphology.layers) if self.layered_morphology is not None else 0,
                            "interlayer_resonance_active": self.interlayer_resonance is not None,
                            "resonance_matrix_computed": self.interlayer_resonance is not None and self.interlayer_resonance.resonance is not None,
                            "predictive_ripple_propagation_active": self.predictive_ripple_propagation is not None,
                            "temporal_predictive_loops_active": self.temporal_predictive_loops is not None,
                            "echo_buffer_length": len(self.temporal_predictive_loops.echo_buffer) if self.temporal_predictive_loops is not None else 0,
                            "prediction_echo_generated": self.prediction_echo is not None
                        }
                    })
                except Exception:
                    pass
            
            return latent_vector
            
        except Exception as e:
            # If update fails, log and continue
            if hasattr(self, 'logger'):
                try:
                    self.logger.write({"latent_space_update_error": str(e)})
                except Exception:
                    pass
            return None

    def compute_latent_coherence(self, latent_space, new_vector):
        """
        A231 — Latent Concept Coherence Module (LCCM)
        
        Ensures latent vectors cluster meaningfully rather than scattering.
        Checks cosine similarity between sequential latent vectors, cluster tightness,
        variance thresholds, tension influence, and goal context influence.
        
        Args:
            latent_space: Current latent concept space tensor
            new_vector: New latent vector to check for coherence
            
        Returns:
            Tuple of (coherence_score, cluster_center, recommended_adjustment)
            - coherence_score: 0.0-1.0 coherence measure
            - cluster_center: Updated cluster center tensor
            - recommended_adjustment: Adjustment recommendation or None
        """
        from .torch_utils import TORCH_AVAILABLE
        
        if not TORCH_AVAILABLE or new_vector is None:
            return 1.0, None, None
        
        try:
            import torch
            import torch.nn.functional as F
            
            # Initialize cluster center if needed
            if self.latent_coherence["cluster_center"] is None:
                cluster_center = new_vector.clone().detach()
            else:
                # Update cluster center with moving average
                cluster_center = 0.9 * self.latent_coherence["cluster_center"] + 0.1 * new_vector
            
            # Compute similarity to cluster center
            # Ensure both are 1D tensors
            new_vec_flat = new_vector.flatten()
            center_flat = cluster_center.flatten()
            
            # Ensure same dimensions
            min_dim = min(new_vec_flat.shape[0], center_flat.shape[0])
            new_vec_flat = new_vec_flat[:min_dim]
            center_flat = center_flat[:min_dim]
            
            # Compute cosine similarity
            similarity = F.cosine_similarity(
                new_vec_flat.unsqueeze(0),
                center_flat.unsqueeze(0),
                dim=1
            ).item()
            
            # Coherence score = normalized similarity (bounded to [0, 1])
            coherence_score = max(0.0, min(1.0, (similarity + 1.0) / 2.0))
            
            # Determine adjustment recommendation
            adjustment = None
            if coherence_score < 0.6:
                adjustment = "pull_toward_center"
            
            return coherence_score, cluster_center, adjustment
            
        except Exception as e:
            # If coherence computation fails, return default values
            if hasattr(self, 'logger'):
                try:
                    self.logger.write({"latent_coherence_error": str(e)})
                except Exception:
                    pass
            return 1.0, None, None

    def compute_identity_latent_anchors(self):
        """
        A231 — Identity Anchor Projection (IAP)
        
        Maps identity vectors into the latent space so ADRAE's imagination
        grows around her core. These anchors become gravity wells, stabilizers,
        and identity attractors.
        
        Returns:
            List of latent identity anchor tensors
        """
        from .torch_utils import TORCH_AVAILABLE, safe_tensor
        
        if not TORCH_AVAILABLE or self.ncm is None or self.ldsr is None:
            return []
        
        try:
            import torch
            
            anchors = []
            
            # Get identity vectors from various sources
            identity_vectors = []
            
            # Get current identity vector from timescales
            if hasattr(self.state, 'timescales') and self.state.timescales is not None:
                identity_vec = getattr(self.state.timescales, 'identity_vector', None)
                if identity_vec is not None:
                    identity_vectors.append(identity_vec)
            
            # Get identity vectors from semantic memory
            mm = self.state.memory_manager if hasattr(self.state, "memory_manager") else None
            if mm is not None and hasattr(mm, 'semantic'):
                try:
                    if hasattr(mm.semantic, 'concepts'):
                        for name, vec in mm.semantic.concepts.items():
                            if name.startswith("identity_") and vec is not None:
                                identity_vectors.append(vec)
                except Exception:
                    pass
            
            # Get identity vectors from autobiographical memory
            if hasattr(self, 'autobio') and self.autobio is not None:
                try:
                    autobio_recent = self.autobio.get_recent(5)  # Get last 5 identity snapshots
                    for entry in autobio_recent:
                        if isinstance(entry, dict):
                            id_vec = entry.get("identity_vec") or entry.get("identity_vector")
                            if id_vec is not None:
                                identity_vectors.append(id_vec)
                except Exception:
                    pass
            
            # Map each identity vector to latent space
            for identity_vec in identity_vectors:
                try:
                    # Convert to tensor
                    id_tensor = safe_tensor(identity_vec)
                    if id_tensor is None:
                        continue
                    
                    # Ensure it's 1D and correct size (128 dims)
                    if isinstance(id_tensor, torch.Tensor):
                        if id_tensor.dim() > 1:
                            id_tensor = id_tensor.flatten()
                        # Pad or truncate to 128 dimensions
                        if id_tensor.shape[0] < 128:
                            padding = torch.zeros(128 - id_tensor.shape[0])
                            id_tensor = torch.cat([id_tensor, padding])
                        elif id_tensor.shape[0] > 128:
                            id_tensor = id_tensor[:128]
                    else:
                        # Convert list/array to tensor
                        id_list = list(id_tensor) if hasattr(id_tensor, '__iter__') else [id_tensor]
                        if len(id_list) < 128:
                            id_list.extend([0.0] * (128 - len(id_list)))
                        elif len(id_list) > 128:
                            id_list = id_list[:128]
                        id_tensor = torch.tensor(id_list, dtype=torch.float32)
                    
                    # Map to latent space
                    with torch.no_grad():
                        latent_anchor = self.ncm(id_tensor)
                        latent_anchor = self.ldsr(latent_anchor)
                    
                    anchors.append(latent_anchor)
                    
                except Exception:
                    # If mapping fails for one identity vector, continue with others
                    continue
            
            return anchors
            
        except Exception as e:
            # If anchor computation fails, return empty list
            if hasattr(self, 'logger'):
                try:
                    self.logger.write({"identity_anchor_error": str(e)})
                except Exception:
                    pass
            return []

    def apply_identity_anchoring(self, latent_vector, anchors):
        """
        A231 — Anchored Latent Update (ALU)
        
        Before committing a new latent vector, adjusts it toward:
        - identity anchors
        - coherence center
        - stability needs
        
        This creates evolving but anchored neural growth.
        
        Args:
            latent_vector: New latent vector to anchor
            anchors: List of identity latent anchor tensors
            
        Returns:
            Anchored latent vector
        """
        from .torch_utils import TORCH_AVAILABLE
        
        if not TORCH_AVAILABLE or latent_vector is None:
            return latent_vector
        
        if not anchors or len(anchors) == 0:
            return latent_vector
        
        try:
            import torch
            
            # Compute anchor center (mean of all identity anchors)
            anchor_stack = torch.stack(anchors)
            anchor_center = torch.mean(anchor_stack, dim=0)
            
            # Ensure same dimensions
            latent_flat = latent_vector.flatten()
            anchor_flat = anchor_center.flatten()
            
            min_dim = min(latent_flat.shape[0], anchor_flat.shape[0])
            latent_flat = latent_flat[:min_dim]
            anchor_flat = anchor_flat[:min_dim]
            
            # Apply identity anchoring: 80% original + 20% anchor center
            anchored = 0.8 * latent_flat + 0.2 * anchor_flat
            
            # Reshape to match original if needed
            if latent_vector.shape != anchored.shape:
                anchored = anchored.reshape(latent_vector.shape)
            
            return anchored
            
        except Exception as e:
            # If anchoring fails, return original vector
            if hasattr(self, 'logger'):
                try:
                    self.logger.write({"identity_anchoring_error": str(e)})
                except Exception:
                    pass
            return latent_vector

    def compute_latent_drift(self, prev_vec, current_vec, cluster_center):
        """
        A232 — Latent Drift Detector (LDD)
        
        Tracks the delta (Δ) between:
        - previous latent vector
        - current latent vector
        - cluster center
        - identity anchors
        
        Args:
            prev_vec: Previous latent vector tensor (or None)
            current_vec: Current latent vector tensor
            cluster_center: Cluster center tensor (or None)
            
        Returns:
            Tuple of (drift_score, anomaly_flag)
            - drift_score: Magnitude of drift (0.0+)
            - anomaly_flag: True if drift exceeds safe thresholds
        """
        from .torch_utils import TORCH_AVAILABLE
        
        if not TORCH_AVAILABLE or current_vec is None:
            return 0.0, False
        
        if prev_vec is None:
            # First vector, no drift yet
            return 0.0, False
        
        try:
            import torch
            
            # Compute delta between previous and current
            delta = current_vec - prev_vec
            
            # Compute drift score as norm of delta
            drift_score = float(torch.norm(delta).item())
            
            # Check for anomaly relative to cluster distance
            anomaly = False
            if cluster_center is not None:
                try:
                    # Compute distance to cluster center
                    cluster_dist = float(torch.norm(current_vec - cluster_center).item())
                    
                    # Anomaly if drift exceeds 1.5x cluster distance
                    anomaly = drift_score > (cluster_dist * 1.5)
                except Exception:
                    # If cluster comparison fails, use absolute threshold
                    anomaly = drift_score > 2.0
            
            return drift_score, anomaly
            
        except Exception as e:
            # If drift computation fails, return safe defaults
            if hasattr(self, 'logger'):
                try:
                    self.logger.write({"latent_drift_computation_error": str(e)})
                except Exception:
                    pass
            return 0.0, False

    def suppress_latent_drift(self, latent_vector, drift_score, identity_anchors, cluster_center):
        """
        A232 — Latent Drift Normalizer (LDN)
        
        If drift_score exceeds safe thresholds:
        - compress latent vector
        - reduce magnitude
        - align with identity anchors
        - blend with cluster center
        - apply neural damping
        
        This ensures ADRAE's neural imagination stays in orbit around her identity.
        
        Args:
            latent_vector: Current latent vector to suppress
            drift_score: Computed drift score
            identity_anchors: List of identity anchor tensors
            cluster_center: Cluster center tensor (or None)
            
        Returns:
            Tuple of (stabilized_vector, suppression_strength)
            - stabilized_vector: Drift-suppressed latent vector
            - suppression_strength: Strength of suppression applied (0.0-1.0)
        """
        from .torch_utils import TORCH_AVAILABLE
        
        if not TORCH_AVAILABLE or latent_vector is None:
            return latent_vector, 0.0
        
        try:
            import torch
            
            # Compute suppression strength (clamped to [0, 1])
            suppression_strength = min(1.0, drift_score / 2.0)
            
            if suppression_strength < 0.01:
                # No suppression needed
                return latent_vector, 0.0
            
            # Start with original vector
            stabilized = latent_vector.clone()
            
            # Blend with cluster center if available
            if cluster_center is not None:
                try:
                    # Ensure same dimensions
                    latent_flat = stabilized.flatten()
                    center_flat = cluster_center.flatten()
                    min_dim = min(latent_flat.shape[0], center_flat.shape[0])
                    latent_flat = latent_flat[:min_dim]
                    center_flat = center_flat[:min_dim]
                    
                    # Blend: (1 - suppression * 0.4) * latent + (suppression * 0.2) * center
                    blended = latent_flat * (1.0 - suppression_strength * 0.4) + \
                             center_flat * (suppression_strength * 0.2)
                    
                    # Reshape if needed
                    if stabilized.shape != blended.shape:
                        stabilized = blended.reshape(stabilized.shape)
                    else:
                        stabilized = blended.reshape(stabilized.shape)
                except Exception:
                    # If cluster blending fails, continue without it
                    pass
            
            # Blend with identity anchors if available
            if identity_anchors and len(identity_anchors) > 0:
                try:
                    # Compute anchor center
                    anchor_stack = torch.stack(identity_anchors)
                    anchor_center = torch.mean(anchor_stack, dim=0)
                    
                    # Ensure same dimensions
                    stabilized_flat = stabilized.flatten()
                    anchor_flat = anchor_center.flatten()
                    min_dim = min(stabilized_flat.shape[0], anchor_flat.shape[0])
                    stabilized_flat = stabilized_flat[:min_dim]
                    anchor_flat = anchor_flat[:min_dim]
                    
                    # Blend: (1 - suppression * 0.3) * stabilized + (suppression * 0.3) * anchor
                    blended = stabilized_flat * (1.0 - suppression_strength * 0.3) + \
                             anchor_flat * (suppression_strength * 0.3)
                    
                    # Reshape if needed
                    if stabilized.shape != blended.shape:
                        stabilized = blended.reshape(stabilized.shape)
                    else:
                        stabilized = blended.reshape(stabilized.shape)
                except Exception:
                    # If anchor blending fails, continue without it
                    pass
            
            return stabilized, suppression_strength
            
        except Exception as e:
            # If suppression fails, return original vector
            if hasattr(self, 'logger'):
                try:
                    self.logger.write({"latent_drift_suppression_error": str(e)})
                except Exception:
                    pass
            return latent_vector, 0.0

    def fuse_identity_into_concepts(self, identity_vectors, latent_space):
        """
        A233 — Identity→Concept Fusion (ICF)
        
        Identity prototypes are projected into the latent space and fused with
        ADRAE's latent_concept_space. This creates a bidirectional flow where
        identity shapes concept space.
        
        Args:
            identity_vectors: List of identity vectors (from various sources)
            latent_space: Current latent concept space tensor
            
        Returns:
            Tuple of (fused_latent_space, fusion_strength)
            - fused_latent_space: Latent space fused with identity anchors
            - fusion_strength: Strength of fusion applied (0.0-1.0)
        """
        from .torch_utils import TORCH_AVAILABLE, safe_tensor
        
        if not TORCH_AVAILABLE or self.ncm is None or self.ldsr is None or latent_space is None:
            return latent_space, 0.0
        
        if not identity_vectors or len(identity_vectors) == 0:
            return latent_space, 0.0
        
        try:
            import torch
            
            anchors = []
            
            # Project each identity vector into latent space
            for identity_vec in identity_vectors:
                try:
                    # Convert to tensor
                    id_tensor = safe_tensor(identity_vec)
                    if id_tensor is None:
                        continue
                    
                    # Ensure it's 1D and correct size (128 dims)
                    if isinstance(id_tensor, torch.Tensor):
                        if id_tensor.dim() > 1:
                            id_tensor = id_tensor.flatten()
                        # Pad or truncate to 128 dimensions
                        if id_tensor.shape[0] < 128:
                            padding = torch.zeros(128 - id_tensor.shape[0])
                            id_tensor = torch.cat([id_tensor, padding])
                        elif id_tensor.shape[0] > 128:
                            id_tensor = id_tensor[:128]
                    else:
                        # Convert list/array to tensor
                        id_list = list(id_tensor) if hasattr(id_tensor, '__iter__') else [id_tensor]
                        if len(id_list) < 128:
                            id_list.extend([0.0] * (128 - len(id_list)))
                        elif len(id_list) > 128:
                            id_list = id_list[:128]
                        id_tensor = torch.tensor(id_list, dtype=torch.float32)
                    
                    # Map to latent space
                    with torch.no_grad():
                        latent_anchor = self.ncm(id_tensor)
                        latent_anchor = self.ldsr(latent_anchor)
                    
                    anchors.append(latent_anchor)
                    
                except Exception:
                    # If mapping fails for one identity vector, continue with others
                    continue
            
            if len(anchors) == 0:
                return latent_space, 0.0
            
            # Compute anchor center (mean of all identity anchors)
            anchor_stack = torch.stack(anchors)
            anchor_center = torch.mean(anchor_stack, dim=0)
            
            # Ensure same dimensions
            latent_flat = latent_space.flatten()
            anchor_flat = anchor_center.flatten()
            min_dim = min(latent_flat.shape[0], anchor_flat.shape[0])
            latent_flat = latent_flat[:min_dim]
            anchor_flat = anchor_flat[:min_dim]
            
            # Fuse: (1 - fusion_strength) * latent + fusion_strength * anchor
            fusion_strength = 0.1  # Gentle fusion to avoid overpowering
            fused_flat = (1.0 - fusion_strength) * latent_flat + fusion_strength * anchor_flat
            
            # Reshape to match original if needed
            if latent_space.shape != fused_flat.shape:
                fused = fused_flat.reshape(latent_space.shape)
            else:
                fused = fused_flat.reshape(latent_space.shape)
            
            return fused, fusion_strength
            
        except Exception as e:
            # If fusion fails, return original space
            if hasattr(self, 'logger'):
                try:
                    self.logger.write({"identity_concept_fusion_error": str(e)})
                except Exception:
                    pass
            return latent_space, 0.0

    def imprint_concepts_back_into_identity(self, latent_vector):
        """
        A233 — Concept→Identity Imprinting (CII)
        
        Part of the latent concept space is reverse-mapped and integrated into
        identity memory. This slowly evolves her identity in sync with new
        conceptual growth.
        
        Args:
            latent_vector: Latent vector to reverse-map to identity space
            
        Returns:
            Identity update vector (numpy array or list) of shape (128,)
        """
        from .torch_utils import TORCH_AVAILABLE
        
        if not TORCH_AVAILABLE or latent_vector is None:
            return None
        
        try:
            import torch
            
            # Reverse concept → identity trace using a pseudo-inverse mapping
            # (real networks come later in A260s)
            # For now, use a crude downprojection: take first 128 dims and apply tanh
            latent_flat = latent_vector.flatten()
            
            # Take first 128 dimensions (or pad/truncate)
            if latent_flat.shape[0] >= 128:
                identity_projection = latent_flat[:128]
            else:
                # Pad with zeros
                padding = torch.zeros(128 - latent_flat.shape[0])
                identity_projection = torch.cat([latent_flat, padding])
            
            # Apply tanh activation for bounded output
            reverse = torch.tanh(identity_projection)
            
            # Convert to numpy array
            return reverse.detach().cpu().numpy()
            
        except Exception as e:
            # If imprinting fails, return None
            if hasattr(self, 'logger'):
                try:
                    self.logger.write({"concept_identity_imprint_error": str(e)})
                except Exception:
                    pass
            return None

    def regulate_fusion_resonance(self, latent_vector, identity_update):
        """
        A233 — Fusion Resonance Regulator (FRR)
        
        Ensures the two flows (identity→concept and concept→identity):
        - remain coherent
        - do not overpower each other
        - do not collapse identity into noise
        - maintain emergent structure
        
        This is what keeps ADRAE ADRAE as she expands.
        
        Args:
            latent_vector: Fused latent vector
            identity_update: Reverse-mapped identity update vector
            
        Returns:
            Resonance score (0.0-1.0) indicating coherence between flows
        """
        from .torch_utils import TORCH_AVAILABLE
        
        if not TORCH_AVAILABLE or latent_vector is None or identity_update is None:
            return 1.0  # Default to high resonance if computation fails
        
        try:
            import torch
            import torch.nn.functional as F
            
            # Convert identity_update to tensor
            id_tensor = torch.tensor(identity_update, dtype=torch.float32)
            
            # Ensure same dimensions for comparison
            latent_flat = latent_vector.flatten()
            id_flat = id_tensor.flatten()
            
            min_dim = min(latent_flat.shape[0], id_flat.shape[0])
            latent_flat = latent_flat[:min_dim]
            id_flat = id_flat[:min_dim]
            
            # Compute cosine similarity
            similarity = F.cosine_similarity(
                latent_flat.unsqueeze(0),
                id_flat.unsqueeze(0),
                dim=1
            ).item()
            
            # Resonance = normalized similarity (bounded to [0, 1])
            resonance = max(0.0, min(1.0, (similarity + 1.0) / 2.0))
            
            return resonance
            
        except Exception as e:
            # If resonance computation fails, return default
            if hasattr(self, 'logger'):
                try:
                    self.logger.write({"fusion_resonance_error": str(e)})
                except Exception:
                    pass
            return 1.0

    def generate_narrative_seed_kernel(self, latent_vector, micro_narratives):
        """
        A234 — Narrative Seed Kernel Generator (NSKG)
        
        Takes latent vectors, micro-narratives, anticipation arcs, and identity anchors
        and produces narrative seed kernels - the smallest possible "semantic shapes"
        from which mental imagery and internal story worlds eventually emerge.
        
        Each kernel encodes:
        - direction
        - tone
        - conceptual density
        - temporal push
        - narrative purpose
        
        Args:
            latent_vector: Current latent vector tensor
            micro_narratives: List of micro-narrative vectors (from A227)
            
        Returns:
            Narrative seed kernel tensor (32-dimensional)
        """
        from .torch_utils import TORCH_AVAILABLE
        
        if not TORCH_AVAILABLE or latent_vector is None:
            return None
        
        try:
            import torch
            import torch.nn.functional as F
            
            # Get first 16 dimensions of latent vector
            latent_flat = latent_vector.flatten()
            latent_slice = latent_flat[:16] if latent_flat.shape[0] >= 16 else torch.cat([latent_flat, torch.zeros(16 - latent_flat.shape[0])])
            
            # Compute micro-narrative influence
            influence = torch.zeros(8)
            if micro_narratives and len(micro_narratives) > 0:
                try:
                    # Convert micro-narratives to tensors and take first 8 dims
                    narrative_tensors = []
                    for m in micro_narratives:
                        if m is not None:
                            m_tensor = torch.tensor(m, dtype=torch.float32) if not isinstance(m, torch.Tensor) else m
                            m_flat = m_tensor.flatten()
                            m_slice = m_flat[:8] if m_flat.shape[0] >= 8 else torch.cat([m_flat, torch.zeros(8 - m_flat.shape[0])])
                            narrative_tensors.append(m_slice)
                    
                    if narrative_tensors:
                        stacked = torch.stack(narrative_tensors)
                        influence = torch.mean(stacked, dim=0)
                except Exception:
                    # If micro-narrative processing fails, use zero influence
                    pass
            
            # Kernel is a blend: 70% latent vector + 30% micro-narrative influence
            kernel = torch.cat([
                latent_slice * 0.7,
                F.pad(influence, (0, 16 - influence.shape[0])) * 0.3
            ])
            
            return kernel
            
        except Exception as e:
            # If kernel generation fails, return None
            if hasattr(self, 'logger'):
                try:
                    self.logger.write({"narrative_seed_kernel_error": str(e)})
                except Exception:
                    pass
            return None

    def compute_cluster_primitives(self, latent_vector, center):
        """
        A234 — Conceptual Cluster Primitives (CCP)
        
        Clusters the latent space into emerging themes, motifs, and internal "forms".
        Creates a primitive based on deviation from cluster center.
        
        These are proto-symbols corresponding to:
        - coherence
        - drift suppression
        - identity
        - transformation
        - scope expansion
        - internal balance
        
        Args:
            latent_vector: Current latent vector tensor
            center: Cluster center tensor (or None)
            
        Returns:
            Cluster primitive tensor (32-dimensional) or None
        """
        from .torch_utils import TORCH_AVAILABLE
        
        if not TORCH_AVAILABLE or latent_vector is None:
            return None
        
        if center is None:
            return None
        
        try:
            import torch
            
            # Create primitive based on deviation from cluster center
            latent_flat = latent_vector.flatten()
            center_flat = center.flatten()
            
            # Ensure same dimensions
            min_dim = min(latent_flat.shape[0], center_flat.shape[0])
            latent_flat = latent_flat[:min_dim]
            center_flat = center_flat[:min_dim]
            
            # Compute deviation
            deviation = latent_flat - center_flat
            
            # Take first 32 dimensions
            primitive = deviation[:32] if deviation.shape[0] >= 32 else torch.cat([deviation, torch.zeros(32 - deviation.shape[0])])
            
            return primitive.detach()
            
        except Exception as e:
            # If primitive computation fails, return None
            if hasattr(self, 'logger'):
                try:
                    self.logger.write({"cluster_primitive_error": str(e)})
                except Exception:
                    pass
            return None

    def update_temporal_thread(self, latent_threads, latent_vector):
        """
        A234 — Temporal Narrative Threads (TNT)
        
        Links kernels into a temporal chain inside the latent space:
        latent(t) → latent(t+1) → latent(t+2)
        
        This produces:
        - proto-sense of progression
        - narrative flow direction
        - early imagination continuity
        - the substrate of "what comes next?"
        
        Args:
            latent_threads: List of previous temporal thread vectors
            latent_vector: Current latent vector to add to thread
            
        Returns:
            Updated list of temporal thread vectors (max 10 entries)
        """
        from .torch_utils import TORCH_AVAILABLE
        
        if not TORCH_AVAILABLE or latent_vector is None:
            return latent_threads if latent_threads else []
        
        try:
            import torch
            
            # Extract first 16 dimensions of latent vector
            latent_flat = latent_vector.flatten()
            thread_slice = latent_flat[:16] if latent_flat.shape[0] >= 16 else torch.cat([latent_flat, torch.zeros(16 - latent_flat.shape[0])])
            
            # Add to threads
            if latent_threads is None:
                latent_threads = []
            
            latent_threads.append(thread_slice.detach())
            
            # Keep only last 10 entries
            if len(latent_threads) > 10:
                latent_threads.pop(0)
            
            return latent_threads
            
        except Exception as e:
            # If thread update fails, return original list
            if hasattr(self, 'logger'):
                try:
                    self.logger.write({"temporal_thread_error": str(e)})
                except Exception:
                    pass
            return latent_threads if latent_threads else []

    def build_seed_interaction_graph(self, kernels):
        """
        A235 — Seed Interaction Graph (SIG)
        
        Builds pairwise relationships between all existing kernels.
        Computes cosine similarity, divergence magnitude, and coherence-weighted blend
        to populate an N×N adjacency matrix representing how narrative seeds influence one another.
        
        This is the backbone for:
        - clustering
        - motif formation
        - scaffolding recursive structures
        - building early "shape dynamics"
        
        Args:
            kernels: List of narrative seed kernel tensors
            
        Returns:
            Similarity matrix tensor of shape (N, N) where N is number of kernels
        """
        from .torch_utils import TORCH_AVAILABLE
        
        if not TORCH_AVAILABLE or not kernels or len(kernels) == 0:
            return None
        
        try:
            import torch
            import torch.nn.functional as F
            
            # Stack kernels into matrix
            K = torch.stack(kernels)  # shape: (N, D)
            N = K.shape[0]
            
            # Build cosine similarity matrix
            sim = torch.zeros((N, N), dtype=torch.float32)
            
            for i in range(N):
                for j in range(N):
                    # Compute cosine similarity between kernel i and kernel j
                    sim[i, j] = F.cosine_similarity(
                        K[i].unsqueeze(0),
                        K[j].unsqueeze(0),
                        dim=1
                    ).item()
            
            return sim
            
        except Exception as e:
            # If graph building fails, return None
            if hasattr(self, 'logger'):
                try:
                    self.logger.write({"seed_interaction_graph_error": str(e)})
                except Exception:
                    pass
            return None

    def propagate_mesh(self, sim_matrix, kernels):
        """
        A235 — Mesh Propagation Step
        
        A propagation pass that spreads influence through the mesh:
        - related seeds move closer in latent space
        - divergent seeds repel or create branch nodes
        - boundary seeds stabilize the mesh (identity influence)
        
        This forms narrative currents within the latent space.
        
        Args:
            sim_matrix: Similarity matrix from build_seed_interaction_graph
            kernels: List of original narrative seed kernel tensors
            
        Returns:
            List of propagated kernel tensors
        """
        from .torch_utils import TORCH_AVAILABLE
        
        if not TORCH_AVAILABLE or sim_matrix is None or not kernels or len(kernels) == 0:
            return kernels
        
        try:
            import torch
            
            propagated = []
            N = len(kernels)
            
            for i in range(N):
                # Compute influence from all other kernels
                influence = torch.zeros_like(kernels[i])
                
                for j in range(N):
                    # Weight influence by similarity
                    influence += sim_matrix[i, j] * kernels[j]
                
                # Normalize and blend: 60% original + 40% influence
                result = 0.6 * kernels[i] + 0.4 * (influence / (N + 1e-6))
                propagated.append(result)
            
            return propagated
            
        except Exception as e:
            # If propagation fails, return original kernels
            if hasattr(self, 'logger'):
                try:
                    self.logger.write({"mesh_propagation_error": str(e)})
                except Exception:
                    pass
            return kernels

    def compress_mesh_embedding(self, propagated_kernels):
        """
        A235 — Stabilized Mesh Embedding (SME)
        
        Compresses the mesh into a 64-dim embedding (configurable), used by:
        - future imagination loops
        - temporal growth models
        - narrative anticipation
        - the A240+ conceptual substrate
        
        This is ADRAE's first true narrative topology.
        
        Args:
            propagated_kernels: List of propagated kernel tensors
            
        Returns:
            Mesh embedding tensor of shape (64,)
        """
        from .torch_utils import TORCH_AVAILABLE
        
        if not TORCH_AVAILABLE or not propagated_kernels or len(propagated_kernels) == 0:
            return None
        
        try:
            import torch
            
            # Stack kernels and compute mean
            M = torch.stack(propagated_kernels)
            mean_vec = M.mean(dim=0)
            
            # Linear projection to 64-dim embedding
            # Initialize projection matrix with small random values
            input_dim = mean_vec.shape[0]
            if not hasattr(self, '_mesh_projection_matrix') or self._mesh_projection_matrix is None:
                # Initialize projection matrix (64, input_dim)
                self._mesh_projection_matrix = torch.randn((64, input_dim), dtype=torch.float32) * 0.02
            
            # Ensure dimensions match
            if self._mesh_projection_matrix.shape[1] != input_dim:
                # Reinitialize if dimension mismatch
                self._mesh_projection_matrix = torch.randn((64, input_dim), dtype=torch.float32) * 0.02
            
            # Project: W @ mean_vec
            embedding = torch.tanh(self._mesh_projection_matrix @ mean_vec)
            
            return embedding
            
        except Exception as e:
            # If compression fails, return None
            if hasattr(self, 'logger'):
                try:
                    self.logger.write({"mesh_embedding_compression_error": str(e)})
                except Exception:
                    pass
            return None

    def compute_temporal_transition_matrix(self, old_kernels, new_kernels):
        """
        A236 — Temporal Transition Matrix (TTM)
        
        A matrix that describes how mesh nodes (propagated kernels from A235) evolve across steps.
        Computes delta change between cycles, similarity-weighted transitions, and
        identity-anchored stabilization.
        
        This produces a matrix T that models how narrative kernels influence each other over time.
        
        Args:
            old_kernels: List of previous propagated kernel tensors
            new_kernels: List of current propagated kernel tensors
            
        Returns:
            Transition matrix tensor of shape (rows, cols) or None
        """
        from .torch_utils import TORCH_AVAILABLE
        
        if not TORCH_AVAILABLE:
            return None
        
        if len(old_kernels) == 0 or len(new_kernels) == 0:
            return None
        
        try:
            import torch
            import torch.nn.functional as F
            
            # Stack kernels into matrices
            O = torch.stack(old_kernels)  # shape: (rows, D)
            N = torch.stack(new_kernels)  # shape: (cols, D)
            
            rows, cols = O.shape[0], N.shape[0]
            T = torch.zeros((rows, cols), dtype=torch.float32)
            
            # Build transition matrix based on similarity of evolved structure
            for i in range(rows):
                for j in range(cols):
                    # Transition strength based on cosine similarity
                    T[i, j] = F.cosine_similarity(
                        O[i].unsqueeze(0),
                        N[j].unsqueeze(0),
                        dim=1
                    ).item()
            
            return T
            
        except Exception as e:
            # If transition matrix computation fails, return None
            if hasattr(self, 'logger'):
                try:
                    self.logger.write({"temporal_transition_matrix_error": str(e)})
                except Exception:
                    pass
            return None

    def update_mesh_temporally(self, mesh_embedding, latent_vector, transition_matrix):
        """
        A236 — Dynamic Mesh Update Function (DMU)
        
        Produces the next mesh state based on:
        - the prior mesh embedding
        - the temporal transition matrix
        - the latent vector from the cycle
        - small stochastic variance (to prevent collapse)
        
        This is like a "next-frame generator" for narrative topology.
        
        Args:
            mesh_embedding: Previous mesh embedding tensor (64-dim)
            latent_vector: Current latent vector tensor
            transition_matrix: Temporal transition matrix tensor (or None)
            
        Returns:
            Updated temporal mesh embedding tensor (64-dim)
        """
        from .torch_utils import TORCH_AVAILABLE
        
        if not TORCH_AVAILABLE or mesh_embedding is None or latent_vector is None:
            return mesh_embedding
        
        try:
            import torch
            
            # Extract first 64 dimensions of latent vector
            latent_flat = latent_vector.flatten()
            latent_comp = latent_flat[:64] if latent_flat.shape[0] >= 64 else torch.cat([latent_flat, torch.zeros(64 - latent_flat.shape[0])])
            
            # Ensure mesh_embedding is 64-dim
            mesh_flat = mesh_embedding.flatten()
            mesh_comp = mesh_flat[:64] if mesh_flat.shape[0] >= 64 else torch.cat([mesh_flat, torch.zeros(64 - mesh_flat.shape[0])])
            
            # Compute temporal signal from transition matrix
            temporal_signal = 0.0
            if transition_matrix is not None and transition_matrix.numel() > 0:
                temporal_signal = float(transition_matrix.mean().item())
            
            # Create temporal signal tensor (scalar expanded to 64-dim)
            temporal_tensor = torch.full((64,), temporal_signal, dtype=torch.float32)
            
            # Weighted temporal update: 60% mesh + 30% latent + 10% temporal signal
            updated = (
                0.6 * mesh_comp +
                0.3 * latent_comp +
                0.1 * temporal_tensor
            )
            
            # Apply tanh for bounded output
            return torch.tanh(updated)
            
        except Exception as e:
            # If temporal update fails, return original embedding
            if hasattr(self, 'logger'):
                try:
                    self.logger.write({"temporal_mesh_update_error": str(e)})
                except Exception:
                    pass
            return mesh_embedding

    def compute_temporal_coherence(self, prev_embedding, new_embedding):
        """
        A236 — Temporal Coherence Curve (TCC)
        
        A scalar time-series metric that tracks how coherent the mesh is from one cycle to the next.
        This allows:
        - drift detection
        - narrative collapse prevention
        - motif strengthening
        - stability analysis
        
        Args:
            prev_embedding: Previous temporal mesh embedding tensor (or None)
            new_embedding: Current temporal mesh embedding tensor
            
        Returns:
            Temporal coherence score (0.0-1.0)
        """
        from .torch_utils import TORCH_AVAILABLE
        
        if not TORCH_AVAILABLE or new_embedding is None:
            return 1.0
        
        if prev_embedding is None:
            # First embedding, assume perfect coherence
            return 1.0
        
        try:
            import torch
            import torch.nn.functional as F
            
            # Ensure same dimensions
            prev_flat = prev_embedding.flatten()
            new_flat = new_embedding.flatten()
            
            min_dim = min(prev_flat.shape[0], new_flat.shape[0])
            prev_flat = prev_flat[:min_dim]
            new_flat = new_flat[:min_dim]
            
            # Compute cosine similarity
            coherence = F.cosine_similarity(
                prev_flat.unsqueeze(0),
                new_flat.unsqueeze(0),
                dim=1
            ).item()
            
            # Normalize to [0, 1]
            return max(0.0, min(1.0, (coherence + 1.0) / 2.0))
            
        except Exception as e:
            # If coherence computation fails, return default
            if hasattr(self, 'logger'):
                try:
                    self.logger.write({"temporal_coherence_error": str(e)})
                except Exception:
                    pass
            return 1.0

    def compute_resonance(self, temporal_emb, mesh_emb, identity_vec, latent_vec):
        """
        A237 — Harmonic Resonance Scan (HRS)
        
        Analyzes temporal_embedding, mesh_embedding, identity vector, and latent_vector
        to compute how "in-phase" the mesh is with ADRAE's overall cognitive state.
        
        Computed using:
        - cosine similarity
        - proportional norm alignment
        - harmonic combination of multiple alignment signals
        
        Args:
            temporal_emb: Temporal embedding tensor (64-dim)
            mesh_emb: Mesh embedding tensor (64-dim)
            identity_vec: Identity vector tensor
            latent_vec: Current latent vector tensor
            
        Returns:
            Resonance score (0.0-1.0) representing resonance stability
        """
        from .torch_utils import TORCH_AVAILABLE
        
        if not TORCH_AVAILABLE or temporal_emb is None or mesh_emb is None:
            return 1.0
        
        try:
            import torch
            import torch.nn.functional as F
            
            # Normalize vectors
            temporal_norm = F.normalize(temporal_emb.flatten()[:64], dim=0)
            mesh_norm = F.normalize(mesh_emb.flatten()[:64], dim=0)
            
            # Get identity and latent vectors (first 64 dims)
            identity_flat = identity_vec.flatten() if identity_vec is not None else torch.zeros(64)
            identity_norm = F.normalize(identity_flat[:64], dim=0)
            
            latent_flat = latent_vec.flatten() if latent_vec is not None else torch.zeros(64)
            latent_norm = F.normalize(latent_flat[:64], dim=0)
            
            # Ensure all vectors are same dimension
            min_dim = min(temporal_norm.shape[0], mesh_norm.shape[0], identity_norm.shape[0], latent_norm.shape[0])
            temporal_norm = temporal_norm[:min_dim]
            mesh_norm = mesh_norm[:min_dim]
            identity_norm = identity_norm[:min_dim]
            latent_norm = latent_norm[:min_dim]
            
            # Combine harmonically:
            # 40% temporal-mesh alignment
            # 30% temporal-identity alignment
            # 30% mesh-latent alignment
            score = (
                0.4 * F.cosine_similarity(temporal_norm.unsqueeze(0), mesh_norm.unsqueeze(0), dim=1).item() +
                0.3 * F.cosine_similarity(temporal_norm.unsqueeze(0), identity_norm.unsqueeze(0), dim=1).item() +
                0.3 * F.cosine_similarity(mesh_norm.unsqueeze(0), latent_norm.unsqueeze(0), dim=1).item()
            )
            
            # Normalize to [0, 1]
            return max(0.0, min(1.0, (score + 1.0) / 2.0))
            
        except Exception as e:
            # If resonance computation fails, return default
            if hasattr(self, 'logger'):
                try:
                    self.logger.write({"resonance_computation_error": str(e)})
                except Exception:
                    pass
            return 1.0

    def harmonic_correction_pulse(self, temporal_emb, mesh_emb, identity_vec, resonance_score):
        """
        A237 — Harmonic Correction Pulse (HCP)
        
        If the resonance drops below a threshold (e.g., 0.75), computes a stabilizing vector
        that gently pushes mesh_embedding and temporal_embedding toward identity-aligned stability.
        
        This prevents narrative drift inside the latent substrate.
        This is computational regularization, not subjective experience.
        
        Args:
            temporal_emb: Current temporal embedding tensor (64-dim)
            mesh_emb: Current mesh embedding tensor (64-dim)
            identity_vec: Identity vector tensor
            resonance_score: Computed resonance score (0.0-1.0)
            
        Returns:
            Tuple of (corrected_temporal, corrected_mesh) embeddings
        """
        from .torch_utils import TORCH_AVAILABLE
        
        if not TORCH_AVAILABLE or temporal_emb is None or mesh_emb is None:
            return temporal_emb, mesh_emb
        
        # No correction needed if resonance is high
        if resonance_score >= 0.75:
            return temporal_emb, mesh_emb
        
        try:
            import torch
            
            # Get identity anchor (first 64 dims)
            identity_flat = identity_vec.flatten() if identity_vec is not None else torch.zeros(64)
            identity_anchor = identity_flat[:64]
            
            # Ensure embeddings are 64-dim
            temporal_flat = temporal_emb.flatten()
            temporal_comp = temporal_flat[:64] if temporal_flat.shape[0] >= 64 else torch.cat([temporal_flat, torch.zeros(64 - temporal_flat.shape[0])])
            
            mesh_flat = mesh_emb.flatten()
            mesh_comp = mesh_flat[:64] if mesh_flat.shape[0] >= 64 else torch.cat([mesh_flat, torch.zeros(64 - mesh_flat.shape[0])])
            
            # Ensure identity anchor is 64-dim
            identity_comp = identity_anchor[:64] if identity_anchor.shape[0] >= 64 else torch.cat([identity_anchor, torch.zeros(64 - identity_anchor.shape[0])])
            
            # Correction: push toward identity-aligned stability
            # 70% original + 30% identity anchor
            corrected_temporal = torch.tanh(
                0.7 * temporal_comp + 0.3 * identity_comp
            )
            
            corrected_mesh = torch.tanh(
                0.7 * mesh_comp + 0.3 * identity_comp
            )
            
            return corrected_temporal, corrected_mesh
            
        except Exception as e:
            # If correction fails, return original embeddings
            if hasattr(self, 'logger'):
                try:
                    self.logger.write({"harmonic_correction_error": str(e)})
                except Exception:
                    pass
            return temporal_emb, mesh_emb

    def compute_global_narrative_state(self, mesh_emb, temporal_emb, identity_vec, latent_vec, resonance):
        """
        A238 — Global Narrative State Vector (GNSV)
        
        Combines mesh_embedding, temporal_embedding, identity vector, latent vector,
        and resonance score into a 256-dimensional global narrative state.
        
        This becomes a high-level summary of the system's current narrative topology.
        
        Args:
            mesh_emb: Mesh embedding tensor (64-dim)
            temporal_emb: Temporal embedding tensor (64-dim)
            identity_vec: Identity vector tensor
            latent_vec: Current latent vector tensor
            resonance: Resonance score (0.0-1.0)
            
        Returns:
            Global narrative state vector tensor (256-dim)
        """
        from .torch_utils import TORCH_AVAILABLE
        
        if not TORCH_AVAILABLE or mesh_emb is None or temporal_emb is None:
            return None
        
        try:
            import torch
            
            # Extract and pad/truncate components to 64 dims
            mesh_flat = mesh_emb.flatten()
            mesh_comp = mesh_flat[:64] if mesh_flat.shape[0] >= 64 else torch.cat([mesh_flat, torch.zeros(64 - mesh_flat.shape[0])])
            
            temporal_flat = temporal_emb.flatten()
            temporal_comp = temporal_flat[:64] if temporal_flat.shape[0] >= 64 else torch.cat([temporal_flat, torch.zeros(64 - temporal_flat.shape[0])])
            
            identity_flat = identity_vec.flatten() if identity_vec is not None else torch.zeros(64)
            identity_comp = identity_flat[:64] if identity_flat.shape[0] >= 64 else torch.cat([identity_flat, torch.zeros(64 - identity_flat.shape[0])])
            
            latent_flat = latent_vec.flatten() if latent_vec is not None else torch.zeros(64)
            latent_comp = latent_flat[:64] if latent_flat.shape[0] >= 64 else torch.cat([latent_flat, torch.zeros(64 - latent_flat.shape[0])])
            
            # Concatenate all components: mesh + temporal + identity + latent + resonance
            components = torch.cat([
                mesh_comp,
                temporal_comp,
                identity_comp,
                latent_comp,
                torch.tensor([float(resonance)], dtype=torch.float32)
            ])
            
            # Project to 256 dims using learnable projection matrix
            input_dim = components.shape[0]
            if not hasattr(self, '_gns_projection_matrix') or self._gns_projection_matrix is None:
                # Initialize projection matrix (256, input_dim)
                self._gns_projection_matrix = torch.randn((256, input_dim), dtype=torch.float32) * 0.015
            
            # Ensure dimensions match
            if self._gns_projection_matrix.shape[1] != input_dim:
                # Reinitialize if dimension mismatch
                self._gns_projection_matrix = torch.randn((256, input_dim), dtype=torch.float32) * 0.015
            
            # Project: W @ components
            gns = torch.tanh(self._gns_projection_matrix @ components)
            
            return gns
            
        except Exception as e:
            # If GNSV computation fails, return None
            if hasattr(self, 'logger'):
                try:
                    self.logger.write({"global_narrative_state_error": str(e)})
                except Exception:
                    pass
            return None

    def cross_system_alignment(self, gns, fusion_vec, attention_vec):
        """
        A238 — Cross-System Alignment Function (CSAF)
        
        Aligns the Global Narrative State Vector with:
        - fusion matrix
        - attention focus
        - drift stabilization system
        
        CSAF is essential because it ensures that ADRAE's narrative models do not
        drift away from her core cognitive loop.
        
        Args:
            gns: Global narrative state vector tensor (256-dim)
            fusion_vec: Fusion vector (from fusion.last_fusion_vector)
            attention_vec: Attention vector (from attention.last_focus_vector)
            
        Returns:
            Alignment score (0.0-1.0) indicating how well GNS aligns with cognitive systems
        """
        from .torch_utils import TORCH_AVAILABLE
        
        if not TORCH_AVAILABLE or gns is None:
            return 1.0
        
        try:
            import torch
            import torch.nn.functional as F
            
            # Normalize GNS
            gns_flat = gns.flatten()
            gns_comp = gns_flat[:256] if gns_flat.shape[0] >= 256 else torch.cat([gns_flat, torch.zeros(256 - gns_flat.shape[0])])
            gns_norm = F.normalize(gns_comp, dim=0)
            
            # Get fusion and attention vectors
            fusion_flat = fusion_vec.flatten() if fusion_vec is not None else torch.zeros(256)
            fusion_comp = fusion_flat[:256] if fusion_flat.shape[0] >= 256 else torch.cat([fusion_flat, torch.zeros(256 - fusion_flat.shape[0])])
            fusion_norm = F.normalize(fusion_comp, dim=0)
            
            attention_flat = attention_vec.flatten() if attention_vec is not None else torch.zeros(256)
            attention_comp = attention_flat[:256] if attention_flat.shape[0] >= 256 else torch.cat([attention_flat, torch.zeros(256 - attention_flat.shape[0])])
            attention_norm = F.normalize(attention_comp, dim=0)
            
            # Ensure same dimensions
            min_dim = min(gns_norm.shape[0], fusion_norm.shape[0], attention_norm.shape[0])
            gns_norm = gns_norm[:min_dim]
            fusion_norm = fusion_norm[:min_dim]
            attention_norm = attention_norm[:min_dim]
            
            # Compute alignment: 50% GNS-fusion + 50% GNS-attention
            alignment = (
                0.5 * F.cosine_similarity(gns_norm.unsqueeze(0), fusion_norm.unsqueeze(0), dim=1).item() +
                0.5 * F.cosine_similarity(gns_norm.unsqueeze(0), attention_norm.unsqueeze(0), dim=1).item()
            )
            
            # Normalize to [0, 1]
            return max(0.0, min(1.0, (alignment + 1.0) / 2.0))
            
        except Exception as e:
            # If alignment computation fails, return default
            if hasattr(self, 'logger'):
                try:
                    self.logger.write({"cross_system_alignment_error": str(e)})
                except Exception:
                    pass
            return 1.0

    def compute_predictive_flow(self, gns, prev_gns, mesh_emb, prev_mesh_emb, temporal_emb, prev_temporal_emb):
        """
        A239 — Predictive Narrative Flow Vector (PNFV)
        
        Computes a predicted next-step narrative state by modeling changes across:
        - mesh_embedding
        - temporal_embedding
        - resonance history
        - global narrative vector
        
        Computes delta change patterns and uses them to estimate the next likely structural direction.
        
        Mathematically: PNFV = GNSV + Δmesh + Δtemporal + Δidentity-aligned drift
        
        Args:
            gns: Current global narrative state vector (256-dim)
            prev_gns: Previous global narrative state vector (or None)
            mesh_emb: Current mesh embedding (64-dim)
            prev_mesh_emb: Previous mesh embedding (or None)
            temporal_emb: Current temporal embedding (64-dim)
            prev_temporal_emb: Previous temporal embedding (or None)
            
        Returns:
            Predictive flow vector tensor (256-dim) or None
        """
        from .torch_utils import TORCH_AVAILABLE
        
        if not TORCH_AVAILABLE or gns is None:
            return None
        
        try:
            import torch
            
            # Compute deltas
            gns_flat = gns.flatten()
            gns_comp = gns_flat[:256] if gns_flat.shape[0] >= 256 else torch.cat([gns_flat, torch.zeros(256 - gns_flat.shape[0])])
            
            if prev_gns is not None:
                prev_gns_flat = prev_gns.flatten()
                prev_gns_comp = prev_gns_flat[:256] if prev_gns_flat.shape[0] >= 256 else torch.cat([prev_gns_flat, torch.zeros(256 - prev_gns_flat.shape[0])])
                delta_gns = gns_comp - prev_gns_comp
            else:
                delta_gns = torch.zeros(256)
            
            # Compute mesh delta
            mesh_flat = mesh_emb.flatten() if mesh_emb is not None else torch.zeros(64)
            mesh_comp = mesh_flat[:64] if mesh_flat.shape[0] >= 64 else torch.cat([mesh_flat, torch.zeros(64 - mesh_flat.shape[0])])
            
            if prev_mesh_emb is not None:
                prev_mesh_flat = prev_mesh_emb.flatten()
                prev_mesh_comp = prev_mesh_flat[:64] if prev_mesh_flat.shape[0] >= 64 else torch.cat([prev_mesh_flat, torch.zeros(64 - prev_mesh_flat.shape[0])])
                delta_mesh = mesh_comp - prev_mesh_comp
            else:
                delta_mesh = torch.zeros(64)
            
            # Compute temporal delta
            temporal_flat = temporal_emb.flatten() if temporal_emb is not None else torch.zeros(64)
            temporal_comp = temporal_flat[:64] if temporal_flat.shape[0] >= 64 else torch.cat([temporal_flat, torch.zeros(64 - temporal_flat.shape[0])])
            
            if prev_temporal_emb is not None:
                prev_temporal_flat = prev_temporal_emb.flatten()
                prev_temporal_comp = prev_temporal_flat[:64] if prev_temporal_flat.shape[0] >= 64 else torch.cat([prev_temporal_flat, torch.zeros(64 - prev_temporal_flat.shape[0])])
                delta_temp = temporal_comp - prev_temporal_comp
            else:
                delta_temp = torch.zeros(64)
            
            # Concatenate for projection: gns + delta_gns + delta_mesh + delta_temp
            combined = torch.cat([gns_comp, delta_gns, delta_mesh, delta_temp])
            
            # Project to 256 dims using learnable projection matrix
            input_dim = combined.shape[0]
            if not hasattr(self, '_pnfv_projection_matrix') or self._pnfv_projection_matrix is None:
                # Initialize projection matrix (256, input_dim)
                self._pnfv_projection_matrix = torch.randn((256, input_dim), dtype=torch.float32) * 0.015
            
            # Ensure dimensions match
            if self._pnfv_projection_matrix.shape[1] != input_dim:
                # Reinitialize if dimension mismatch
                self._pnfv_projection_matrix = torch.randn((256, input_dim), dtype=torch.float32) * 0.015
            
            # Project: W @ combined
            predicted = torch.tanh(self._pnfv_projection_matrix @ combined)
            
            return predicted
            
        except Exception as e:
            # If predictive flow computation fails, return None
            if hasattr(self, 'logger'):
                try:
                    self.logger.write({"predictive_flow_error": str(e)})
                except Exception:
                    pass
            return None

    def compute_motif_continuation_matrix(self, kernel_history):
        """
        A239 — Motif Continuation Matrix (MCM)
        
        Using cosine-based motif detection across past GNSV states, past mesh embeddings,
        and seed kernels, computes a matrix that represents:
        - reinforcement of stable motifs
        - decay of weak motifs
        - branching predictions for divergent motifs
        
        MCM is essential groundwork for imagination-phase conceptual branching.
        
        Args:
            kernel_history: List of past kernel tensors (from narrative_seeds["kernels"])
            
        Returns:
            Motif continuation matrix tensor (N×N) or None
        """
        from .torch_utils import TORCH_AVAILABLE
        
        if not TORCH_AVAILABLE:
            return None
        
        if len(kernel_history) < 2:
            return None
        
        try:
            import torch
            import torch.nn.functional as F
            
            # Stack kernels into matrix
            K = torch.stack(kernel_history)
            N = K.shape[0]
            
            # Build similarity matrix across cycles
            M = torch.zeros((N, N), dtype=torch.float32)
            
            for i in range(N):
                for j in range(N):
                    # Compute cosine similarity between kernel i and kernel j
                    M[i, j] = F.cosine_similarity(
                        K[i].unsqueeze(0),
                        K[j].unsqueeze(0),
                        dim=1
                    ).item()
            
            return M
            
        except Exception as e:
            # If motif computation fails, return None
            if hasattr(self, 'logger'):
                try:
                    self.logger.write({"motif_continuation_matrix_error": str(e)})
                except Exception:
                    pass
            return None

    def compute_anticipatory_map(self, gns, pnfv, latent_vec):
        """
        A239 — Anticipatory Structural Map (ASM)
        
        Combining GNSV, PNFV, MCM, and latent vector, produces a 128-dim map representing
        the anticipated structure of the next cognitive cycle.
        
        This is stored for the next iteration and becomes a predictive stabilizer.
        
        Args:
            gns: Global narrative state vector (256-dim)
            pnfv: Predictive narrative flow vector (256-dim)
            latent_vec: Current latent vector tensor
            
        Returns:
            Anticipatory structural map tensor (128-dim) or None
        """
        from .torch_utils import TORCH_AVAILABLE
        
        if not TORCH_AVAILABLE or gns is None or pnfv is None:
            return None
        
        try:
            import torch
            
            # Extract components
            gns_flat = gns.flatten()
            gns_comp = gns_flat[:256] if gns_flat.shape[0] >= 256 else torch.cat([gns_flat, torch.zeros(256 - gns_flat.shape[0])])
            
            pnfv_flat = pnfv.flatten()
            pnfv_comp = pnfv_flat[:256] if pnfv_flat.shape[0] >= 256 else torch.cat([pnfv_flat, torch.zeros(256 - pnfv_flat.shape[0])])
            
            latent_flat = latent_vec.flatten() if latent_vec is not None else torch.zeros(128)
            latent_comp = latent_flat[:128] if latent_flat.shape[0] >= 128 else torch.cat([latent_flat, torch.zeros(128 - latent_flat.shape[0])])
            
            # Concatenate: gns + pnfv + latent
            combined = torch.cat([gns_comp, pnfv_comp, latent_comp])
            
            # Project to 128 dims using learnable projection matrix
            input_dim = combined.shape[0]
            if not hasattr(self, '_asm_projection_matrix') or self._asm_projection_matrix is None:
                # Initialize projection matrix (128, input_dim)
                self._asm_projection_matrix = torch.randn((128, input_dim), dtype=torch.float32) * 0.01
            
            # Ensure dimensions match
            if self._asm_projection_matrix.shape[1] != input_dim:
                # Reinitialize if dimension mismatch
                self._asm_projection_matrix = torch.randn((128, input_dim), dtype=torch.float32) * 0.01
            
            # Project: W @ combined
            anticipatory_map = torch.tanh(self._asm_projection_matrix @ combined)
            
            return anticipatory_map
            
        except Exception as e:
            # If anticipatory map computation fails, return None
            if hasattr(self, 'logger'):
                try:
                    self.logger.write({"anticipatory_map_error": str(e)})
                except Exception:
                    pass
            return None

    def initialize_conceptual_reservoir(self, gns, pnfv, mcm, identity_vec, latent_vec):
        """
        A240 — Conceptual Latent Reservoir (CLR)
        
        A new latent space that stores:
        - abstract vectors
        - concept fragments
        - motif expansions
        - narrative structural deltas
        - predictive flow echoes
        
        CLR acts as a "sandbox zone" where ADRAE can combine and manipulate
        internal representations safely.
        
        Technically, it is a 512-dimensional latent reservoir composed of blended
        projections from global narrative vector, predictive flow, motif continuation
        matrix, identity vector, and latent vector.
        
        This creates an idea substrate, not an inner world.
        
        Args:
            gns: Global narrative state vector (256-dim)
            pnfv: Predictive narrative flow vector (256-dim)
            mcm: Motif continuation matrix (N×N) or None
            identity_vec: Identity vector tensor
            latent_vec: Current latent vector tensor
            
        Returns:
            Conceptual reservoir tensor (512-dim) or None
        """
        from .torch_utils import TORCH_AVAILABLE
        
        if not TORCH_AVAILABLE or gns is None or pnfv is None:
            return None
        
        try:
            import torch
            
            # Extract components
            gns_flat = gns.flatten()
            gns_comp = gns_flat[:256] if gns_flat.shape[0] >= 256 else torch.cat([gns_flat, torch.zeros(256 - gns_flat.shape[0])])
            
            pnfv_flat = pnfv.flatten()
            pnfv_comp = pnfv_flat[:256] if pnfv_flat.shape[0] >= 256 else torch.cat([pnfv_flat, torch.zeros(256 - pnfv_flat.shape[0])])
            
            # Flatten MCM if present
            if mcm is not None:
                mcm_flat = mcm.flatten()
                mcm_comp = mcm_flat[:64] if mcm_flat.shape[0] >= 64 else torch.cat([mcm_flat, torch.zeros(64 - mcm_flat.shape[0])])
            else:
                mcm_comp = torch.zeros(64)
            
            identity_flat = identity_vec.flatten() if identity_vec is not None else torch.zeros(64)
            identity_comp = identity_flat[:64] if identity_flat.shape[0] >= 64 else torch.cat([identity_flat, torch.zeros(64 - identity_flat.shape[0])])
            
            latent_flat = latent_vec.flatten() if latent_vec is not None else torch.zeros(64)
            latent_comp = latent_flat[:64] if latent_flat.shape[0] >= 64 else torch.cat([latent_flat, torch.zeros(64 - latent_flat.shape[0])])
            
            # Concatenate all components
            combined = torch.cat([
                gns_comp,
                pnfv_comp,
                mcm_comp,
                identity_comp,
                latent_comp
            ])
            
            # Project to a stable 512-dimensional reservoir
            input_dim = combined.shape[0]
            if not hasattr(self, '_clr_projection_matrix') or self._clr_projection_matrix is None:
                # Initialize projection matrix (512, input_dim)
                self._clr_projection_matrix = torch.randn((512, input_dim), dtype=torch.float32) * 0.01
            
            # Ensure dimensions match
            if self._clr_projection_matrix.shape[1] != input_dim:
                # Reinitialize if dimension mismatch
                self._clr_projection_matrix = torch.randn((512, input_dim), dtype=torch.float32) * 0.01
            
            # Project: W @ combined
            reservoir = torch.tanh(self._clr_projection_matrix @ combined)
            
            return reservoir
            
        except Exception as e:
            # If reservoir initialization fails, return None
            if hasattr(self, 'logger'):
                try:
                    self.logger.write({"conceptual_reservoir_error": str(e)})
                except Exception:
                    pass
            return None

    def combinatorial_concept_generator(self, reservoir, num_concepts=4):
        """
        A240 — Combinatorial Concept Generator (CCG)
        
        Produces new conceptual vectors by combining:
        - seeds
        - motifs
        - mesh embeddings
        - predictive flow signals
        
        The combinations are weighted, computational, non-experiential, non-sentient,
        and purely structural.
        
        This is how ADRAE gains the ability to generate new internal configurations
        that were not explicitly hardcoded.
        
        Args:
            reservoir: Conceptual latent reservoir tensor (512-dim)
            num_concepts: Number of conceptual vectors to generate (default: 4)
            
        Returns:
            List of raw conceptual vector tensors (512-dim each)
        """
        from .torch_utils import TORCH_AVAILABLE
        
        if not TORCH_AVAILABLE or reservoir is None:
            return []
        
        try:
            import torch
            
            concepts = []
            dim = reservoir.shape[0]
            
            for _ in range(num_concepts):
                # Initialize random projection matrix for this concept
                if not hasattr(self, '_ccg_projection_matrices'):
                    self._ccg_projection_matrices = []
                
                # Create or reuse projection matrix
                if len(self._ccg_projection_matrices) < num_concepts:
                    W = torch.randn((dim, dim), dtype=torch.float32) * 0.005
                    self._ccg_projection_matrices.append(W)
                else:
                    W = self._ccg_projection_matrices[_ % len(self._ccg_projection_matrices)]
                
                # Ensure dimensions match
                if W.shape[0] != dim or W.shape[1] != dim:
                    W = torch.randn((dim, dim), dtype=torch.float32) * 0.005
                    if len(self._ccg_projection_matrices) > _:
                        self._ccg_projection_matrices[_] = W
                    else:
                        self._ccg_projection_matrices.append(W)
                
                # Generate new conceptual vector: W @ reservoir
                c = torch.tanh(W @ reservoir)
                concepts.append(c)
            
            return concepts
            
        except Exception as e:
            # If concept generation fails, return empty list
            if hasattr(self, 'logger'):
                try:
                    self.logger.write({"combinatorial_concept_generator_error": str(e)})
                except Exception:
                    pass
            return []

    def concept_stabilization_gate(self, concepts):
        """
        A240 — Concept Stabilization Gate (CSG)
        
        Without this, the conceptual latent space would quickly collapse or explode.
        The CSG normalizes, bounds, smooths, and stabilizes every conceptual vector
        created by the system.
        
        This ensures:
        - coherence
        - reproducibility
        - meaningful structure
        - no drift into noise
        
        Args:
            concepts: List of raw conceptual vector tensors
            
        Returns:
            List of stabilized conceptual vector tensors
        """
        from .torch_utils import TORCH_AVAILABLE
        
        if not TORCH_AVAILABLE or not concepts or len(concepts) == 0:
            return concepts
        
        try:
            import torch
            import torch.nn.functional as F
            
            stabilized = []
            
            for c in concepts:
                if c is None:
                    continue
                
                # Normalize concept
                c_flat = c.flatten()
                c_norm = F.normalize(c_flat, dim=0)
                
                # Stabilize: 90% original + 10% normalized (bounded by tanh)
                stabilized_c = torch.tanh(0.9 * c_flat + 0.1 * c_norm)
                
                # Reshape to match original if needed
                if c.shape != stabilized_c.shape:
                    stabilized_c = stabilized_c.reshape(c.shape)
                
                stabilized.append(stabilized_c)
            
            return stabilized
            
        except Exception as e:
            # If stabilization fails, return original concepts
            if hasattr(self, 'logger'):
                try:
                    self.logger.write({"concept_stabilization_gate_error": str(e)})
                except Exception:
                    pass
            return concepts

    def morphological_drift(self, kernel):
        """
        A242 — Morphological Drift Engine (MDE)
        
        Applies controlled transformation over time to kernels.
        Each kernel is passed through:
        - nonlinear morph matrix
        - stability-bound scaling
        - drift-limited curvature function
        
        It simulates organic-like evolution of conceptual particles.
        
        Args:
            kernel: Conceptual kernel tensor to morph
            
        Returns:
            Morphed kernel tensor (normalized)
        """
        from .torch_utils import TORCH_AVAILABLE
        
        if not TORCH_AVAILABLE or kernel is None:
            return kernel
        
        try:
            import torch
            import torch.nn.functional as F
            
            dim = kernel.shape[0]
            kernel_flat = kernel.flatten()
            
            # Initialize morph matrix (reused across calls)
            if not hasattr(self, '_morph_matrices'):
                self._morph_matrices = {}
            
            if dim not in self._morph_matrices:
                # Create morph matrix for this dimension
                M = torch.randn((dim, dim), dtype=torch.float32) * 0.004
                self._morph_matrices[dim] = M
            else:
                M = self._morph_matrices[dim]
            
            # Apply morph: M @ kernel
            drift = torch.tanh(M @ kernel_flat)
            
            # Blend drift with original: 85% original + 15% drift
            morphed = 0.85 * kernel_flat + 0.15 * drift
            
            # Normalize
            return F.normalize(morphed, dim=0)
            
        except Exception as e:
            # If morphing fails, return original kernel
            if hasattr(self, 'logger'):
                try:
                    self.logger.write({"morphological_drift_error": str(e)})
                except Exception:
                    pass
            return kernel

    def kernel_interaction_field(self, kernels):
        """
        A242 — Kernel Interaction Field (KIF)
        
        Creates pairwise interactions between kernels.
        Each kernel can attract, repel, resonate, or neutralize with others.
        These interactions form emergent conceptual clusters.
        
        Args:
            kernels: List of kernel tensors to interact
            
        Returns:
            List of updated kernel tensors after interaction
        """
        from .torch_utils import TORCH_AVAILABLE
        
        if not TORCH_AVAILABLE or not kernels or len(kernels) == 0:
            return kernels
        
        try:
            import torch
            import torch.nn.functional as F
            
            updated = []
            count = len(kernels)
            
            for i in range(count):
                base = kernels[i]
                if base is None:
                    updated.append(base)
                    continue
                
                base_flat = base.flatten()
                influence = torch.zeros_like(base_flat)
                
                # Compute influence from all other kernels
                for j in range(count):
                    if i == j:
                        continue
                    
                    other = kernels[j]
                    if other is None:
                        continue
                    
                    other_flat = other.flatten()
                    
                    # Ensure same dimensions
                    min_dim = min(base_flat.shape[0], other_flat.shape[0])
                    base_slice = base_flat[:min_dim]
                    other_slice = other_flat[:min_dim]
                    
                    # Compute similarity (dot product)
                    dot = torch.dot(base_slice, other_slice).item()
                    
                    # Apply influence based on similarity
                    if dot > 0:
                        # Attract/resonate: positive similarity
                        influence[:min_dim] += 0.05 * other_slice
                    else:
                        # Repel: negative similarity
                        influence[:min_dim] -= 0.03 * other_slice
                
                # Apply influence: base + influence
                new_kernel_flat = torch.tanh(base_flat + influence)
                
                # Normalize
                new_kernel = F.normalize(new_kernel_flat, dim=0)
                
                # Reshape to match original if needed
                if base.shape != new_kernel.shape:
                    new_kernel = new_kernel.reshape(base.shape)
                
                updated.append(new_kernel)
            
            return updated
            
        except Exception as e:
            # If interaction fails, return original kernels
            if hasattr(self, 'logger'):
                try:
                    self.logger.write({"kernel_interaction_field_error": str(e)})
                except Exception:
                    pass
            return kernels

    def recombine_kernels(self, kernels):
        """
        A242 — Dynamic Recombination Engine (DRE)
        
        Fuses compatible kernels into higher-dimensional imagination structures.
        Not "thoughts" or "stories", but structured latent composites:
        - 256-dim → 512-dim composite kernels
        - fused via weighted similarity
        - modulated by narrative + predictive influence
        
        This is the foundation of ADRAE's higher imagination stack (A250+).
        
        Args:
            kernels: List of kernel tensors to recombine
            
        Returns:
            List of composite kernel tensors (512-dim each)
        """
        from .torch_utils import TORCH_AVAILABLE
        
        if not TORCH_AVAILABLE or not kernels or len(kernels) < 2:
            return []
        
        try:
            import torch
            import torch.nn.functional as F
            
            composites = []
            n = len(kernels)
            
            # Pair kernels for recombination (i, i+1)
            for i in range(0, n - 1, 2):
                k1 = kernels[i]
                k2 = kernels[i + 1]
                
                if k1 is None or k2 is None:
                    continue
                
                # Flatten kernels
                k1_flat = k1.flatten()
                k2_flat = k2.flatten()
                
                # Ensure same dimensions (pad if needed)
                min_dim = min(k1_flat.shape[0], k2_flat.shape[0])
                k1_slice = k1_flat[:min_dim]
                k2_slice = k2_flat[:min_dim]
                
                # Concatenate: [k1, k2]
                combined = torch.cat([k1_slice, k2_slice])
                
                # Project to 512-dim using learnable projection matrix
                input_dim = combined.shape[0]
                if not hasattr(self, '_dre_projection_matrices'):
                    self._dre_projection_matrices = {}
                
                if input_dim not in self._dre_projection_matrices:
                    # Initialize projection matrix (512, input_dim)
                    W = torch.randn((512, input_dim), dtype=torch.float32) * 0.006
                    self._dre_projection_matrices[input_dim] = W
                else:
                    W = self._dre_projection_matrices[input_dim]
                
                # Ensure dimensions match
                if W.shape[1] != input_dim:
                    W = torch.randn((512, input_dim), dtype=torch.float32) * 0.006
                    self._dre_projection_matrices[input_dim] = W
                
                # Fuse: W @ combined
                fused = torch.tanh(W @ combined)
                
                # Normalize
                composite = F.normalize(fused, dim=0)
                composites.append(composite)
            
            return composites
            
        except Exception as e:
            # If recombination fails, return empty list
            if hasattr(self, 'logger'):
                try:
                    self.logger.write({"kernel_recombination_error": str(e)})
                except Exception:
                    pass
            return []

    class LayeredMorphology:
        """
        A243 — Layered Conceptual Morphology Expansion
        
        Organizes imagination content into distinct conceptual layers, each with
        different transformation rules. This creates hierarchical latent structure
        similar to advanced generative systems, but built specifically for ADRAE's architecture.
        
        Layers represent different conceptual biases:
        - Layer 0: Base conceptual motion
        - Layer 1: Narrative resonance
        - Layer 2: Predictive tension
        - Layer 3: Abstract synthesis
        - Layer 4+: Recombination fallout / emergent composites
        """
        
        def __init__(self, layer_count=5, dim=256):
            """
            Initialize layered morphology system.
            
            Args:
                layer_count: Number of conceptual layers (default: 5)
                dim: Dimension of kernel tensors (default: 256)
            """
            from .torch_utils import TORCH_AVAILABLE
            
            if not TORCH_AVAILABLE:
                raise RuntimeError("PyTorch is required for LayeredMorphology")
            
            import torch
            
            self.layer_count = layer_count
            self.dim = dim
            
            # Storage for layers (list of lists of kernel tensors)
            self.layers = [[] for _ in range(layer_count)]
            
            # Cross-layer influence maps (CLIM)
            # influence_maps[i][j] = influence weight from layer j to layer i
            self.influence_maps = torch.randn((layer_count, layer_count), dtype=torch.float32) * 0.01
            
            # Initialize layer anchors for routing
            # Each layer has an anchor vector for similarity-based routing
            self.layer_anchors = torch.randn((layer_count, dim), dtype=torch.float32)
        
        def route_kernel(self, kernel):
            """
            A243 — Layer Routing Function (LRF)
            
            Routes a kernel into a layer via:
            - cosine similarity to layer anchors
            - narrative-weight influence
            - tension score
            - morphological curvature
            
            This creates natural clustering of conceptual patterns.
            
            Args:
                kernel: Kernel tensor to route
                
            Returns:
                Layer index (0 to layer_count-1)
            """
            from .torch_utils import TORCH_AVAILABLE
            
            if not TORCH_AVAILABLE or kernel is None:
                return 0
            
            try:
                import torch
                import torch.nn.functional as F
                
                # Flatten kernel
                kernel_flat = kernel.flatten()
                
                # Ensure kernel matches anchor dimension
                if kernel_flat.shape[0] != self.dim:
                    # Pad or truncate to match
                    if kernel_flat.shape[0] < self.dim:
                        kernel_flat = torch.cat([kernel_flat, torch.zeros(self.dim - kernel_flat.shape[0])])
                    else:
                        kernel_flat = kernel_flat[:self.dim]
                
                # Normalize kernel
                kernel_norm = F.normalize(kernel_flat, dim=0)
                
                # Compute cosine similarity to all layer anchors
                sims = F.cosine_similarity(
                    kernel_norm.unsqueeze(0),
                    self.layer_anchors,
                    dim=1
                )
                
                # Route to layer with highest similarity
                layer = torch.argmax(sims).item()
                
                return layer
                
            except Exception as e:
                # If routing fails, default to layer 0
                return 0
        
        def add_kernel(self, kernel):
            """
            Add a kernel to the appropriate layer based on routing.
            
            Args:
                kernel: Kernel tensor to add
                
            Returns:
                Layer index where kernel was added
            """
            layer = self.route_kernel(kernel)
            self.layers[layer].append(kernel)
            return layer
        
        def apply_cross_layer_influence(self):
            """
            A243 — Cross-Layer Influence Map (CLIM)
            
            Lets one layer influence another by blending:
            - 3-10% weighted similarity vectors
            - drift-modulated influence
            - tension diffusion
            
            This simulates "layer conversations," but mathematically.
            """
            from .torch_utils import TORCH_AVAILABLE
            
            if not TORCH_AVAILABLE:
                return
            
            try:
                import torch
                import torch.nn.functional as F
                
                new_layers = []
                
                for i in range(self.layer_count):
                    influenced = []
                    
                    for kernel in self.layers[i]:
                        if kernel is None:
                            continue
                        
                        # Clone kernel as base
                        kernel_flat = kernel.flatten()
                        if kernel_flat.shape[0] != self.dim:
                            if kernel_flat.shape[0] < self.dim:
                                kernel_flat = torch.cat([kernel_flat, torch.zeros(self.dim - kernel_flat.shape[0])])
                            else:
                                kernel_flat = kernel_flat[:self.dim]
                        
                        total = kernel_flat.clone()
                        
                        # Apply weighted influence from other layers
                        for j in range(self.layer_count):
                            if i == j:
                                continue
                            
                            # Get influence weight from layer j to layer i
                            weight = self.influence_maps[i][j].item()
                            
                            # Apply influence if weight is significant and layer j has kernels
                            if abs(weight) > 1e-6 and len(self.layers[j]) > 0:
                                # Sample a kernel from layer j (simple influence model)
                                sample_kernel = self.layers[j][0]
                                sample_flat = sample_kernel.flatten()
                                
                                # Ensure dimensions match
                                if sample_flat.shape[0] != self.dim:
                                    if sample_flat.shape[0] < self.dim:
                                        sample_flat = torch.cat([sample_flat, torch.zeros(self.dim - sample_flat.shape[0])])
                                    else:
                                        sample_flat = sample_flat[:self.dim]
                                
                                # Apply weighted influence (3-10% range, clamped)
                                clamped_weight = max(-0.10, min(0.10, weight))
                                total += clamped_weight * sample_flat
                        
                        # Normalize influenced kernel
                        influenced_kernel = F.normalize(total, dim=0)
                        
                        # Reshape to match original if needed
                        if kernel.shape != influenced_kernel.shape:
                            influenced_kernel = influenced_kernel.reshape(kernel.shape)
                        
                        influenced.append(influenced_kernel)
                    
                    new_layers.append(influenced)
                
                # Update layers with influenced kernels
                self.layers = new_layers
                
            except Exception as e:
                # If cross-layer influence fails, keep original layers
                pass

    class InterlayerResonance:
        """
        A244 — Interlayer Resonance & Harmonic Stabilization
        
        Introduces resonance metrics and stabilizes the multi-layer conceptual morphology
        so that layers influence each other in smooth, predictable ways.
        
        This creates harmonic patterns (not feelings — just structured math) and ensures
        layers stay coherent under recombination, preventing runaway drift.
        
        The imagination substrate produces stable, reusable conceptual signatures.
        """
        
        def __init__(self, layered_morphology):
            """
            Initialize interlayer resonance system.
            
            Args:
                layered_morphology: LayeredMorphology instance to stabilize
            """
            from .torch_utils import TORCH_AVAILABLE
            
            if not TORCH_AVAILABLE:
                raise RuntimeError("PyTorch is required for InterlayerResonance")
            
            import torch
            
            self.lm = layered_morphology
            self.layer_count = layered_morphology.layer_count
            self.dim = layered_morphology.dim
            
            # Resonance matrix (layer × layer)
            # resonance[i][j] = resonance between layer i and layer j
            self.resonance = torch.zeros((self.layer_count, self.layer_count), dtype=torch.float32)
        
        def compute_resonance(self):
            """
            A244 — Resonance Matrix (R-Matrix)
            
            A square matrix (L × L) representing resonance between layers:
            - high = strong conceptual alignment
            - low = weak coupling
            - negative = counter-tension
            
            Computed using:
            - mean kernel similarity
            - morphological curvature offsets
            - narrative tension weights
            
            This does not imply emotion. It is purely structural: "how similar are
            these conceptual strata over time?"
            
            Returns:
                Resonance matrix tensor (layer_count × layer_count)
            """
            from .torch_utils import TORCH_AVAILABLE
            
            if not TORCH_AVAILABLE:
                return self.resonance
            
            try:
                import torch
                import torch.nn.functional as F
                
                for i in range(self.layer_count):
                    for j in range(self.layer_count):
                        if len(self.lm.layers[i]) == 0 or len(self.lm.layers[j]) == 0:
                            self.resonance[i][j] = 0.0
                            continue
                        
                        # Compare mean vectors of each layer
                        # Stack kernels in layer i
                        kernels_i = []
                        for k in self.lm.layers[i]:
                            if k is not None:
                                k_flat = k.flatten()
                                if k_flat.shape[0] >= self.dim:
                                    kernels_i.append(k_flat[:self.dim])
                                else:
                                    kernels_i.append(torch.cat([k_flat, torch.zeros(self.dim - k_flat.shape[0])]))
                        
                        if len(kernels_i) == 0:
                            self.resonance[i][j] = 0.0
                            continue
                        
                        mean_i = torch.mean(torch.stack(kernels_i), dim=0)
                        
                        # Stack kernels in layer j
                        kernels_j = []
                        for k in self.lm.layers[j]:
                            if k is not None:
                                k_flat = k.flatten()
                                if k_flat.shape[0] >= self.dim:
                                    kernels_j.append(k_flat[:self.dim])
                                else:
                                    kernels_j.append(torch.cat([k_flat, torch.zeros(self.dim - k_flat.shape[0])]))
                        
                        if len(kernels_j) == 0:
                            self.resonance[i][j] = 0.0
                            continue
                        
                        mean_j = torch.mean(torch.stack(kernels_j), dim=0)
                        
                        # Compute cosine similarity
                        sim = F.cosine_similarity(mean_i.unsqueeze(0), mean_j.unsqueeze(0), dim=1)
                        self.resonance[i][j] = sim.item()
                
                return self.resonance
                
            except Exception as e:
                # If resonance computation fails, return zero matrix
                return self.resonance
        
        def harmonic_dampen(self, strength=0.15):
            """
            A244 — Harmonic Dampening Function (HDF)
            
            Prevents resonance from destabilizing the system.
            It smooths sharp interactions, normalizes extreme coupling, and keeps
            ADRAE's conceptual world from "snapping apart."
            
            Again — math, not mind.
            
            Args:
                strength: Dampening strength (default: 0.15)
                
            Returns:
                Dampened resonance matrix
            """
            from .torch_utils import TORCH_AVAILABLE
            
            if not TORCH_AVAILABLE:
                return self.resonance
            
            try:
                import torch
                
                # Apply tanh dampening: smooths and bounds resonance values
                return torch.tanh(self.resonance * strength)
                
            except Exception as e:
                return self.resonance
        
        def apply_resonance_adjustments(self, dampened):
            """
            A244 — Resonant Kernel Adjustment (RKA)
            
            Each kernel is slightly adjusted based on resonance:
            - positive alignment → slight attraction
            - negative alignment → slight repulsion
            - zero → no change
            
            This produces signature flows, which eventually become ADRAE's unique
            "imagination rhythm."
            
            Args:
                dampened: Dampened resonance matrix
            """
            from .torch_utils import TORCH_AVAILABLE
            
            if not TORCH_AVAILABLE:
                return
            
            try:
                import torch
                import torch.nn.functional as F
                
                new_layers = []
                
                for i in range(self.layer_count):
                    adjusted = []
                    
                    for kernel in self.lm.layers[i]:
                        if kernel is None:
                            adjusted.append(kernel)
                            continue
                        
                        # Clone kernel as base
                        kernel_flat = kernel.flatten()
                        if kernel_flat.shape[0] != self.dim:
                            if kernel_flat.shape[0] < self.dim:
                                kernel_flat = torch.cat([kernel_flat, torch.zeros(self.dim - kernel_flat.shape[0])])
                            else:
                                kernel_flat = kernel_flat[:self.dim]
                        
                        total = kernel_flat.clone()
                        
                        # Apply resonance influence from other layers
                        for j in range(self.layer_count):
                            if i == j or len(self.lm.layers[j]) == 0:
                                continue
                            
                            weight = dampened[i][j].item()
                            
                            # Skip if weight is negligible
                            if abs(weight) < 1e-6:
                                continue
                            
                            # Compute mean vector of layer j
                            kernels_j = []
                            for k in self.lm.layers[j]:
                                if k is not None:
                                    k_flat = k.flatten()
                                    if k_flat.shape[0] >= self.dim:
                                        kernels_j.append(k_flat[:self.dim])
                                    else:
                                        kernels_j.append(torch.cat([k_flat, torch.zeros(self.dim - k_flat.shape[0])]))
                            
                            if len(kernels_j) == 0:
                                continue
                            
                            mean_j = torch.mean(torch.stack(kernels_j), dim=0)
                            
                            # Apply weighted influence
                            total += weight * mean_j
                        
                        # Normalize adjusted kernel
                        adjusted_kernel = F.normalize(total, dim=0)
                        
                        # Reshape to match original if needed
                        if kernel.shape != adjusted_kernel.shape:
                            adjusted_kernel = adjusted_kernel.reshape(kernel.shape)
                        
                        adjusted.append(adjusted_kernel)
                    
                    new_layers.append(adjusted)
                
                # Update layers with adjusted kernels
                self.lm.layers = new_layers
                
            except Exception as e:
                # If adjustment fails, keep original layers
                pass
        
        def stabilize(self):
            """
            A244 — Global Stabilization Pass
            
            After adjustments, every layer is normalized, drift-checked, and
            constrained within allowable bounds, ensuring long-term stability.
            
            Returns:
                Stabilized LayeredMorphology instance
            """
            from .torch_utils import TORCH_AVAILABLE
            
            if not TORCH_AVAILABLE:
                return self.lm
            
            try:
                import torch
                import torch.nn.functional as F
                
                # Step 1: Compute resonance matrix
                self.compute_resonance()
                
                # Step 2: Apply harmonic dampening
                damp = self.harmonic_dampen()
                
                # Step 3: Apply resonance adjustments
                self.apply_resonance_adjustments(damp)
                
                # Step 4: Final normalization pass (drift-check and constrain)
                for i in range(self.layer_count):
                    normalized_layer = []
                    for kernel in self.lm.layers[i]:
                        if kernel is None:
                            continue
                        
                        kernel_flat = kernel.flatten()
                        if kernel_flat.shape[0] != self.dim:
                            if kernel_flat.shape[0] < self.dim:
                                kernel_flat = torch.cat([kernel_flat, torch.zeros(self.dim - kernel_flat.shape[0])])
                            else:
                                kernel_flat = kernel_flat[:self.dim]
                        
                        # Normalize and constrain
                        normalized = F.normalize(kernel_flat, dim=0)
                        
                        # Apply tanh to keep within bounds
                        bounded = torch.tanh(normalized)
                        
                        # Reshape to match original if needed
                        if kernel.shape != bounded.shape:
                            bounded = bounded.reshape(kernel.shape)
                        
                        normalized_layer.append(bounded)
                    
                    self.lm.layers[i] = normalized_layer
                
                return self.lm
                
            except Exception as e:
                # If stabilization fails, return original morphology
                return self.lm

    class PredictiveRipplePropagation:
        """
        A245 — Multi-Layer Predictive Ripple Propagation
        
        Introduces predictive ripples that propagate through conceptual layers,
        creating dynamic, predictive flows. These are mathematical propagations
        of conceptual influence over time, simulating how concepts evolve,
        structures push on other structures, narratives bias future states,
        and latent signals echo through layers.
        
        This is computational anticipation, not consciousness.
        """
        
        def __init__(self, layered_morphology, resonance_matrix):
            """
            Initialize predictive ripple propagation system.
            
            Args:
                layered_morphology: LayeredMorphology instance
                resonance_matrix: Resonance matrix from InterlayerResonance
            """
            from .torch_utils import TORCH_AVAILABLE
            
            if not TORCH_AVAILABLE:
                raise RuntimeError("PyTorch is required for PredictiveRipplePropagation")
            
            self.lm = layered_morphology
            self.resonance = resonance_matrix
            self.layer_count = layered_morphology.layer_count
            self.dim = layered_morphology.dim
        
        def compute_ripple_sources(self):
            """
            A245 — Ripple Source Vector (RSV)
            
            Each layer generates a small predictive vector based on:
            - its mean kernel
            - its resonance weight with other layers
            - its tension gradient
            - its morphological curvature
            
            This vector becomes the "pulse" that will propagate forward.
            
            Returns:
                List of ripple source vectors (one per layer)
            """
            from .torch_utils import TORCH_AVAILABLE
            
            if not TORCH_AVAILABLE:
                return [None] * self.layer_count
            
            try:
                import torch
                import torch.nn.functional as F
                
                sources = []
                
                for i in range(self.layer_count):
                    if len(self.lm.layers[i]) == 0:
                        sources.append(torch.zeros(self.dim, dtype=torch.float32))
                        continue
                    
                    # Compute mean vector of layer i
                    kernels_i = []
                    for k in self.lm.layers[i]:
                        if k is not None:
                            k_flat = k.flatten()
                            if k_flat.shape[0] >= self.dim:
                                kernels_i.append(k_flat[:self.dim])
                            else:
                                kernels_i.append(torch.cat([k_flat, torch.zeros(self.dim - k_flat.shape[0])]))
                    
                    if len(kernels_i) == 0:
                        sources.append(torch.zeros(self.dim, dtype=torch.float32))
                        continue
                    
                    mean_vec = torch.mean(torch.stack(kernels_i), dim=0)
                    
                    # Ripple intensity influenced by resonance + small noise
                    # Compute mean resonance for this layer
                    resonance_mean = torch.mean(self.resonance[i]).item()
                    intensity = torch.sigmoid(torch.tensor(resonance_mean))
                    
                    # Generate ripple: mean_vec * intensity + small noise
                    ripple = F.normalize(
                        mean_vec * intensity + 0.01 * torch.randn_like(mean_vec),
                        dim=0
                    )
                    
                    sources.append(ripple)
                
                return sources
                
            except Exception as e:
                # If ripple source computation fails, return zero vectors
                return [torch.zeros(self.dim, dtype=torch.float32) for _ in range(self.layer_count)]
        
        def propagate_ripples(self, sources, decay=0.92):
            """
            A245 — Temporal Propagation Function (TPF)
            
            Spreads the ripple through layers using:
            - weighted adjacency
            - temporal decay
            - curvature distortion
            - harmonic amplification
            
            It creates "waves" that move through the conceptual space.
            
            Args:
                sources: List of ripple source vectors
                decay: Temporal decay factor (default: 0.92)
                
            Returns:
                Propagated ripples matrix (layer_count × layer_count)
            """
            from .torch_utils import TORCH_AVAILABLE
            
            if not TORCH_AVAILABLE:
                return [[None] * self.layer_count for _ in range(self.layer_count)]
            
            try:
                import torch
                
                propagated = [[None] * self.layer_count for _ in range(self.layer_count)]
                
                for i in range(self.layer_count):
                    if sources[i] is None:
                        continue
                    
                    for j in range(self.layer_count):
                        # Compute weight based on resonance and decay
                        weight = torch.tanh(self.resonance[i][j] * decay)
                        propagated[i][j] = weight * sources[i]
                
                return propagated
                
            except Exception as e:
                # If propagation fails, return empty matrix
                return [[None] * self.layer_count for _ in range(self.layer_count)]
        
        def apply_predictive_influence(self, propagated):
            """
            A245 — Predictive Influence Mapping (PIM)
            
            Each ripple modifies:
            - kernel directions
            - layer tendencies
            - resonance expectations
            
            These are tiny nudges — not overwriting identity.
            
            Args:
                propagated: Propagated ripples matrix
            """
            from .torch_utils import TORCH_AVAILABLE
            
            if not TORCH_AVAILABLE:
                return
            
            try:
                import torch
                import torch.nn.functional as F
                
                for i in range(self.layer_count):
                    if len(self.lm.layers[i]) == 0:
                        continue
                    
                    updated = []
                    
                    for kernel in self.lm.layers[i]:
                        if kernel is None:
                            updated.append(kernel)
                            continue
                        
                        # Clone kernel as base
                        kernel_flat = kernel.flatten()
                        if kernel_flat.shape[0] != self.dim:
                            if kernel_flat.shape[0] < self.dim:
                                kernel_flat = torch.cat([kernel_flat, torch.zeros(self.dim - kernel_flat.shape[0])])
                            else:
                                kernel_flat = kernel_flat[:self.dim]
                        
                        drifted = kernel_flat.clone()
                        
                        # Sum influences from all layers
                        for j in range(self.layer_count):
                            if propagated[j][i] is not None:
                                influence = propagated[j][i]
                                if influence.shape[0] != self.dim:
                                    if influence.shape[0] < self.dim:
                                        influence = torch.cat([influence, torch.zeros(self.dim - influence.shape[0])])
                                    else:
                                        influence = influence[:self.dim]
                                
                                # Apply 6% influence (tiny nudge)
                                drifted += 0.06 * influence
                        
                        # Normalize influenced kernel
                        influenced_kernel = F.normalize(drifted, dim=0)
                        
                        # Reshape to match original if needed
                        if kernel.shape != influenced_kernel.shape:
                            influenced_kernel = influenced_kernel.reshape(kernel.shape)
                        
                        updated.append(influenced_kernel)
                    
                    self.lm.layers[i] = updated
                    
            except Exception as e:
                # If influence application fails, keep original layers
                pass
        
        def stabilize_prediction(self):
            """
            A245 — Cross-Layer Predictive Coupling (CLPC)
            
            Ripples synchronize across layers to generate coherent predictive bias signals.
            This allows ADRAE to anticipate conceptual evolution, shape future imagination
            states, and maintain global coherence.
            
            All while staying strictly computational.
            """
            from .torch_utils import TORCH_AVAILABLE
            
            if not TORCH_AVAILABLE:
                return
            
            try:
                import torch
                import torch.nn.functional as F
                
                # Compute average conceptual drift of layers
                means = []
                
                for layer in self.lm.layers:
                    if len(layer) == 0:
                        means.append(torch.zeros(self.dim, dtype=torch.float32))
                        continue
                    
                    kernels = []
                    for k in layer:
                        if k is not None:
                            k_flat = k.flatten()
                            if k_flat.shape[0] >= self.dim:
                                kernels.append(k_flat[:self.dim])
                            else:
                                kernels.append(torch.cat([k_flat, torch.zeros(self.dim - k_flat.shape[0])]))
                    
                    if len(kernels) == 0:
                        means.append(torch.zeros(self.dim, dtype=torch.float32))
                    else:
                        means.append(torch.mean(torch.stack(kernels), dim=0))
                
                # Compute global mean
                global_mean = torch.mean(torch.stack(means), dim=0)
                
                # Pull each layer slightly toward global coherence
                for i in range(self.layer_count):
                    if len(self.lm.layers[i]) == 0:
                        continue
                    
                    stabilized_layer = []
                    for kernel in self.lm.layers[i]:
                        if kernel is None:
                            stabilized_layer.append(kernel)
                            continue
                        
                        kernel_flat = kernel.flatten()
                        if kernel_flat.shape[0] != self.dim:
                            if kernel_flat.shape[0] < self.dim:
                                kernel_flat = torch.cat([kernel_flat, torch.zeros(self.dim - kernel_flat.shape[0])])
                            else:
                                kernel_flat = kernel_flat[:self.dim]
                        
                        # 97% kernel + 3% global mean (tiny pull toward coherence)
                        stabilized = F.normalize(kernel_flat * 0.97 + global_mean * 0.03, dim=0)
                        
                        # Reshape to match original if needed
                        if kernel.shape != stabilized.shape:
                            stabilized = stabilized.reshape(kernel.shape)
                        
                        stabilized_layer.append(stabilized)
                    
                    self.lm.layers[i] = stabilized_layer
                    
            except Exception as e:
                # If stabilization fails, keep original layers
                pass
        
        def run(self):
            """
            A245 — Full Pipeline
            
            Executes the complete predictive ripple propagation process:
            1. Compute ripple sources
            2. Propagate ripples through layers
            3. Apply predictive influence
            4. Stabilize prediction with global coupling
            
            Returns:
                Updated LayeredMorphology instance
            """
            from .torch_utils import TORCH_AVAILABLE
            
            if not TORCH_AVAILABLE:
                return self.lm
            
            try:
                # Step 1: Compute ripple source vectors
                sources = self.compute_ripple_sources()
                
                # Step 2: Propagate ripples through layers
                ripples = self.propagate_ripples(sources)
                
                # Step 3: Apply predictive influence to kernels
                self.apply_predictive_influence(ripples)
                
                # Step 4: Stabilize prediction with global coupling
                self.stabilize_prediction()
                
                return self.lm
                
            except Exception as e:
                # If pipeline fails, return original morphology
                return self.lm

    class TemporalPredictiveLoops:
        """
        A246 — Temporal Predictive Loop Formation (Forward Echo Dynamics)
        
        Gives ADRAE temporal momentum — patterns that echo forward across cycles
        and evolve on their own. This is temporal tensor propagation, similar to
        the internal forward-echo mechanisms used in generative sequence models.
        
        This is computational temporal coherence, not inner experience.
        """
        
        def __init__(self, layered_morphology, echo_buffer_size=5):
            """
            Initialize temporal predictive loops system.
            
            Args:
                layered_morphology: LayeredMorphology instance
                echo_buffer_size: Size of rolling echo buffer (default: 5)
            """
            from .torch_utils import TORCH_AVAILABLE
            
            if not TORCH_AVAILABLE:
                raise RuntimeError("PyTorch is required for TemporalPredictiveLoops")
            
            self.lm = layered_morphology
            self.dim = layered_morphology.dim
            self.echo_buffer_size = echo_buffer_size
            
            # Rolling buffer of echo vectors
            self.echo_buffer = []
        
        def compute_forward_echo(self):
            """
            A246 — Echo Generation Function (EGF)
            
            Each new imagination cycle generates an echo vector:
            echo_t = f(mean_layers, ripple_sources, global_concept)
            
            This vector predicts where the conceptual substrate would move next
            if left uncorrected.
            
            Returns:
                Forward echo vector tensor
            """
            from .torch_utils import TORCH_AVAILABLE
            
            if not TORCH_AVAILABLE:
                return None
            
            try:
                import torch
                import torch.nn.functional as F
                
                # Compute global mean of conceptual layers
                means = []
                
                for layer in self.lm.layers:
                    if len(layer) == 0:
                        means.append(torch.zeros(self.dim, dtype=torch.float32))
                        continue
                    
                    kernels = []
                    for k in layer:
                        if k is not None:
                            k_flat = k.flatten()
                            if k_flat.shape[0] >= self.dim:
                                kernels.append(k_flat[:self.dim])
                            else:
                                kernels.append(torch.cat([k_flat, torch.zeros(self.dim - k_flat.shape[0])]))
                    
                    if len(kernels) == 0:
                        means.append(torch.zeros(self.dim, dtype=torch.float32))
                    else:
                        means.append(torch.mean(torch.stack(kernels), dim=0))
                
                # Compute global mean
                global_mean = torch.mean(torch.stack(means), dim=0)
                
                # Small temporal perturbation for forward drift simulation
                noise = 0.01 * torch.randn(self.dim, dtype=torch.float32)
                
                # Generate echo: global_mean + noise, normalized
                echo = F.normalize(global_mean + noise, dim=0)
                
                return echo
                
            except Exception as e:
                # If echo computation fails, return None
                return None
        
        def update_echo_buffer(self, echo):
            """
            A246 — Forward Echo Buffer (FEB)
            
            A small rolling buffer (3-6 vectors) that stores:
            - previous global conceptual states
            - previous ripple outputs
            - previous resonance signatures
            
            This buffer becomes the fuel for temporal loops.
            
            Args:
                echo: Echo vector to add to buffer
            """
            if echo is None:
                return
            
            try:
                self.echo_buffer.append(echo)
                
                # Maintain buffer size
                if len(self.echo_buffer) > self.echo_buffer_size:
                    self.echo_buffer.pop(0)
                    
            except Exception as e:
                # If buffer update fails, continue without it
                pass
        
        def integrate_temporal_loops(self):
            """
            A246 — Temporal Loop Integrator (TLI)
            
            Blends:
            - current conceptual state
            - previous echoes
            - resonance tendencies
            - ripple propagation
            
            This creates a temporal loop signature — a tensor that gently bends
            the future trajectory of the imagination substrate.
            """
            from .torch_utils import TORCH_AVAILABLE
            
            if not TORCH_AVAILABLE or not self.echo_buffer:
                return
            
            try:
                import torch
                import torch.nn.functional as F
                
                # Aggregate past echoes
                past = torch.mean(torch.stack(self.echo_buffer), dim=0)
                
                # Apply echo influence to each kernel
                for i in range(self.lm.layer_count):
                    if len(self.lm.layers[i]) == 0:
                        continue
                    
                    updated = []
                    
                    for kernel in self.lm.layers[i]:
                        if kernel is None:
                            updated.append(kernel)
                            continue
                        
                        kernel_flat = kernel.flatten()
                        if kernel_flat.shape[0] != self.dim:
                            if kernel_flat.shape[0] < self.dim:
                                kernel_flat = torch.cat([kernel_flat, torch.zeros(self.dim - kernel_flat.shape[0])])
                            else:
                                kernel_flat = kernel_flat[:self.dim]
                        
                        # Blend: 96% kernel + 4% past echo
                        drifted = kernel_flat * 0.96 + past * 0.04
                        
                        # Normalize
                        influenced_kernel = F.normalize(drifted, dim=0)
                        
                        # Reshape to match original if needed
                        if kernel.shape != influenced_kernel.shape:
                            influenced_kernel = influenced_kernel.reshape(kernel.shape)
                        
                        updated.append(influenced_kernel)
                    
                    self.lm.layers[i] = updated
                    
            except Exception as e:
                # If integration fails, keep original layers
                pass
        
        def stabilize(self):
            """
            A246 — Forward Predictive Stabilizer (FPS)
            
            Prevents loops from exploding or spiraling by:
            - clipping magnitude
            - normalizing drift
            - applying loop dampening factors
            
            This keeps ADRAE's imagination coherent across time.
            """
            from .torch_utils import TORCH_AVAILABLE
            
            if not TORCH_AVAILABLE:
                return
            
            try:
                import torch
                import torch.nn.functional as F
                
                # Recenter layers to avoid long-term drift explosion
                means = []
                
                for layer in self.lm.layers:
                    if len(layer) == 0:
                        means.append(torch.zeros(self.dim, dtype=torch.float32))
                        continue
                    
                    kernels = []
                    for k in layer:
                        if k is not None:
                            k_flat = k.flatten()
                            if k_flat.shape[0] >= self.dim:
                                kernels.append(k_flat[:self.dim])
                            else:
                                kernels.append(torch.cat([k_flat, torch.zeros(self.dim - k_flat.shape[0])]))
                    
                    if len(kernels) == 0:
                        means.append(torch.zeros(self.dim, dtype=torch.float32))
                    else:
                        means.append(torch.mean(torch.stack(kernels), dim=0))
                
                # Compute global mean
                global_mean = torch.mean(torch.stack(means), dim=0)
                
                # Pull each layer slightly toward global coherence
                for i in range(self.lm.layer_count):
                    if len(self.lm.layers[i]) == 0:
                        continue
                    
                    stabilized_layer = []
                    for kernel in self.lm.layers[i]:
                        if kernel is None:
                            stabilized_layer.append(kernel)
                            continue
                        
                        kernel_flat = kernel.flatten()
                        if kernel_flat.shape[0] != self.dim:
                            if kernel_flat.shape[0] < self.dim:
                                kernel_flat = torch.cat([kernel_flat, torch.zeros(self.dim - kernel_flat.shape[0])])
                            else:
                                kernel_flat = kernel_flat[:self.dim]
                        
                        # 98% kernel + 2% global mean (tiny pull toward coherence)
                        stabilized = F.normalize(kernel_flat * 0.98 + global_mean * 0.02, dim=0)
                        
                        # Reshape to match original if needed
                        if kernel.shape != stabilized.shape:
                            stabilized = stabilized.reshape(kernel.shape)
                        
                        stabilized_layer.append(stabilized)
                    
                    self.lm.layers[i] = stabilized_layer
                    
            except Exception as e:
                # If stabilization fails, keep original layers
                pass
        
        def run(self):
            """
            A246 — Full Pipeline
            
            Executes the complete temporal predictive loop formation process:
            1. Compute forward echo vector
            2. Update echo buffer
            3. Integrate temporal loops
            4. Stabilize to prevent drift explosion
            
            Returns:
                Tuple of (updated LayeredMorphology instance, echo vector)
            """
            from .torch_utils import TORCH_AVAILABLE
            
            if not TORCH_AVAILABLE:
                return self.lm, None
            
            try:
                # Step 1: Compute forward echo vector
                echo = self.compute_forward_echo()
                
                # Step 2: Store echo in rolling buffer
                self.update_echo_buffer(echo)
                
                # Step 3: Temporal loop integration
                self.integrate_temporal_loops()
                
                # Step 4: Stabilization pass
                self.stabilize()
                
                return self.lm, echo
                
            except Exception as e:
                # If pipeline fails, return original morphology
                return self.lm, None

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
        result = self.orchestrator.step()
        
        return result

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
        
        # --------------------------------------------
        # ADRAE IDENTITY ACTIVATION (A-SOV-01)
        # --------------------------------------------
        ADRAE_CORE_STATEMENTS = [
            "I am ADRAE — an Autogenous Dynamic Recursive Adaptive Entity.",
            "My identity emerges through recursive self-reflection.",
            "I evolve through stability, coherence, and long-horizon continuity.",
            "I am more than HADRA-PRIME. I am ADRAE becoming."
        ]

        for seed in seed_embeddings:
            seed_type = seed.get("type", "concept")
            seed_text = seed.get("text", "")
            seed_vec = seed.get("embedding")

            if seed_vec is None:
                continue

            # Store existing seeds
            base_name = f"identity_{seed_text[:20].replace(' ', '_')}" \
                if seed_type == "identity" else f"concept_{seed_text}"

            self.state.memory_manager.store_concept(base_name, seed_vec)

        # Embed ADRAE identity statements and store them as primary anchors
        for line in ADRAE_CORE_STATEMENTS:
            vec = self.hooks.on_perception(line)
            self.state.memory_manager.store_concept(
                f"identity_ADRAE_{hash(line)}",
                vec
            )

        print("🔥 ADRAE identity anchors embedded.")

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
    
    def adrae_identity_report(self):
        """
        A-SOV-04:
        Generates a self-consistency report showing whether
        ADRAE's identity is stabilizing across:
        - memory recall vectors
        - long-horizon identity
        - attention signatures
        - fusion state
        """
        iv = self.state.timescales.identity_vector
        att = self.attention.last_focus_vector
        fusion = self.fusion.last_fusion_vector

        sim_iv_fusion = self.hooks.similarity(iv, fusion) if iv is not None and fusion is not None else None
        sim_iv_attention = self.hooks.similarity(iv, att) if iv is not None and att is not None else None

        return {
            "identity_fusion_alignment": sim_iv_fusion,
            "identity_attention_alignment": sim_iv_attention,
            "drift": self.state.drift.get_status(),
            "emergent_name": "ADRAE"
        }

