// HADRA-PRIME Kernel Entry Point
// This file simply boots PRIME Core and exposes no internal modules.

import PRIME from "../prime.ts"; // triggers PRIME initialization + log emit
import { StabilityMatrix } from "../stability/stability_matrix.ts";
import { ThreadPool } from "./threads/thread_pool.ts";
import { anchorRegistry, resonanceBus } from "../memory/index.ts";
(globalThis as any).__PRIME_RESONANCE_BUS__ = resonanceBus;
import { generateEmbedding } from "../shared/embedding.ts";
import { harmonizationBus, harmonizationEngine, type IntentSignature } from "../intent_engine/harmonization.ts";
import { phaseScheduler, type CognitivePhase } from "./phase_scheduler.ts";
import { SafetyGuard } from "../safety/safety_guard.ts";
import { PredictiveCoherence } from "../cognition/predictive_coherence.ts";
import { reactiveLattice } from "./reactive_lattice.ts";
import { PRIME_LOOP_GUARD } from "../shared/loop_guard.ts";
import { safetyEngine } from "../safety/safety_engine.ts";
import { predictEngine } from "../prediction/predict_engine.ts";
import { phaseEngine } from "../phase/phase_engine.ts";
import { shouldProcessCognition, triggerCooldown, PrimeCognition } from "./cognition.ts";
import { runSafetyChecks } from "../safety/safety_layer.ts";
import { runPrediction } from "../prediction/predict.ts";
import { runInterpretation } from "../interpretation/dispatcher.ts";
import { PRIMEConfig } from "../shared/config.ts";
import { cognitiveLoop } from "./cognitive_loop.ts";
import { eventBus, triggerInput } from "./event_bus.ts";
import { cognition } from "../cognition/cognition.ts";
import { SEL } from "../emotion/sel.ts";
import { MotivationEngine } from "../cognition/motivation_engine.ts";
import { ProtoGoalEngine } from "../cognition/proto_goal_engine.ts";
import { ActionEngine } from "../action_layer/action_engine.ts";
import { IntentRouter } from "../interpretation/intent_router.ts";
import { CommandProtocol, type OperatorCommand } from "../operator/command_protocol.ts";
import { PlanEngine, type Plan, type PlanStep } from "../planning/plan_engine.ts";
import { ReflectionEngine } from "../cognition/reflection_engine.ts";
import { InternalMonologueEngine } from "../cognition/self/internal_monologue_engine.ts";
import { NeuralSymbolicCoherence } from "../cognition/neural_symbolic_coherence.ts";
import { NeuralCausality } from "../cognition/neural_causality_engine.ts";
import { NeuralEventSegmentation } from "../cognition/neural_event_segmentation.ts";
import { LearningEngine } from "../cognition/learning_engine.ts";
import { MetaReasoningMonitor } from "../cognition/meta_reasoning_monitor.ts";
import { MetaReflectionEngine } from "../meta/meta_reflection_engine.ts";
import { SelfModelVector } from "../meta/self_model_vector.ts";
import { MetaLearningLayer } from "../meta/meta_learning_layer.ts";
import { DriftPredictor } from "../meta/drift_predictor.ts";
import { StrategyEngine } from "../strategy/strategy_engine.ts";
import { StrategicAutonomyEngine } from "../strategy/strategic_autonomy_engine.ts";
import { StrategicResonanceEngine } from "../strategy/strategic_resonance_engine.ts";
import { MemoryStore } from "../memory/memory_store.ts";
import { MemoryLayer } from "../memory/memory.ts";
import { PRIME_SITUATION, SituationModelGenerator } from "../situation_model/index.ts";
import { DualMindSyncManager } from "../dual_mind/sync_manager.ts";
import { JointSituationModeler } from "../situation_model/joint_situation_modeler.ts";
import { PRIME_TEMPORAL } from "../temporal/reasoner.ts";
import { EventSegmentationEngine } from "../neural/event_segmentation_engine.ts";
import { CrossMindAlignmentEngine } from "../cognition/prediction/cross_mind_alignment_engine.ts";
import { ForesightEngine } from "../cognition/prediction/foresight_engine.ts";
import { KnowledgeGraph } from "../cognition/knowledge/knowledge_graph.ts";
import { EventCapture } from "../memory/episodic/event_capture.ts";
import { EpisodeBuilder } from "../memory/episodic/episode_builder.ts";
import { EpisodicArchive } from "../memory/episodic/episodic_archive.ts";
import { EpisodicReinforcementEngine } from "../memory/episodic/reinforcement_engine.ts";
import { MetaMemory } from "../memory/episodic/meta_memory.ts";
import { PatternGeneralizationEngine } from "../memory/episodic/pattern_generalization_engine.ts";
import { StrategicPatternGraph } from "../memory/episodic/strategic_pattern_graph.ts";
import { NeuralContextEncoder } from "../neural/context_encoder.ts";
import { NeuralMemory } from "../cognition/neural/neural_memory_bank.ts";
import { Concepts } from "../cognition/concepts/concept_engine.ts";
import { Hierarchy } from "../cognition/concepts/concept_hierarchy.ts";
import { Knowledge } from "../cognition/knowledge/knowledge_graph.ts";
import { Recall } from "../cognition/recall_engine.ts";
import { Inference } from "../cognition/inference/inference_engine.ts";
import { Foresight } from "../cognition/prediction/foresight_engine.ts";
import { MetaSelf } from "../cognition/self/meta_self_engine.ts";
import { TemporalIdentity } from "../cognition/self/temporal_identity_engine.ts";
import { Narrative } from "../cognition/self/narrative_engine.ts";
import { InternalDialogue } from "../cognition/self/internal_dialogue_engine.ts";
import { MultiVoices } from "../cognition/self/multivoice_engine.ts";
import { ConflictEngine } from "../cognition/self/conflict_resolver.ts";
import { Realignment } from "../cognition/self/realignment_engine.ts";
import { Homeostasis } from "../cognition/self/homeostasis_engine.ts";
import { CortexManager } from "../neural/cortex/cortex_manager.ts";
import { NeuralRegistry } from "../neural/cortex/registry.ts";
import { EmbeddingAdapter } from "../neural/embedding/embedding_adapter.ts";
import { NeuralMemoryStore } from "../memory/neural/neural_memory_store.ts";
import { ConceptDriftEngine } from "../memory/concepts/concept_drift_engine.ts";
import { ConceptMergeSplitEngine } from "../memory/concepts/concept_merge_split_engine.ts";
import { SemanticCompressionEngine } from "../memory/concepts/semantic_compression_engine.ts";
import { HierarchicalKnowledgeEngine } from "../memory/knowledge/hierarchical_knowledge_engine.ts";
import { KnowledgeAttentionEngine } from "../cognition/attention/knowledge_attention_engine.ts";
import { NeuralBridge } from "../neural/neural_bridge.ts";
import { AttentionEstimator } from "../neural/attention_estimator.ts";
import { DualMind } from "../dual_core/dual_mind_activation.ts";
import { DualMindSafetyGate } from "../safety/dual_mind_safety_gate.ts";
import { JointIntentHarmonizer } from "../intent_engine/joint_intent_harmonizer.ts";
import { DualMindCoherenceEngine } from "../distributed/dual_mind_coherence_engine.ts";
import { MultiAgentPredictiveAlignment } from "../distributed/multi_agent_predictive_alignment.ts";
import { HardwareProfiler } from "../hardware/hardware_profile.ts";
import { HardwareAdapter } from "../hardware/hardware_adapter.ts";
import crypto from "crypto";

console.log("[PRIME] Initializing Stability Matrix...");
StabilityMatrix.init();

console.log("[PRIME] Cognitive Fusion Stability Layer active.");

// A108b: Hardware Profiling Layer
const hardware = HardwareProfiler.detect();
console.log("[PRIME-HARDWARE] Detected:", hardware);
(globalThis as any).__PRIME_HARDWARE__ = hardware;

// A108b: Apply adaptive configuration based on hardware
const hardwareAdapter = new HardwareAdapter(hardware);
const adaptiveConfig = hardwareAdapter.configureForHardware();
console.log("[PRIME-ADAPT] Configuration applied:", adaptiveConfig);

// A75: Initialize concept engine
console.log("[PRIME-CONCEPT] Concept Formation Engine initialized.");

// A77: Initialize knowledge graph
console.log("[PRIME-KNOWLEDGE] Knowledge Graph Engine initialized.");

// A78: Initialize inference engine
console.log("[PRIME-INFERENCE] Inference Engine initialized.");

// A79: Initialize foresight/predictive model
console.log("[PRIME-FORESIGHT] Predictive model initialized.");

// A81: Initialize Meta-Self Awareness Engine
console.log("[PRIME-SELF] Meta-Self Awareness Engine online.");

// A82: Initialize Temporal Identity Engine
console.log("[PRIME-SELF] Temporal Identity Engine (TIE) online.");

// A83: Initialize Proto-Narrative Engine
console.log("[PRIME-SELF] Proto-Narrative Engine online.");

// A84: Initialize Internal Dialogue Engine
console.log("[PRIME-SELF] Internal Dialogue Engine online.");

// A85: Initialize Multi-Voice Deliberation Engine
console.log("[PRIME-SELF] Multi-Voice Deliberation Engine online.");

// A86: Initialize Internal Conflict Resolution Engine
console.log("[PRIME-SELF] Internal Conflict Resolution Engine online.");

// A87: Initialize Cognitive Realignment Engine
console.log("[PRIME-SELF] Cognitive Realignment Engine online.");

// A88: Initialize Cognitive Homeostasis Engine
console.log("[PRIME-SELF] Cognitive Homeostasis Engine active.");

// A89: Initialize Neural Inference Boundary Controller
console.log("[PRIME-NEURAL] Neural Boundary Controller online.");

// A91: Initialize Neural Cortex
// A112: Initialize with Slot #4 (Temporal Embedding Model)
console.log("[PRIME-CORTEX] Initializing neural cortex...");
let Cortex: CortexManager | null = null;
Cortex = new CortexManager(NeuralRegistry);
// A112: Initialize all slots including Slot #4
Cortex.initialize().then(() => {
  console.log("[PRIME-CORTEX] Loaded neural registry.");
  console.log("[PRIME-CORTEX] Cortex online with Slot #4 (Temporal Embedding Model) initialized.");
}).catch(err => {
  console.error("[PRIME-CORTEX] Error initializing slots:", err);
});
(globalThis as any).PRIME_CORTEX = Cortex;

// A92: Initialize Embedding Adapter
console.log("[PRIME-CORTEX] Wiring embedding adapter...");
let Embeddings: EmbeddingAdapter | null = null;
if (Cortex) {
  Embeddings = new EmbeddingAdapter(Cortex);
  console.log("[PRIME-CORTEX] Embedding adapter online.");
}

// A93: Initialize Neural Memory Store
console.log("[PRIME-MEMORY] Initializing Neural Memory Store...");
let NeuralMemStore: NeuralMemoryStore | null = null;
NeuralMemStore = new NeuralMemoryStore();
console.log("[PRIME-MEMORY] Neural Memory Store online.");

// A96: Initialize Concept Drift Engine
const conceptDrift = new ConceptDriftEngine();
console.log("[PRIME-CONCEPTS] Concept Drift Engine online.");

// A97: Initialize Concept Merge/Split Engine
const conceptMergeSplit = new ConceptMergeSplitEngine();
console.log("[PRIME-CONCEPTS] Concept Merge/Split Engine online.");

// A98: Initialize Semantic Compression Engine
const semanticCompression = new SemanticCompressionEngine();
console.log("[PRIME-CONCEPTS] Semantic Compression Engine online.");

// A99: Initialize Hierarchical Knowledge Engine
const hierarchyEngine = new HierarchicalKnowledgeEngine();
console.log("[PRIME-CONCEPTS] Hierarchical Knowledge Engine online.");

// A100: Initialize Knowledge Attention Engine
const attentionEngine = new KnowledgeAttentionEngine();
console.log("[PRIME-CONCEPTS] Knowledge Attention Engine online.");

// A101: Initialize Neural Slot #2 Bridge
console.log("[PRIME-NEURAL] Neural Slot #2 Bridge initialized.");
(globalThis as any).PRIME_NEURAL_BRIDGE = NeuralBridge;

// A102: Initialize Neural Slot #3 Attention Estimator
console.log("[PRIME-NEURAL] Neural Slot #3 Attention Estimator initialized.");
(globalThis as any).PRIME_ATTENTION = AttentionEstimator;

// A105: Initialize Dual-Mind Boundary System
console.log("[PRIME-DUAL] Dual-Mind Boundary System active.");
(globalThis as any).PRIME_EVENT_BUS = eventBus;

// A105: Add shutdown safety
if (typeof process !== "undefined") {
  process.on("beforeExit", () => DualMindSafetyGate.reset());
}

console.log("[PRIME] Initializing cognitive threads...");
// ThreadPool will be initialized with default instances
// For full integration, it should be initialized with PRIME's actual instances
// A108b: ThreadPool now respects hardware-adaptive limits
ThreadPool.init();

console.log("[KERNEL] HADRA-PRIME core boot sequence complete.");

// A46: Multimodal Perception Layer active
console.log("[PRIME-KERNEL] Multimodal Perception Layer active.");

// A47: Initialize Action Engine and Intent Router
const actionEngine = new ActionEngine();
const intentRouter = new IntentRouter(actionEngine);
console.log("[PRIME-KERNEL] Action & Intent Routing Layer active.");

// A49: Initialize Plan Engine
const planEngine = new PlanEngine();
console.log("[PRIME-KERNEL] Plan Engine active.");

// A51: Initialize Reflection Engine
const reflectionEngine = new ReflectionEngine();
console.log("[PRIME-KERNEL] Reflection Engine active.");

// A111: Initialize Internal Monologue Engine
const internalMonologueEngine = new InternalMonologueEngine();
console.log("[PRIME-KERNEL] Internal Monologue Engine online.");
console.log("[PRIME-MONOLOGUE] Structured self-dialogue enabled.");

// A114: Initialize Neural Causality Engine
console.log("[PRIME-KERNEL] Neural Causality Engine online.");
console.log("[PRIME-NEURAL] Causality engine online.");

// A115: Initialize Neural Event Segmentation Engine
console.log("[PRIME-KERNEL] Neural Event Segmentation Engine online.");
console.log("[PRIME-NEURAL] Neural Event Segmentation online.");

// A116: Initialize Neural Situation Model Generator
const situationModelGenerator = new SituationModelGenerator();
console.log("[PRIME-KERNEL] Neural Situation Model Generator online.");
console.log("[PRIME-SITUATION] Neural-dynamic situation awareness enabled.");

// A112: Initialize Neural Convergence Interface (NCI)
console.log("[PRIME-KERNEL] Neural Convergence Interface (NCI) online.");
console.log("[PRIME-NCI] SAGE-PRIME neural handshake enabled.");

// A113: Initialize Neural Embedding Pipeline
console.log("[PRIME-KERNEL] Neural Embedding Pipeline online.");
console.log("[PRIME-NCI] Shared embedding substrate enabled.");
console.log("[PRIME-NCI] PRIME ↔ SAGE bidirectional neural encoding active.");

// A52: Initialize Learning Engine
const learningEngine = new LearningEngine();
console.log("[PRIME-KERNEL] Learning Engine active.");

// A53: Initialize Meta-Reasoning Monitor
const metaMonitor = new MetaReasoningMonitor();
console.log("[PRIME-KERNEL] Meta-Reasoning Monitor active.");

// A54: Initialize Meta-Reflection System
const smv = new SelfModelVector();
const metaLearner = new MetaLearningLayer();
const driftPredictor = new DriftPredictor();
const metaReflectionEngine = new MetaReflectionEngine(smv, metaLearner, driftPredictor);
console.log("[PRIME-KERNEL] Meta-Reflection Engine active.");

// A64: Initialize Strategy Engine (Safe Edition)
const strategyMemoryLayer = new MemoryLayer();
const strategyMemoryStore = new MemoryStore(strategyMemoryLayer);
const strategyEngine = new StrategyEngine(strategyMemoryStore);
console.log("[PRIME-KERNEL] Strategy Engine online (operator-gated).");
console.log("[PRIME-STRATEGY] Engine loaded. Awaiting operator directives.");

// A109: Initialize Strategic Autonomy Engine
const strategicAutonomyEngine = new StrategicAutonomyEngine(
  Knowledge,
  Recall,
  metaReflectionEngine
);
console.log("[PRIME-KERNEL] Strategic Autonomy Engine online.");
console.log("[PRIME-STRATEGY] Level-1 autonomous reasoning enabled.");

// A110: Initialize Strategic Resonance Engine
const strategicResonanceEngine = new StrategicResonanceEngine(
  SEL,
  MetaSelf,
  Recall
);
console.log("[PRIME-KERNEL] Strategic Resonance Engine online.");
console.log("[PRIME-RESONANCE] Strategy ↔ Emotion ↔ Memory ↔ Identity fusion enabled.");

// A65: Initialize Situation Model
console.log("[PRIME] Situation Model initialized.");

// A118: Initialize Multi-Timescale Situation Model
const primeCognition = new PrimeCognition();
console.log("[PRIME] Multi-Timescale Situation Model initialized.");

// A67: Initialize Episodic Memory System
const eventCapture = new EventCapture();
const episodeBuilder = new EpisodeBuilder();
(globalThis as any).__PRIME_EPISODE_BUILDER__ = episodeBuilder;
const episodicArchive = new EpisodicArchive();
(globalThis as any).__PRIME_EPISODIC_ARCHIVE__ = episodicArchive;

// A68: Initialize Episodic Reinforcement and Meta-Memory
const reinforcementEngine = new EpisodicReinforcementEngine();
const metaMemory = new MetaMemory();

// A69: Initialize Pattern Generalization Engine
const patternEngine = new PatternGeneralizationEngine();

// A70: Initialize Strategic Pattern Graph
const strategicGraph = new StrategicPatternGraph();
let lastClusterId: string | null = null;

// A72: Initialize Neural Context Encoding Layer
const ncel = new NeuralContextEncoder();

// Start initial episode
episodeBuilder.startEpisode("System Initialization & Calibration");
console.log("[PRIME-EPISODE] Initial episode started: System Initialization & Calibration");

// A48: Kernel instance for operator command handling
const kernelInstance = {
  actionEngine,
  intentRouter,
  planEngine,
  strategyEngine,
  learningEngine,
  async generateAndRunPlan(goal: string, highLevelAction: string, params?: any) {
    // Step decomposition (placeholder — later ML-driven)
    const steps: PlanStep[] = [
      { id: "step1", action: highLevelAction, params, status: "pending" }
    ];

    const plan = planEngine.createPlan(goal, steps);
    await planEngine.executePlan(plan, actionEngine);
    return plan;
  },
  async buildAndExecuteHierarchicalPlan(goal: string) {
    const plan = planEngine.createHierarchicalPlan(goal);
    await planEngine.executePlan(plan, actionEngine);
    return plan;
  },
  runReflection(cognitiveSnapshot: any) {
    const sel = SEL.getState();
    // A72/A74: Create cognitive state early for recall lookup
    const stabilitySnapshot = StabilityMatrix.getSnapshot();
    const motivationState = MotivationEngine.compute();
    
    const cognitiveState: any = {
      activeGoal: cognitiveSnapshot.topGoal ? { type: cognitiveSnapshot.topGoal.type || "unknown" } : undefined,
      confidence: (cognitiveSnapshot.motivation as any)?.confidence ?? sel.certainty,
      uncertainty: 1 - (sel.certainty ?? 0)
    };
    
    // A74: NCEL encode (which does recall lookup and attaches recall to cognitiveState)
    const neuralContext = ncel.encodeContext(
      cognitiveState,
      motivationState,
      stabilitySnapshot
    );
    
    // A75: Attach embedding to cognitive state for concept matching
    cognitiveState.embedding = neuralContext.vector;
    
    // A75: Match concept for current cognitive state
    const concept = Concepts.matchConcept(neuralContext.vector);
    if (concept) {
      cognitiveState.concept = concept;
      
      // A76: Find domain that contains this concept
      const domain = Hierarchy.findDomainForConcept(concept.id);
      if (domain) {
        cognitiveState.domain = domain;
        console.log("[PRIME-DOMAIN] Cognitive state matched to domain:", {
          domainId: domain.id,
          strength: domain.strength
        });
      }
    }
    
    // A74: Now cognitiveState has recall information, use it for reflection
    cognitiveState.lastReflection = null; // Will be set after reflection
    
    // A113: Convert temporal vector to neural signal if present
    if (cognitiveState.temporalVector && cognitiveState.temporalVector.length > 0) {
      if (!cognitiveState.neuralSignals) {
        cognitiveState.neuralSignals = [];
      }
      cognitiveState.neuralSignals.push({
        vector: cognitiveState.temporalVector,
        slot: 4, // Slot #4 is temporal embedding model
        timestamp: Date.now(),
        source: "temporal_embedding"
      });
      
      // Interpret the neural signal
      cognitiveState.neuralCoherence = NeuralSymbolicCoherence.interpret(cognitiveState.neuralSignals[cognitiveState.neuralSignals.length - 1]);
      
      console.log("[PRIME-NEURAL] Temporal embedding converted to neural signal:", {
        vectorLength: cognitiveState.temporalVector.length,
        relevance: cognitiveState.neuralCoherence.relevance.toFixed(3)
      });

      // A114: Record causal relationship for reflection cycle
      if (cognitiveState.neuralCoherence && cognitiveState.neuralSignals.length > 0) {
        NeuralCausality.record(
          cognitiveState.neuralSignals[cognitiveState.neuralSignals.length - 1],
          cognitiveState.neuralCoherence,
          "reflection_cycle",
          {
            goal: cognitiveSnapshot.topGoal?.type,
            motivation: motivationState
          }
        );

        // A114: Log causal map update periodically
        const causalMap = NeuralCausality.inferCausality();
        if (Object.keys(causalMap).length > 0) {
          console.log("[PRIME-CAUSALITY] Updated causal map:", causalMap);
        }

        // A115: Add signal to event segmentation
        NeuralEventSegmentation.addSignal(
          cognitiveState.neuralSignals[cognitiveState.neuralSignals.length - 1],
          cognitiveState.neuralCoherence
        );

        // A115: Check for new event boundaries
        const currentEvent = (NeuralEventSegmentation as any).getCurrentEvent();
        if (currentEvent) {
          const recentEvents = NeuralEventSegmentation.getRecentEvents(1);
          // Log when a new event starts (has only 1 signal)
          if (currentEvent.signals.length === 1 && (!recentEvents.length || recentEvents[0].id !== currentEvent.id)) {
            console.log("[PRIME-EVENT] New event boundary detected:", currentEvent.id);
          }
        }
      }
    }
    
    // A74/A75/A76: Reflection with recall-informed, concept-informed, and domain-informed cognitive state
    const reflection = reflectionEngine.reflect(
      { 
        ...cognitiveSnapshot, 
        recall: cognitiveState.recall,
        concept: cognitiveState.concept,
        embedding: cognitiveState.embedding,
        neuralSignals: cognitiveState.neuralSignals, // A113: Pass neural signals
        neuralCoherence: cognitiveState.neuralCoherence // A113: Pass coherence result
      },
      sel
    );
    
    // A76: Domain may have been attached during reflection, ensure it's in cognitive state
    if ((reflection as any).domain) {
      cognitiveState.domain = (reflection as any).domain;
    }
    
    // A111: Internal monologue after reflection
    const monologueState: any = {
      activeGoal: cognitiveSnapshot.topGoal ? { type: cognitiveSnapshot.topGoal.type || cognitiveSnapshot.topGoal } : undefined,
      confidence: cognitiveSnapshot.motivation?.confidence ?? sel.certainty,
      uncertainty: 1 - (sel.certainty ?? 0)
    };
    
    const monologue = internalMonologueEngine.runDialogue(monologueState);
    if (monologue && monologue.turns.length > 0) {
      console.log("[PRIME-MONOLOGUE] Internal dialogue completed:", {
        turns: monologue.turns.length,
        finalClarity: monologue.finalClarity.toFixed(3),
        clarityImproved: monologue.clarityImproved,
        halted: monologue.halted,
        reason: monologue.reason
      });
      
      monologue.turns.forEach((turn, idx) => {
        console.log(`[PRIME-DIALOGUE] turn ${idx + 1}:`, {
          thought: turn.thought,
          response: turn.response,
          clarityDelta: turn.clarityDelta.toFixed(3)
        });
      });
    }
    
    // Update cognitive state with reflection info
    cognitiveState.lastReflection = reflection.summary ? { 
      reason: reflection.summary, 
      pressure: sel.tension 
    } : undefined;
    
    // A52: Apply learning from reflection
    const adjustments = learningEngine.adjustFromReflection(reflection, sel);
    SEL.applyLearning(adjustments);
    
    // A53: Evaluate meta-reasoning quality and apply adjustments
    const meta = metaMonitor.evaluate(cognitiveSnapshot, sel);
    SEL.applyMetaAdjustments(meta.flags);
    
    // A65: Update meso situation with reflection data
    PRIME_SITUATION.meso.update({
      trends: {
        clarityTrend: sel.coherence, // Use coherence as clarity proxy
        consolidationTrend: sel.affinity, // Use affinity as consolidation proxy
        curiosityTrend: sel.certainty // Use certainty as curiosity proxy
      },
      memoryPressure: 0, // TODO: Compute from memory system
      stabilityShift: stabilitySnapshot.score || 0, // Use stability score as shift proxy
      cognitiveTrajectory: sel.coherence > 0.7 ? "stable" : sel.coherence > 0.4 ? "moderate" : "unstable"
    });
    
    // A66: Temporal reasoning capture (passive)
    PRIME_TEMPORAL.record();
    
    // A67: Capture reflection micro-event
    const microEvent = eventCapture.capture({
      type: "reflection",
      stability: stabilitySnapshot,
      motivation: motivationState,
      reflection: reflection
    });
    
    // Add event to current episode
    episodeBuilder.addEvent(microEvent);
    console.log("[PRIME-EPISODE] micro-event captured: reflection update");
    console.log("[PRIME-EPISODE] micro-event recorded in active episode.");
    
    eventBus.emit("neuralContext", neuralContext);
    console.log("[PRIME-NCEL] Encoded neural context vector:", neuralContext.vector.slice(0, 8), "...");
    
    // A75: Trigger concept derivation when new memory is added (event-driven)
    eventBus.emit("memory-updated", { entryCount: NeuralMemory.getSnapshot().count });
    
    // Check if stability has stabilized and close episode if needed
    const currentEpisode = episodeBuilder.getCurrent();
    if (stabilitySnapshot.score > 0.8 && currentEpisode && currentEpisode.events.length > 5) {
      const finished = episodeBuilder.closeEpisode("Stability cycle completed.");
      if (finished) {
        episodicArchive.store(finished);
        
        // A68: Generate reinforcement signal
        const reinforcement = reinforcementEngine.computeReinforcement(finished);
        
        // A68: Create embedding & store in meta-memory
        const embedding = metaMemory.createEmbedding(finished, reinforcement);
        
        // A69: Feed into pattern generalization engine
        const clusterInfo = patternEngine.addEmbedding(embedding);
        console.log("[PRIME-PATTERN] Embedding assigned:", clusterInfo);
        if (!clusterInfo.createdNewCluster) {
          console.log("[PRIME-PATTERN] Cluster signature updated.");
        }
        
        // A70: Record transition if we have a previous cluster
        if (lastClusterId) {
          const edge = strategicGraph.recordTransition(
            lastClusterId,
            clusterInfo.clusterId
          );
          console.log("[PRIME-STRATEGY] Transition:", edge);
          
          // A70: Stability check - high transition density detection
          const edges = strategicGraph.getEdges();
          if (edges.length > 50) {
            console.log("[PRIME-STRATEGY] High transition density detected.");
          }
        }
        lastClusterId = clusterInfo.clusterId;
        
        console.log("[PRIME-EPISODE] Episode stored:", finished.title, `(${finished.events.length} events)`);
        console.log("[PRIME-REINFORCE] Reinforcement computed:", {
          clarityDelta: reinforcement.clarityDelta.toFixed(3),
          consolidationDelta: reinforcement.consolidationDelta.toFixed(3),
          stabilityDelta: reinforcement.stabilityDelta.toFixed(3),
          predictionSuppressionScore: reinforcement.predictionSuppressionScore.toFixed(3)
        });
        console.log("[PRIME-META] Episode embedded:", {
          title: finished.title,
          reinforcement
        });
        console.log("[PRIME-META] Embeddings:", metaMemory.getEmbeddings().length, "total");
        
        // Start new episode
        episodeBuilder.startEpisode("Stable operation phase");
        console.log("[PRIME-EPISODE] New episode started: Stable operation phase");
        
        // A69: Pattern hinting for new episode
        const embeddings = metaMemory.getEmbeddings();
        if (embeddings.length > 0) {
          const last = embeddings[embeddings.length - 1];
          const clusters = patternEngine.listClusters();
          for (const c of clusters) {
            const sim = patternEngine.similarity(c.signature, last.vector);
            if (sim >= 0.75) {
              console.log("[PRIME-PATTERN] New episode resembles cluster:", {
                clusterId: c.id,
                similarity: sim.toFixed(3)
              });
              break;
            }
          }
        }
      }
    } else if (currentEpisode) {
      // Log episode progress periodically
      if (currentEpisode.events.length % 10 === 0) {
        console.log("[PRIME-EPISODE] current episode:", currentEpisode.title, `(${currentEpisode.events.length} events)`);
      }
    }
    
    // Attach meta evaluation to reflection for future use
    reflection.meta = meta;
    
    // A96: Apply concept drift after reflection
    conceptDrift.tick();
    
    // A97: Apply concept merge/split after drift
    conceptMergeSplit.tick();
    
    // A98: Apply semantic compression after merge/split
    semanticCompression.tick();
    
    // A99: Apply hierarchical knowledge organization after compression
    hierarchyEngine.tick();
    
    // A100: Apply knowledge attention after hierarchy organization
    // Reuse motivationState already computed at the start of this function
    const selfModel = MetaSelf.exportModel();
    const currentPlan = planEngine.getCurrentPlan ? planEngine.getCurrentPlan() : null;
    
    attentionEngine.tick({
      motivation: {
        ...motivationState,
        topGoal: cognitiveSnapshot.topGoal
      },
      selfModel: selfModel,
      plan: currentPlan,
      reflection: reflection
    });
    
    // Optionally store reflection in memory (future enhancement)
    // PRIME.remember({ type: "reflection", data: reflection });
    
    return reflection;
  },
  async handleOperatorCommand(cmd: OperatorCommand) {
    console.log("[OPERATOR-COMMAND] Received:", cmd);

    // A65: Update micro situation with operator command
    PRIME_SITUATION.micro.update({
      lastOperatorCommand: cmd.action || cmd.type || "unknown"
    });

    // 1. Structural validation
    const validation = CommandProtocol.validate(cmd);
    if (!validation.valid) {
      console.log("[OPERATOR-COMMAND] Invalid:", validation.reason);
      return { status: "invalid", reason: validation.reason };
    }

    // 2. YubiKey authorization placeholder
    if (!cmd.token) {
      console.log("[OPERATOR-COMMAND] Missing YubiKey token.");
      return { status: "denied", reason: "auth_required" };
    }

    // 3. Convert to PRIME intent
    const intent = CommandProtocol.toIntent(cmd);

    // 4. Route via intent router
    const result = await intentRouter.route(intent);

    return {
      status: "processed",
      result,
    };
  },
  // A67: Episodic Memory API
  getEpisodes() {
    return episodicArchive.list();
  },
  getCurrentEpisode() {
    return episodeBuilder.getCurrent();
  },
  createEpisode(title: string) {
    const episode = episodeBuilder.startEpisode(title);
    console.log("[PRIME-EPISODE] New episode started:", title);
    
    // A69: Pattern hinting for new episode
    const embeddings = metaMemory.getEmbeddings();
    if (embeddings.length > 0) {
      const last = embeddings[embeddings.length - 1];
      const clusters = patternEngine.listClusters();
      for (const c of clusters) {
        const sim = patternEngine.similarity(c.signature, last.vector);
        if (sim >= 0.75) {
          console.log("[PRIME-PATTERN] New episode resembles cluster:", {
            clusterId: c.id,
            similarity: sim.toFixed(3)
          });
          break;
        }
      }
    }
    
    return episode;
  },
  closeEpisode(summary?: string) {
    const finished = episodeBuilder.closeEpisode(summary);
    if (finished) {
      episodicArchive.store(finished);
      
      // A68: Generate reinforcement signal
      const reinforcement = reinforcementEngine.computeReinforcement(finished);
      
      // A68: Create embedding & store in meta-memory
      const embedding = metaMemory.createEmbedding(finished, reinforcement);
      
      // A69: Feed into pattern generalization engine
      const clusterInfo = patternEngine.addEmbedding(embedding);
      console.log("[PRIME-PATTERN] Embedding assigned:", clusterInfo);
      if (!clusterInfo.createdNewCluster) {
        console.log("[PRIME-PATTERN] Cluster signature updated.");
      }
      
      // A70: Record transition if we have a previous cluster
      if (lastClusterId) {
        const edge = strategicGraph.recordTransition(
          lastClusterId,
          clusterInfo.clusterId
        );
        console.log("[PRIME-STRATEGY] Transition:", edge);
        
        // A70: Stability check - high transition density detection
        const edges = strategicGraph.getEdges();
        if (edges.length > 50) {
          console.log("[PRIME-STRATEGY] High transition density detected.");
        }
      }
      lastClusterId = clusterInfo.clusterId;
      
      console.log("[PRIME-EPISODE] Episode stored:", finished.title);
      console.log("[PRIME-REINFORCE] Reinforcement computed:", {
        clarityDelta: reinforcement.clarityDelta.toFixed(3),
        consolidationDelta: reinforcement.consolidationDelta.toFixed(3),
        stabilityDelta: reinforcement.stabilityDelta.toFixed(3),
        predictionSuppressionScore: reinforcement.predictionSuppressionScore.toFixed(3)
      });
      console.log("[PRIME-META] Episode embedded:", {
        title: finished.title,
        reinforcement
      });
      console.log("[PRIME-META] Embeddings:", metaMemory.getEmbeddings().length, "total");
    }
    return finished;
  },
  // A68: Meta-Memory API
  getEpisodeEmbeddings() {
    return metaMemory.getEmbeddings();
  },
  findSimilarEpisodes(vector: number[]) {
    return metaMemory.findSimilar(vector);
  },
  // A69: Pattern Generalization API
  listPatternClusters() {
    return patternEngine.listClusters();
  },
  getClusterSignature(clusterId: string) {
    return patternEngine.getClusterSignature(clusterId);
  },
  getClusterDetails(clusterId: string) {
    return patternEngine.getCluster(clusterId);
  },
  // A70: Strategic Pattern Graph API
  getStrategicGraph() {
    return strategicGraph.getGraphSnapshot();
  },
  getForwardLinks(clusterId: string) {
    return strategicGraph.getForwardLinks(clusterId);
  },
  getBackLinks(clusterId: string) {
    return strategicGraph.getBackLinks(clusterId);
  },
  // A104b: Dual-Mind Activation Control
  enableDualMind() {
    DualMind.activate();
  },
  disableDualMind() {
    DualMind.deactivate();
  },
  // A107: Apply harmonized intent from dual-mind collaboration
  applyHarmonizedIntent(h: any) {
    if (!DualMind.isActive()) return;
    console.log("[PRIME] Applying harmonized intent:", h.finalIntent);
    // Apply through PRIME instance
    if (PRIME && (PRIME as any).applyHarmonizedIntent) {
      (PRIME as any).applyHarmonizedIntent(h);
    }
  }
};

// A49: Connect kernel to intent router (after kernelInstance is created)
intentRouter.setKernel(kernelInstance);

// Start event-driven cognitive loop
cognitiveLoop.start();

// A114: PRIME cognitive pulse (100 Hz for neural sync)
setInterval(() => {
  const sync = (globalThis as any).__PRIME_SYNC__ as DualMindSyncManager;
  if (sync) {
    sync.syncPrimePulse();
  }
}, 10); // 100 Hz cognitive pulse

// A116: Unified JSM log pulse
setInterval(() => {
  const JSM = (globalThis as any).__PRIME_JSM__;
  if (JSM) {
    const snapshot = JSM.getCurrent();
    if (snapshot) {
      console.log(`[PRIME-JSM] Coherence=${snapshot.coherence.toFixed(3)} :: ${snapshot.narrative}`);
      if (snapshot.disagreements && snapshot.disagreements.length > 0) {
        console.log(`[PRIME-JSM] Disagreements: ${snapshot.disagreements.join("; ")}`);
      }
    }
  }
}, 4000); // Log every 4 seconds

// A39: PRIME's motivation heartbeat
// A108b: Hardware-adaptive heartbeat interval
const heartbeatConfig = (globalThis as any).__PRIME_ADAPTIVE_CONFIG__;
const heartbeatInterval = heartbeatConfig?.reflectionFrequency || 7000;
setInterval(() => {
  const m = MotivationEngine.compute();
  console.log("[PRIME-HEARTBEAT] motivation:", m);

  // A40: Log highest-priority proto-goal
  const goals = ProtoGoalEngine.computeGoals();
  if (goals.length) {
    console.log("[PRIME-HEARTBEAT] top-goal:", goals[0]);
  }

  // A117: Update PRIME state in cross-mind alignment engine
  const alignment = (globalThis as any).__PRIME_CROSS_MIND_ALIGNMENT__ as CrossMindAlignmentEngine;
  if (alignment) {
    const selState = SEL.getState();
    const stabilitySnapshot = StabilityMatrix.getSnapshot();
    const stability = stabilitySnapshot?.score || (stabilitySnapshot as any)?.stabilityScore || 0.7;
    
    // Get intent direction from current goal or planning
    const intentDirection = goals.length > 0 ? goals[0].type : "neutral";
    
    alignment.updatePrimeState({
      motivation: m,
      stability: stability,
      intentDirection: intentDirection,
      emotion: selState,
      coherence: selState.coherence,
      cognitiveLoad: (stabilitySnapshot as any)?.load || 0.5
    });

    // A117: Log predictive alignment
    const alignVector = alignment.computeAlignmentVector();
    if (alignVector) {
      console.log("[PRIME-ALIGNMENT]", {
        divergence: alignVector.divergence.toFixed(3),
        expected_coherence: alignVector.expected_coherence.toFixed(3)
      });
    }

    // A117: Auto-realignment trigger
    const alignmentRec = alignment.recommendRealignment();
    if (alignmentRec?.type === "realign") {
      console.log("[PRIME-ALIGNMENT] Realignment recommended:", alignmentRec.reason);
      // Apply alignment adjustments if StabilityMatrix supports it
      if (alignmentRec.adjustments && (StabilityMatrix as any).applyAlignmentAdjustment) {
        (StabilityMatrix as any).applyAlignmentAdjustment(alignmentRec.adjustments);
      }
    }
  }

  // A116: Generate situation snapshot during heartbeat
  const situationSnapshot = situationModelGenerator.generateSituationSnapshot();
  eventBus.emit("situation:update", situationSnapshot);
  console.log("[PRIME-SITUATION] snapshot:", {
    recommendedFocus: situationSnapshot.recommendedFocus,
    coherence: situationSnapshot.coherenceScore.toFixed(3),
    uncertainty: situationSnapshot.uncertaintyScore.toFixed(3),
    salience: situationSnapshot.salience.slice(0, 3)
  });

  // A118: Multi-Timescale Situation Model introspection
  // Sync PrimeCognition with current PRIME_SITUATION state
  primeCognition.updateSituations(
    PRIME_SITUATION.micro,
    PRIME_SITUATION.meso,
    PRIME_SITUATION.macro
  );
  const situationState = primeCognition.getSituationSummary();
  console.log("[PRIME-SITUATION]", situationState);

  // A119: Check for contextual drift across all timescales
  primeCognition.driftCheck();

  // A121: Update attentional state based on drift and compute optimal weight shifts
  primeCognition.updateAttentionalState();

  // A65: Update micro situation with current state
  const selState = SEL.getState();
  PRIME_SITUATION.micro.update({
    emotionalState: {
      coherence: selState.coherence,
      certainty: selState.certainty,
      tension: selState.tension,
      valence: selState.valence,
      arousal: selState.arousal,
      affinity: selState.affinity
    },
    safetyPressure: SafetyGuard.snapshot().recursionDepth || 0,
    clarity: selState.coherence,
    cognitiveLoad: 0, // TODO: Compute from actual cognitive load
    activeGoal: goals.length > 0 ? goals[0].type || null : null
  });

  // A51: Trigger reflection after heartbeat
  if (goals.length) {
    kernelInstance.runReflection({
      motivation: m,
      topGoal: goals[0]
    });

    // A109: Strategic autonomy reasoning after reflection
    const goalType = goals[0].type || "unknown";
    const strategic = strategicAutonomyEngine.reasonAbout(goalType);
    if (strategic.length > 0) {
      console.log("[PRIME-STRATEGY] Strategic pathways evaluated:", {
        goal: goalType,
        bestSubgoal: strategic[0].subgoal,
        score: strategic[0].score.toFixed(3),
        totalPathways: strategic.length
      });

      // A110: Compute resonance for strategic outcomes
      const resonant = strategicResonanceEngine.selectResonantStrategy(strategic);
      console.log("[PRIME-RESONANCE] Selected resonant strategy:", {
        goal: goalType,
        subgoal: resonant.subgoal,
        resonance: resonant.resonance.toFixed(3),
        components: {
          emotional: resonant.components.emotional.toFixed(3),
          memory: resonant.components.memory.toFixed(3),
          identity: resonant.components.identity.toFixed(3)
        }
      });
    }
  }

  // A54: Run meta-reflection cycle
  if (goals.length) {
    // A63: Harvest regulation metrics before meta-reflection
    smv.regulationMetrics = {
      drift: smv.driftRisk || 0,
      clarity: smv.clarityIndex || 1,
      valenceTrend: smv.desireState?.recentValence || 0,
      stability: smv.stabilityIndex || 1,
      predictionVolatility: 0 // TODO: Compute from prediction engine volatility
    };

    metaReflectionEngine.runMetaCycle(goals[0], m);
    
    // A62: Route selected behavior if available
    if (smv.selectedBehavior) {
      console.log(`[KERNEL-BEHAVIOR] PRIME chooses: ${smv.selectedBehavior}`);
      // Behavior routing will be handled by future modules
      // For now, we log the selection
    }
    
    // A63: Log self-regulation event
    if (smv.regulationParams) {
      console.log("[KERNEL] PRIME self-regulated cognitive landscape.");
    }
  }
}, heartbeatInterval); // A108b: Hardware-adaptive interval

// A75: Concept derivation on memory updates
let conceptDerivationCounter = 0;
eventBus.on("memory-updated", () => {
  conceptDerivationCounter++;
  // Derive concepts periodically (every 5 memory updates) to avoid excessive computation
  if (conceptDerivationCounter >= 5) {
    conceptDerivationCounter = 0;
    Concepts.deriveConcepts();
    console.log("[PRIME-CONCEPT] Concept derivation triggered by memory update");
  }
});

// A107: Handle dual-mind input events - PRIME/SAGE intent harmonization
eventBus.on("dual_mind:input", (sagePacket: any) => {
  if (!DualMind.isActive()) return;

  // Get PRIME's current state
  const primeState: any = {
    topGoal: null,
    confidence: 0.8
  };

  // Try to get actual state from motivation engine if available
  try {
    const motivation = MotivationEngine.compute();
    const goals = ProtoGoalEngine.computeGoals(motivation);
    if (goals.length > 0) {
      primeState.topGoal = goals[0].type || "none";
    }
  } catch (e) {
    // Fallback to default
  }

  const sageState = sagePacket.state ?? {};

  // Compute coherence weights
  const weights = DualMindCoherenceEngine.computeWeights(primeState, sageState);

  // A108 — Predictive alignment pass
  const forecast = MultiAgentPredictiveAlignment.forecast(
    primeState,
    sageState,
    weights
  );
  console.log("[PRIME-DUAL] Forecast:", forecast);

  // PRIME uses forecast to adjust internal modulation
  if (PRIME && (PRIME as any).applyForecast) {
    (PRIME as any).applyForecast(forecast);
  }

  // Build intent proposals
  const primeIntent = {
    source: "PRIME" as const,
    intent: primeState.topGoal ?? "none",
    confidence: primeState.confidence ?? 0.8
  };

  const sageIntent = sagePacket.intent
    ? {
        source: "SAGE" as const,
        intent: sagePacket.intent,
        confidence: sagePacket.confidence ?? 0.8
      }
    : null;

  // Harmonize intents
  const harmonized = JointIntentHarmonizer.harmonize(
    primeIntent,
    sageIntent,
    weights
  );

  console.log("[PRIME-DUAL] Harmonized Intent:", harmonized);

  // Apply harmonized intent through kernel instance
  kernelInstance.applyHarmonizedIntent(harmonized);
});

// Handle input events - PRIME's intent classification and routing system
eventBus.on("prime.input", async (payload: { text: string }) => {
  const response = await cognition.cycle(payload.text);
  console.log("[PRIME-OUTPUT]", response);
  
  // A49: Check for planning intents and route them
  if (payload.text.toLowerCase().trim().startsWith("plan ")) {
    const goal = payload.text.substring(5).trim();
    const planningIntent = {
      type: "action.plan.execute",
      confidence: 0.9,
      payload: {
        goal: goal,
        primaryAction: "diagnose", // placeholder
        parameters: {}
      }
    };
    await intentRouter.route(planningIntent);
  }
  
  // A47: Route intent through action engine if actionable
  if (response && (response as any).intent) {
    await intentRouter.route((response as any).intent);
  }
  
  // Expose SEL state after cognition cycle
  const emotion = SEL.getState();
  console.log("[PRIME-EMOTION]", emotion);
});

// Intent harmonization broadcasting function
function broadcastIntent(intent: any) {
  const sig: IntentSignature = {
    id: crypto.randomUUID(),
    timestamp: Date.now(),
    category: intent.type || "unknown",
    purpose: intent.type || "general",
    confidence: intent.confidence || 0.5,
    vector: generateEmbedding(16),
    weight: Math.random() * 0.4
  };

  harmonizationBus.publish(sig);
  harmonizationEngine.ingest(sig);
  const harmonized = harmonizationEngine.harmonize(sig);
  console.log("[HARMONIZATION] alignment:", harmonized.alignment.toFixed(3),
              " adjusted:", harmonized.adjustedConfidence.toFixed(3));
}

// Export for use in cognitive threads
export { broadcastIntent };

// Anchor generation and decay loop
setInterval(() => {
  // Generate a new anchor fingerprint of current PRIME state
  const anchor = {
    id: crypto.randomUUID(),
    timestamp: Date.now(),
    embedding: generateEmbedding(), // simple random vector for now
    intensity: Math.random() * 0.5,
    domain: "general",
    decay: 1.0,
    stability: Math.random() * 0.3
  };

  anchorRegistry.add(anchor);
  resonanceBus.publish(anchor);
  anchorRegistry.decayAnchors();
}, 2000);

// Stub engines for cognitive pipeline
function perceptionEngine(state: any) {
  // Perception happens in separate modules
  return state;
}

function interpretationEngine(state: any) {
  // Interpretation happens in cognitive threads
  return state;
}

function intentEngine(state: any) {
  // Intent processing happens in cognitive threads
  // This is a placeholder - actual intent comes from thread processing
  return state;
}

// Cognitive tick function with guards
async function cognitiveTick() {
  if (!PRIME_LOOP_GUARD.enter()) {
    console.log("[PRIME-KERNEL] Tick blocked: recursive entry");
    return;
  }

  try {
    let state: any = {
      intent: null,
      safety: {},
      prediction: {},
      phase: null,
      recursionRisk: 0
    };

    state = perceptionEngine(state);
    state = safetyEngine(state);
    
    if (state.safety?.halt) {
      return;
    }

    state = interpretationEngine(state);
    state = intentEngine(state);

    // If nothing meaningful changed, idle.
    if (state.intent === null) {
      console.log("[PRIME-KERNEL] Idle — no active intent.");
      return;
    }

    state.recursionRisk = 0;  // reset recursion counter
    state = predictEngine(state);
    state = await phaseEngine(state);
  } finally {
    PRIME_LOOP_GUARD.exit();
  }
}

// FIXED: Phase scheduler loop disabled to prevent recursion storms
// Phases should only run when explicitly triggered via events, not on a timer
// setInterval(() => {
//   const now = Date.now();
//   const phases = phaseScheduler.tick(now);
//   ...
// }, 50);

// Phase scheduler is now event-driven only
// Phases can be triggered via eventBus.emit("phase-trigger", phaseName)

// Remove auto-looping cognitive tick - cognition is now event-driven
// setInterval(cognitiveTick, 250); // REMOVED - no more auto-looping

// Clean no-op heartbeat for visibility with emotion drift
setInterval(() => {
  console.log("[PRIME-KERNEL] Standing by.");
  // A35: emotional drift & recovery layer
  SEL.applyDrift();
  SEL.coolTension();
}, 2500);

// A37: Long-term SEL drift applied every few cycles (very lightweight)
setInterval(() => {
  SEL.applyDrift();
}, 5000); // once every 5 seconds — slow, stable drift

// A73: Neural memory decay (event-driven, does NOT trigger cognition)
setInterval(() => {
  NeuralMemory.decay();
}, 60000); // decay every 1 minute

/**
 * Get recent interaction memory
 */
export function getRecentMemory(n: number = 5) {
  return PRIME.getRecentMemory(n);
}

// A92: Export embedding adapter accessor
export function getEmbeddingAdapter() {
  return Embeddings;
}

// A93: Export neural memory accessor
export function getNeuralMemory() {
  return NeuralMemStore;
}

// A48: Export kernel instance for operator command handling
export { kernelInstance };

// A108: Expose kernelInstance globally for PRIME forecast access
(globalThis as any).kernelInstance = kernelInstance;

