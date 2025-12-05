// HADRA-PRIME Kernel Entry Point
// This file simply boots PRIME Core and exposes no internal modules.

import PRIME from "../prime.ts"; // triggers PRIME initialization + log emit
import { StabilityMatrix } from "../stability/stability_matrix.ts";
import { ThreadPool } from "./threads/thread_pool.ts";
import { anchorRegistry, resonanceBus } from "../memory/index.ts";
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
import { shouldProcessCognition, triggerCooldown } from "./cognition.ts";
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
import { LearningEngine } from "../cognition/learning_engine.ts";
import { MetaReasoningMonitor } from "../cognition/meta_reasoning_monitor.ts";
import { MetaReflectionEngine } from "../meta/meta_reflection_engine.ts";
import { SelfModelVector } from "../meta/self_model_vector.ts";
import { MetaLearningLayer } from "../meta/meta_learning_layer.ts";
import { DriftPredictor } from "../meta/drift_predictor.ts";
import { StrategyEngine } from "../strategy/strategy_engine.ts";
import { MemoryStore } from "../memory/memory_store.ts";
import { MemoryLayer } from "../memory/memory.ts";
import { PRIME_SITUATION } from "../situation_model/index.ts";
import { PRIME_TEMPORAL } from "../temporal/reasoner.ts";
import { EventCapture } from "../memory/episodic/event_capture.ts";
import { EpisodeBuilder } from "../memory/episodic/episode_builder.ts";
import { EpisodicArchive } from "../memory/episodic/episodic_archive.ts";
import { EpisodicReinforcementEngine } from "../memory/episodic/reinforcement_engine.ts";
import { MetaMemory } from "../memory/episodic/meta_memory.ts";
import { PatternGeneralizationEngine } from "../memory/episodic/pattern_generalization_engine.ts";
import { StrategicPatternGraph } from "../memory/episodic/strategic_pattern_graph.ts";
import { NeuralContextEncoder } from "../neural/context_encoder.ts";
import { NeuralMemory } from "../cognition/neural/neural_memory_bank.ts";
import crypto from "crypto";

console.log("[PRIME] Initializing Stability Matrix...");
StabilityMatrix.init();

console.log("[PRIME] Cognitive Fusion Stability Layer active.");

console.log("[PRIME] Initializing cognitive threads...");
// ThreadPool will be initialized with default instances
// For full integration, it should be initialized with PRIME's actual instances
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

// A65: Initialize Situation Model
console.log("[PRIME] Situation Model initialized.");

// A67: Initialize Episodic Memory System
const eventCapture = new EventCapture();
const episodeBuilder = new EpisodeBuilder();
const episodicArchive = new EpisodicArchive();

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
    
    // A74: Now cognitiveState has recall information, use it for reflection
    cognitiveState.lastReflection = null; // Will be set after reflection
    
    // A74: Reflection with recall-informed cognitive state
    const reflection = reflectionEngine.reflect(
      { ...cognitiveSnapshot, recall: cognitiveState.recall },
      sel
    );
    
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
  }
};

// A49: Connect kernel to intent router (after kernelInstance is created)
intentRouter.setKernel(kernelInstance);

// Start event-driven cognitive loop
cognitiveLoop.start();

// A39: PRIME's motivation heartbeat
setInterval(() => {
  const m = MotivationEngine.compute();
  console.log("[PRIME-HEARTBEAT] motivation:", m);

  // A40: Log highest-priority proto-goal
  const goals = ProtoGoalEngine.computeGoals();
  if (goals.length) {
    console.log("[PRIME-HEARTBEAT] top-goal:", goals[0]);
  }

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
}, 7000); // every 7 seconds — slow and intentional

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

// A48: Export kernel instance for operator command handling
export { kernelInstance };

