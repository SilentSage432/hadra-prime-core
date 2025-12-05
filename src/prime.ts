/**
 * HADRA-PRIME CORE ENGINE
 * Hybrid Neural Loop Kernel with Adaptive Heartbeat
 */

import { EventEmitter } from "events";
import MemoryBroker from "./memory/index.ts";
import { MemoryLayer } from "./memory/memory.ts";
import { MemoryStore } from "./memory/memory_store.ts";
import { IntentEngine } from "./intent_engine/intent_engine.ts";
import { ExpressionRouter } from "./expression/expression_router.ts";
import { SafetyGuard } from "./safety/safety_guard.ts";
import { ContextManager } from "./context/context_manager.ts";
import { FusionEngine } from "./cognition/fusion_engine.ts";
import { ToneEngine } from "./expression/tone/tone_engine.ts";
import { metaEngine } from "./meta/meta_engine.ts";
import { StabilityMatrix } from "./stability/stability_matrix.ts";
import { triggerCognition, triggerInput } from "./kernel/event_bus.ts";
import { PRIMEConfig } from "./shared/config.ts";
import { cognition } from "./cognition/cognition.ts";
import { HierarchicalPlanner } from "./planning/hierarchical_planner.ts";
import { SelfConsistencyEngine } from "./reflection/self_consistency.ts";
import { PerceptionHub } from "./perception/perception_hub.ts";
import { MultimodalPerception } from "./perception/multimodal.ts";
import { PRIME_SITUATION } from "./situation_model/index.ts";
import { PRIME_TEMPORAL } from "./temporal/reasoner.ts";

export type PrimeLifecycleState = "initializing" | "online" | "degraded" | "offline";

export interface PrimeStatus {
  state: PrimeLifecycleState;
  cognitiveLoad: number;
  uptime: number;
  lastHeartbeat: string;
  tickRate: number; // ms
}

type LogListener = (msg: string) => void;
const logListeners: Set<LogListener> = new Set();

class PrimeEngine extends EventEmitter {
  private startTime = Date.now();
  private state: PrimeLifecycleState = "initializing";

  // Adaptive heartbeat baseline values
  private minTick = 250;  // fastest
  private maxTick = 1200; // slowest
  private currentTick = 800;

  private cognitiveLoad = 5; // 1–100 dynamic load (later modules adjust this)
  private throttleFactor = 0; // 0–1, increases when unstable
  private loopInterval: any = null;
  private dualMindActive = false; // A105: Dual-Mind mode state
  private memory: MemoryBroker;
  private memoryLayer: MemoryLayer; // Enhanced memory for interactions
  private memoryStore: MemoryStore; // Memory wrapper for expression system
  private intentEngine: IntentEngine;
  private expression: ExpressionRouter; // Expression routing system
  private safety: SafetyGuard; // Safety filtering
  private context: ContextManager; // Context lattice for continuity awareness
  private fusion: FusionEngine; // Cognitive fusion engine
  private toneEngine: ToneEngine; // Tone analysis for cognitive state
  planner: HierarchicalPlanner; // A43: Hierarchical planning engine
  reflector: SelfConsistencyEngine; // A44: Reflective self-consistency engine
  perceptionHub: PerceptionHub; // A46: Perception hub for multimodal signals
  multimodal: MultimodalPerception; // A46: Multimodal perception layer

  // Module placeholders (wired in later phases)
  private modules: Record<string, any> = {
    perception: null,
    interpretation: null,
    intent: null,
    memory: null,
    action: null,
    safety: null,
    expression: null,
  };

  constructor() {
    super();
    // Wire up log listeners to EventEmitter
    this.on("log", (message: string) => {
      const formatted = `[PRIME] ${new Date().toISOString()} — ${message}`;
      for (const listener of logListeners) listener(formatted);
    });

    this.memory = new MemoryBroker(); // NEW: cognitive memory system
    this.memoryLayer = new MemoryLayer(); // NEW: enhanced interaction memory
    this.memoryStore = new MemoryStore(this.memoryLayer); // Memory wrapper for expression
    this.safety = new SafetyGuard(); // Safety filtering
    this.intentEngine = new IntentEngine(); // NEW: intent processing engine
    this.expression = new ExpressionRouter(this.memoryStore, this.safety); // Expression routing
    this.context = new ContextManager(this.memoryStore); // Context lattice integration
    this.toneEngine = new ToneEngine(); // Tone analysis for cognitive fusion
    this.fusion = new FusionEngine(this.memoryStore); // Cognitive fusion engine
    this.planner = new HierarchicalPlanner(); // A43: Hierarchical planning engine
    this.reflector = new SelfConsistencyEngine(); // A44: Reflective self-consistency engine
    this.perceptionHub = new PerceptionHub(); // A46: Perception hub for multimodal signals
    this.multimodal = new MultimodalPerception(this.perceptionHub); // A46: Multimodal perception layer

    // Initialize Stability Matrix
    StabilityMatrix.init();

    this.bootstrap(); // system startup
    this.loadModules(); // NEW: dynamic module wiring
  }

  /**
   * STEP 1 — Bootstrap PRIME core & modules
   */
  private bootstrap() {
    this.state = "initializing";
    this.log("BOOT → PRIME INTELLIGENCE CORE initializing…");

    // In future phases, modules will be loaded here dynamically.
    this.state = "online";
    this.log("PRIME CORE ONLINE.");

    this.startLoop();
  }

  /**
   * STEP 2 — Load PRIME modules dynamically.
   * Each module must export a default class with an init(context) method.
   */
  private loadModules() {
    this.log("MODULE LOADER → Wiring PRIME internal systems…");

    const context = {
      emit: this.emit.bind(this),
      log: this.log.bind(this),
      getStatus: this.getStatus.bind(this),
      remember: this.remember.bind(this),
      store: this.store.bind(this),
      recall: this.recall.bind(this),
      recallRecent: this.recallRecent.bind(this),
    };

    const moduleMap: Record<string, string> = {
      perception: "./perception/index.ts",
      interpretation: "./interpretation/index.ts",
      intent: "./intent_engine/index.ts",
      memory: "./memory/index.ts",
      action: "./action_layer/index.ts",
      safety: "./safety/index.ts",
      expression: "./expression/index.ts",
    };

    for (const [key, path] of Object.entries(moduleMap)) {
      try {
        // Dynamic import so PRIME can evolve without modifying core
        import(path).then((mod) => {
          if (mod && typeof mod.default === "function") {
            const instance = new mod.default();
            if (instance.init) instance.init(context);
            this.modules[key] = instance;
            this.log(`MODULE ONLINE → ${key}`);
          } else {
            this.log(`MODULE ERROR → Invalid module at ${key}`);
          }
        }).catch((err) => {
          this.log(`MODULE LOAD FAILED → ${key}: ${err}`);
        });
      } catch (err) {
        this.log(`MODULE LOAD FAILED → ${key}: ${err}`);
      }
    }
  }

  /**
   * STEP 3 — Hybrid cognitive loop
   * A synchronous "tick" plus asynchronous subsystems.
   * 
   * FIXED: Disabled auto-looping tick. PRIME is now event-driven.
   * The tick() method is only called when explicitly triggered via events.
   */
  private startLoop() {
    // DISABLED: Auto-looping tick removed to prevent recursion storms
    // PRIME now operates in event-driven mode via cognitive_loop.ts
    // this.updateTickRate(); // set initial adaptive tick
    // this.loopInterval = setInterval(() => { this.tick().catch(err => console.error("Tick error:", err)); }, this.currentTick);
    this.log("PRIME tick loop disabled — operating in event-driven mode");
  }

  /**
   * Adaptive heartbeat: tick interval changes based on cognitive load.
   */
  private updateTickRate() {
    // Convert load (1–100) into ms delay between minTick and maxTick
    const loadRatio = Math.min(Math.max(this.cognitiveLoad / 100, 0), 1);
    this.currentTick = this.maxTick - (this.maxTick - this.minTick) * loadRatio;
  }

  /**
   * Synchronous neural pulse: PRIME's thinking cycle.
   */
  private async tick() {
    // Pre-cognition safety check
    if (!SafetyGuard.preCognitionCheck()) {
      this.log("⚠️ Cognition cycle skipped due to safety constraints.");
      return; // skip cycle but keep loop alive
    }

    const tickStart = Date.now();
    this.updateTickRate();
    this.emit("heartbeat", this.getStatus());

    // Placeholder for future cognition steps:
    // 1. Perception → collect signals
    // 2. Interpretation → semantic frames
    // 3. Intent engine → choose goal
    // 4. Action layer → execute

    const cycleLatency = Date.now() - tickStart;
    const taskLoad = this.cognitiveLoad / 100;
    const taskErrors = 0; // TODO: track actual errors

    // Update memory pressure (estimate based on memory layer size)
    const memorySize = this.memoryLayer.getRecent(1000).length;
    const memoryPressure = Math.min(memorySize / 1000, 1); // Normalize to 0-1
    SafetyGuard.limiter.setMemoryPressure(memoryPressure);

    // Update stability for each cycle
    StabilityMatrix.update("cognition", {
      latency: cycleLatency,
      load: taskLoad,
      errors: taskErrors,
    });

    // Check for instability
    if (StabilityMatrix.unstable()) {
      this.log("⚠️ Stability degradation detected — throttling cognition loop.");
      this.reduceLoad();
    }

    this.log(`TICK (${this.currentTick.toFixed(0)}ms) load=${this.cognitiveLoad}`);
  }

  /**
   * Reduce cognitive load when instability is detected
   */
  private reduceLoad() {
    this.throttleFactor = Math.min(this.throttleFactor + 0.1, 1);
    this.cognitiveLoad = Math.min(this.cognitiveLoad + 5, 100);
    this.log(`Throttle factor increased to ${this.throttleFactor.toFixed(2)}`);
  }

  /**
   * Apply healing pulse effects to recover from instability
   */
  private applyHealingPulse(pulse: any) {
    this.log("🔄 Applying healing pulse — re-centering cognition.");

    // Clear transient scratch memory (reset recursion counter)
    SafetyGuard.limiter.resetRecursion();

    // Reduce cognitive load
    if (pulse.reduceCognitiveLoad) {
      this.cognitiveLoad = Math.max(5, this.cognitiveLoad - 10);
      this.log(`Cognitive load reduced to ${this.cognitiveLoad}`);
    }

    // Recompute stability matrix (already updated, but log the healing)
    const snapshot = StabilityMatrix.getSnapshot();
    this.log(`Stability score after healing: ${snapshot.score.toFixed(2)}`);

    // Recenter by clearing conflicting context spans
    if (pulse.recenter) {
      // Clear expired or conflicting context
      // The context manager will naturally expire old contexts via TTL
      this.log("Context spans re-centered.");
    }
  }

  /**
   * Logging pipeline
   */
  private log(message: string) {
    this.emit("log", message);
  }

  /**
   * Public API consumed by Gateway
   */
  getStatus(): PrimeStatus {
    return {
      state: this.state,
      cognitiveLoad: this.cognitiveLoad,
      uptime: (Date.now() - this.startTime) / 1000,
      lastHeartbeat: new Date().toISOString(),
      tickRate: this.currentTick,
    };
  }

  /**
   * Get stability snapshot for monitoring
   */
  getStabilitySnapshot() {
    return StabilityMatrix.getSnapshot();
  }

  /**
   * Get safety limiter snapshot for monitoring
   */
  getSafetySnapshot() {
    return SafetyGuard.snapshot();
  }

  /**
   * Process command (gateway compatibility)
   * Now uses Intent Engine for cognitive interpretation
   */
  async processCommand(input: string) {
    // Reset recursion counter for new command
    SafetyGuard.limiter.resetRecursion();

    // Trigger event-driven cognition with intent classification
    PRIMEConfig.runtime.hasStimulus = true;
    triggerInput(input); // Triggers intent classification via event bus
    triggerCognition(true);

    this.log(`Command received: ${input}`);

    const processStart = Date.now();

    // Process through Intent Engine
    const intentStart = Date.now();
    const intent = this.intentEngine.process(input);
    const intentLatency = Date.now() - intentStart;
    
    StabilityMatrix.update("intent_engine", {
      latency: intentLatency,
      load: intent.confidence || 0.5,
      errors: 0,
    });
    
    // Extract tone if content exists
    const tone = this.toneEngine.analyze(input || "");

    // Build context snapshot
    const recentEvents = this.perceptionHub.getRecentEvents(20);
    const contextSnapshot = {
      latestSession: this.context.latest("session"),
      latestEmotion: this.context.latest("emotion"),
      recentEvents, // A46: Include recent events for multimodal fusion
    };

    // Build PRIME's unified cognitive state (with fusion stability)
    const fusionStart = performance.now();
    const cognitiveState = await this.fusion.buildCognitiveState(
      intent,
      tone,
      contextSnapshot
    );
    const fusionLatency = performance.now() - fusionStart;
    const fusionSize = JSON.stringify(cognitiveState).length;
    const fusionErrors = cognitiveState.meta?.contextQuality === "low" ? 1 : 0;
    
    // Get integrity score from cognitive state (attached by fusion engine)
    const integrityScore = (cognitiveState as any).integrity?.integrity ?? 1.0;
    const integrityErrors = integrityScore < 0.4 ? 1 : 0;

    // Update stability matrix with fusion metrics (including integrity)
    StabilityMatrix.update("cognition", {
      latency: fusionLatency,
      load: Math.min(fusionSize / 5000, 1), // Normalize to 0-1
      errors: fusionErrors + integrityErrors, // Combine meta and integrity errors
    });

    // Step 4: Run meta-reasoning layer
    const metaStart = Date.now();
    const metaState = metaEngine.evaluate(cognitiveState);
    const metaLatency = Date.now() - metaStart;

    StabilityMatrix.update("meta", {
      latency: metaLatency,
      load: metaState.certaintyLevel,
      errors: metaState.contextQuality === "low" ? 1 : 0,
    });

    // Check if healing pulse was applied
    const stabilityState = metaEngine.getStabilityState();
    const recentHealing = stabilityState.healingEvents.slice(-1)[0];
    if (recentHealing && Date.now() - recentHealing.timestamp < 100) {
      // Apply healing pulse effects
      this.applyHealingPulse(recentHealing);
    }

    // Attach meta state to cognitive state for downstream awareness
    cognitiveState.meta = metaState;

    // Attach cognitive state for downstream use
    (intent as any).cognitiveState = cognitiveState;
    
    // Update micro-context
    this.context.update(
      "session",
      "intent",
      intent,
      1000 * 60 * 5 // 5-minute TTL
    );

    // Store emotional tone when available
    if (input) {
      this.context.update(
        "emotion",
        "input",
        input,
        1000 * 60 * 10
      );
    }
    
    // Store interaction in enhanced memory layer
    this.memoryLayer.storeInteraction(input, intent);
    
    // Also store in general memory
    this.remember({ type: "command", input, intent });
    this.store("interactions", { input, intent, timestamp: new Date() });

    this.log(`Intent detected: ${intent.type} (confidence: ${intent.confidence.toFixed(2)})`);

    // Route through expression system to generate response
    const expressionStart = Date.now();
    const expressionPacket = this.expression.route({
      type: intent.type === "operator" && input.trim().toLowerCase() === "status" 
        ? "status_check" 
        : intent.type,
      content: input,
      intent: intent,
      context: contextSnapshot,
      cognitiveState: cognitiveState,
    });
    const expressionLatency = Date.now() - expressionStart;

    StabilityMatrix.update("expression", {
      latency: expressionLatency,
      load: expressionPacket.confidence || 0.5,
      errors: expressionPacket.type === "error" ? 1 : 0,
    });

    const totalLatency = Date.now() - processStart;
    StabilityMatrix.update("cognition", {
      latency: totalLatency,
      load: cognitiveState.priorityLevel,
      errors: 0,
    });

    // Handle specific intent types
    if (intent.type === "operator" && input.trim().toLowerCase() === "status") {
      const status = this.getStatus();
      this.log("Status report generated.");
      return {
        type: "status",
        data: status,
        intent: intent,
        expression: expressionPacket,
      };
    }

    // Return expression packet for gateway
    return {
      type: "intent",
      data: intent,
      expression: expressionPacket,
      message: expressionPacket.message,
    };
  }

  /**
   * Memory API - PRIME calls this for all internal events
   */
  remember(event: any) {
    this.memory.remember(event);
    this.log(`MEMORY(STM) + event captured`);
  }

  /**
   * Memory API - Topic-based long-term storage
   */
  store(topic: string, entry: any) {
    this.memory.store(topic, entry);
    this.log(`MEMORY(LTM) + stored under ${topic}`);
  }

  /**
   * Memory API - Recall from long-term memory by topic
   */
  recall(topic: string) {
    return this.memory.recall(topic);
  }

  /**
   * Memory API - Get recent short-term memory
   */
  recallRecent() {
    return this.memory.recallRecent();
  }

  /**
   * Memory API - Get recent interaction memory (enhanced)
   */
  getRecentMemory(n: number = 5) {
    return this.memoryLayer.getRecent(n);
  }

  /**
   * A43: Build hierarchical plan from intent
   */
  async buildPlan(intent: string) {
    const context = this.contextSnapshot();
    const plan = this.planner.buildPlan(intent, context);
    
    // A44: Run self-consistency review
    console.log("[PRIME-REFLECT] Running self-consistency review...");
    const reflection = await this.evaluatePlan(plan);
    console.log("[PRIME-REFLECT-RESULT]", reflection);
    
    return plan;
  }

  /**
   * A44: Evaluate plan for self-consistency
   */
  async evaluatePlan(plan: any) {
    return this.reflector.review(plan);
  }

  /**
   * A43: Generate context snapshot for planning
   */
  private contextSnapshot() {
    return {
      latestSession: this.context.latest("session"),
      latestEmotion: this.context.latest("emotion"),
      stability: StabilityMatrix.getSnapshot(),
      status: this.getStatus(),
      recentMemory: this.getRecentMemory(5)
    };
  }

  /**
   * Expose meta-engine insights for debugging or UI in future
   */
  async getMetaInsights(input: string) {
    const result = await this.processCommand(input);
    const cognitiveState = (result.data as any)?.cognitiveState;
    const metaState = cognitiveState?.meta;
    return { cognitiveState, metaState };
  }

  // A105: Dual-Mind Activation Control (boundary-aware)
  enableDualMind() {
    this.dualMindActive = true;
    console.log("[PRIME-DUAL] Dual-Mind Mode ENABLED.");
  }

  disableDualMind() {
    this.dualMindActive = false;
    console.log("[PRIME-DUAL] Dual-Mind Mode DISABLED.");
  }

  isDualMindActive() {
    return this.dualMindActive;
  }

  // A107: Apply harmonized intent from dual-mind collaboration
  applyHarmonizedIntent(h: any) {
    if (!this.dualMindActive) return;

    console.log("[PRIME] Applying harmonized intent:", h.finalIntent);

    // Apply intent through intent engine if method exists
    if (this.intentEngine && (this.intentEngine as any).setTopIntent) {
      (this.intentEngine as any).setTopIntent(h.finalIntent);
    }

    // Reflection learns from disagreement
    if (h.conflictScore > 0 && this.reflector && (this.reflector as any).logIntentConflict) {
      (this.reflector as any).logIntentConflict(h);
    }
  }

  // A108: Apply predictive forecast to adjust internal cognitive modulation
  applyForecast(forecast: any) {
    console.log("[PRIME] Forecast integration:", forecast);

    // PRIME gently adjusts curiosity, consolidation, or claritySeeking
    // based on expected alignment with SAGE.
    // Access learning engine through global kernel instance
    const kernelInstance = (globalThis as any).kernelInstance;
    const learningEngine = kernelInstance?.learningEngine;

    if (forecast.divergenceRisk > 0.6) {
      // PRIME becomes more introspective to avoid conflict
      learningEngine?.adjustExplorationBias?.(-0.01);
    } else {
      // PRIME becomes more open to external guidance
      learningEngine?.adjustExplorationBias?.(0.01);
    }

    // Store for reflection
    (this as any).predictedJointIntent = forecast.predictedIntent;
  }
}

const PRIME = new PrimeEngine();

// Named exports for gateway compatibility
export function getStatus(): PrimeStatus {
  return PRIME.getStatus();
}

export async function processCommand(input: string) {
  return PRIME.processCommand(input);
}

export async function getMetaInsights(input: string) {
  return PRIME.getMetaInsights(input);
}

export function primeReceive(input: string) {
  console.log("[PRIME] Received external input:", input);
  // Trigger intent classification and cognition
  PRIMEConfig.runtime.hasStimulus = true;
  triggerInput(input); // Triggers intent classification
  triggerCognition(true);
  // Process the command
  return PRIME.processCommand(input);
}

export function subscribeLogs(listener: LogListener) {
  logListeners.add(listener);
  return () => logListeners.delete(listener);
}

// A65: Safe accessor API for situation model
export function getSituationSnapshot() {
  return PRIME_SITUATION.snapshot();
}

// A66: Safe accessor API for temporal reasoning
export function getTemporalSummary() {
  return PRIME_TEMPORAL.summarize();
}

// A104b: Dual-Mind Activation Control
import { DualMind } from "./dual_core/dual_mind_activation.ts";

export const PRIME_DUAL_MIND = {
  enableDualMind: () => DualMind.activate(),
  disableDualMind: () => DualMind.deactivate()
};

export default PRIME;

