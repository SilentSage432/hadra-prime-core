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
  private loopInterval: any = null;
  private memory: MemoryBroker;
  private memoryLayer: MemoryLayer; // Enhanced memory for interactions
  private memoryStore: MemoryStore; // Memory wrapper for expression system
  private intentEngine: IntentEngine;
  private expression: ExpressionRouter; // Expression routing system
  private safety: SafetyGuard; // Safety filtering
  private context: ContextManager; // Context lattice for continuity awareness
  private fusion: FusionEngine; // Cognitive fusion engine
  private toneEngine: ToneEngine; // Tone analysis for cognitive state

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
   */
  private startLoop() {
    this.updateTickRate(); // set initial adaptive tick
    this.loopInterval = setInterval(() => this.tick(), this.currentTick);
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
  private tick() {
    this.updateTickRate();
    this.emit("heartbeat", this.getStatus());

    // Placeholder for future cognition steps:
    // 1. Perception → collect signals
    // 2. Interpretation → semantic frames
    // 3. Intent engine → choose goal
    // 4. Action layer → execute

    this.log(`TICK (${this.currentTick.toFixed(0)}ms) load=${this.cognitiveLoad}`);
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
   * Process command (gateway compatibility)
   * Now uses Intent Engine for cognitive interpretation
   */
  async processCommand(input: string) {
    this.log(`Command received: ${input}`);

    // Process through Intent Engine
    const intent = this.intentEngine.process(input);
    
    // Extract tone if content exists
    const tone = this.toneEngine.analyze(input || "");

    // Build context snapshot
    const contextSnapshot = {
      latestSession: this.context.latest("session"),
      latestEmotion: this.context.latest("emotion"),
    };

    // Build PRIME's unified cognitive state
    const cognitiveState = this.fusion.buildCognitiveState(
      intent,
      tone,
      contextSnapshot
    );

    // Step 4: Run meta-reasoning layer
    const metaState = metaEngine.evaluate(cognitiveState);

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
    const expressionPacket = this.expression.route({
      type: intent.type === "operator" && input.trim().toLowerCase() === "status" 
        ? "status_check" 
        : intent.type,
      content: input,
      intent: intent,
      context: contextSnapshot,
      cognitiveState: cognitiveState,
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
   * Expose meta-engine insights for debugging or UI in future
   */
  async getMetaInsights(input: string) {
    const result = await this.processCommand(input);
    const cognitiveState = (result.data as any)?.cognitiveState;
    const metaState = cognitiveState?.meta;
    return { cognitiveState, metaState };
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

export function subscribeLogs(listener: LogListener) {
  logListeners.add(listener);
  return () => logListeners.delete(listener);
}

export default PRIME;

