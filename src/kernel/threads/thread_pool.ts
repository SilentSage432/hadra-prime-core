// src/kernel/threads/thread_pool.ts

import { CognitiveThread } from "./cognitive_thread.ts";
import { StabilityMatrix } from "../../stability/stability_matrix.ts";
import { FusionEngine } from "../../cognition/fusion_engine.ts";
import { IntentEngine } from "../../intent_engine/intent_engine.ts";
import { ExpressionRouter } from "../../expression/expression_router.ts";
import { MemoryStore } from "../../memory/memory_store.ts";
import { ToneEngine } from "../../expression/tone/tone_engine.ts";
import { ContextManager } from "../../context/context_manager.ts";
import { SafetyGuard } from "../../safety/safety_guard.ts";
import { MemoryLayer } from "../../memory/memory.ts";
import { PredictiveHorizon } from "../../cognition/predictive_horizon.ts";
import { PredictiveCoherence } from "../../cognition/predictive_coherence.ts";

export class ThreadPool {
  static threads: CognitiveThread[] = [];
  static maxThreads = 4; // scalable in future
  private static memoryStore: MemoryStore | null = null;
  private static contextManager: ContextManager | null = null;

  static init(memoryStore?: MemoryStore, contextManager?: ContextManager) {
    this.threads = [];

    // A108b: Adjust thread limit based on hardware
    const adaptiveConfig = (globalThis as any).__PRIME_ADAPTIVE_CONFIG__;
    if (adaptiveConfig) {
      this.maxThreads = adaptiveConfig.threadLimit;
      console.log(`[PRIME-THREADS] Hardware-adaptive thread limit: ${this.maxThreads}`);
    }

    // Create shared memory store and context manager if not provided
    if (!memoryStore) {
      const memoryLayer = new MemoryLayer();
      this.memoryStore = new MemoryStore(memoryLayer);
    } else {
      this.memoryStore = memoryStore;
    }

    if (!contextManager) {
      this.contextManager = new ContextManager(this.memoryStore);
    } else {
      this.contextManager = contextManager;
    }

    // Create thread instances with shared components
    for (let i = 0; i < this.maxThreads; i++) {
      const fusionEngine = new FusionEngine(this.memoryStore);
      const intentEngine = new IntentEngine();
      const expressionRouter = new ExpressionRouter(this.memoryStore, new SafetyGuard());
      const toneEngine = new ToneEngine();

      this.threads.push(
        new CognitiveThread(
          `T${i + 1}`,
          fusionEngine,
          intentEngine,
          expressionRouter,
          this.memoryStore,
          toneEngine,
          this.contextManager
        )
      );
    }

    console.log(`[PRIME] ThreadPool initialized with ${this.maxThreads} threads.`);
  }

  static async dispatch(payload: any) {
    if (this.threads.length === 0) {
      console.warn("[PRIME] ThreadPool not initialized. Call ThreadPool.init() first.");
      return null;
    }

    // Pick least-loaded thread (simple round-robin for now)
    const thread = this.threads.shift()!;
    this.threads.push(thread);

    const result = await thread.process(payload);

    // Update stability metrics
    StabilityMatrix.update("cognition", {
      load: Math.random() * 0.2, // placeholder — real load model soon
    });

    // FIXED: Thread predictions are now conditional to prevent recursion storms
    // Only submit predictions when explicitly requested (e.g., via event flag)
    if (result && (result as any).allowPrediction) {
      const prediction = PredictiveHorizon.analyze();
      // Extract numeric thread ID from string ID (e.g., "T1" -> 1)
      const threadIdNum = parseInt(thread.id.replace(/\D/g, "")) || 0;
      PredictiveCoherence.submit(threadIdNum, prediction);
    }

    return result;
  }
}

