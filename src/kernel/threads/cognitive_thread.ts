// src/kernel/threads/cognitive_thread.ts

import { FusionEngine } from "../../cognition/fusion_engine.ts";
import { IntentEngine } from "../../intent_engine/intent_engine.ts";
import { ExpressionRouter } from "../../expression/expression_router.ts";
import { MemoryStore } from "../../memory/memory_store.ts";
import { SafetyGuard } from "../../safety/safety_guard.ts";
import { ToneEngine } from "../../expression/tone/tone_engine.ts";
import { ContextManager } from "../../context/context_manager.ts";

export class CognitiveThread {
  id: string;
  active: boolean = true;
  private fusionEngine: FusionEngine;
  private intentEngine: IntentEngine;
  private expressionRouter: ExpressionRouter;
  private memoryStore: MemoryStore;
  private toneEngine: ToneEngine;
  private contextManager: ContextManager;

  constructor(
    id: string,
    fusionEngine: FusionEngine,
    intentEngine: IntentEngine,
    expressionRouter: ExpressionRouter,
    memoryStore: MemoryStore,
    toneEngine: ToneEngine,
    contextManager: ContextManager
  ) {
    this.id = id;
    this.fusionEngine = fusionEngine;
    this.intentEngine = intentEngine;
    this.expressionRouter = expressionRouter;
    this.memoryStore = memoryStore;
    this.toneEngine = toneEngine;
    this.contextManager = contextManager;
  }

  async process(payload: any) {
    if (!this.active) return null;

    // Safety first
    if (!SafetyGuard.preCognitionCheck()) return null;

    try {
      // 1. Intent resolution
      const intent = this.intentEngine.process(payload.input || payload.content || "");

      // 2. Extract tone
      const tone = this.toneEngine.analyze(payload.input || payload.content || "");

      // 3. Build context snapshot
      const contextSnapshot = {
        latestSession: this.contextManager.latest("session"),
        latestEmotion: this.contextManager.latest("emotion"),
      };

      // 4. Fusion (build cognitive state)
      const fused = this.fusionEngine.buildCognitiveState(intent, tone, contextSnapshot);

      // 5. Expression
      const expression = this.expressionRouter.route({
        type: intent.type,
        content: payload.input || payload.content,
        intent: intent,
        cognitiveState: fused,
        context: contextSnapshot,
      });

      // 6. Store memory
      this.memoryStore.logInteraction("thread", {
        thread: this.id,
        fused,
        intent,
        output: expression,
      });

      return {
        thread: this.id,
        fused,
        intent,
        output: expression,
      };
    } catch (error) {
      console.error(`[PRIME-THREAD-${this.id}] Processing error:`, error);
      return null;
    }
  }
}

