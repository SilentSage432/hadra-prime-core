// src/cognition/fusion_engine.ts

import type { ToneVector } from "../expression/tone/tone_detector.ts";
import type { CognitiveState } from "./cognitive_state.ts";
import { MemoryStore } from "../memory/memory_store.ts";

export class FusionEngine {
  private memory: MemoryStore;

  constructor(memory: MemoryStore) {
    this.memory = memory;
  }

  buildCognitiveState(intent: any, tone: ToneVector, contextSnapshot: any): CognitiveState {
    const memoryRecall = this.memory.retrieveRelevant(intent?.type || "general");

    const priorityLevel = this.computePriority(intent, contextSnapshot);
    const riskLevel = this.computeRisk(intent);
    const operatorFocus = this.inferOperatorFocus(intent, contextSnapshot, tone);
    const recommendedResponseMode = this.determineResponseMode(
      operatorFocus,
      tone,
      priorityLevel,
      riskLevel
    );

    return {
      intent,
      tone,
      context: contextSnapshot,
      memory: memoryRecall,
      priorityLevel,
      riskLevel,
      operatorFocus,
      recommendedResponseMode,
    };
  }

  private computePriority(intent: any, context: any): number {
    if (!intent) return 0.2;
    if (intent.type === "error") return 1.0;
    if (intent.type === "system_action") return 0.9;
    if (intent.type === "question") return 0.5;
    return 0.3;
  }

  private computeRisk(intent: any): number {
    if (!intent) return 0.1;
    if (intent.type === "system_action") return 0.7;
    if (intent.type === "dangerous") return 1.0;
    return 0.1;
  }

  private inferOperatorFocus(intent: any, context: any, tone: ToneVector): "build" | "debug" | "ideate" | "learn" | "unknown" {
    if (tone.emotionalState === "frustrated") return "debug";

    const lastIntent = context?.latestSession?.value || {};

    if (lastIntent.type === "build_step") return "build";
    if (lastIntent.type === "phase_switch") return "ideate";
    if (intent?.type === "question") return "learn";

    return "unknown";
  }

  private determineResponseMode(
    operatorFocus: string,
    tone: ToneVector,
    priority: number,
    risk: number
  ): "direct" | "supportive" | "detailed" | "cautious" {
    if (risk > 0.7) return "cautious";
    if (operatorFocus === "debug") return "direct";
    if (operatorFocus === "learn") return "detailed";
    if (tone.emotionalState === "frustrated") return "supportive";
    return "direct";
  }
}

