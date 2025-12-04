// src/cognition/fusion_engine.ts

import type { ToneVector } from "../expression/tone/tone_detector.ts";
import type { CognitiveState } from "./cognitive_state.ts";
import { MemoryStore } from "../memory/memory_store.ts";
import { CognitiveStabilizer } from "./stability/cognitive_stabilizer.ts";
import { ReasoningIntegrity } from "./bias/integrity_layer.ts";

export class FusionEngine {
  private memory: MemoryStore;

  constructor(memory: MemoryStore) {
    this.memory = memory;
  }

  buildCognitiveState(intent: any, tone: ToneVector, contextSnapshot: any): CognitiveState {
    const start = performance.now();

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

    // Build fusion payload
    const fusion = {
      meaning: {
        priorityLevel,
        riskLevel,
        operatorFocus,
        recommendedResponseMode,
        memoryRecall,
      },
      intent,
      context: contextSnapshot,
    };

    // Measure fusion latency
    const latency = performance.now() - start;

    // Stabilize the fusion
    const stabilized = CognitiveStabilizer.stabilize(fusion, latency);

    // If stabilization returned null or degraded, use simplified state
    if (!stabilized || stabilized.meaning?.degraded) {
      return {
        intent: stabilized?.intent || intent,
        tone,
        context: stabilized?.context || contextSnapshot,
        memory: memoryRecall,
        priorityLevel: stabilized?.meaning?.priorityLevel || priorityLevel,
        riskLevel: stabilized?.meaning?.riskLevel || riskLevel,
        operatorFocus: stabilized?.meaning?.operatorFocus || operatorFocus,
        recommendedResponseMode: stabilized?.meaning?.recommendedResponseMode || recommendedResponseMode,
      };
    }

    // Evaluate reasoning integrity and correct bias
    const integrityResult = ReasoningIntegrity.evaluate(stabilized);
    const corrected = ReasoningIntegrity.correct(stabilized, integrityResult.integrity);

    // Store integrity result for downstream use
    (corrected as any).integrity = integrityResult;

    // Return stabilized and integrity-corrected cognitive state
    const cognitiveState: CognitiveState = {
      intent: corrected.intent,
      tone,
      context: corrected.context,
      memory: corrected.meaning.memoryRecall,
      priorityLevel: corrected.meaning.priorityLevel,
      riskLevel: corrected.meaning.riskLevel,
      operatorFocus: corrected.meaning.operatorFocus,
      recommendedResponseMode: corrected.meaning.recommendedResponseMode,
    };

    // Attach integrity result for downstream use
    (cognitiveState as any).integrity = integrityResult;

    return cognitiveState;
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

