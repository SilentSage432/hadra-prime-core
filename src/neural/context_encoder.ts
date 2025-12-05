// src/neural/context_encoder.ts
// A72: Neural Context Encoding Layer (NCEL)

import type { CognitiveState, MotivationState } from "../shared/types.ts";
import { NeuralMemory } from "../cognition/neural/neural_memory_bank.ts";

export interface NeuralContextVector {
  vector: number[];
  confidence: number;
  timestamp: number;
  metadata: Record<string, any>;
}

export interface StabilitySnapshot {
  score?: number;
  stabilityScore?: number;
  recursionRisk?: number;
  coherenceScore?: number;
  subsystems?: any;
}

export class NeuralContextEncoder {
  private VECTOR_SIZE = 64; // Stable, fixed-size embedding window

  encodeContext(
    cognitive: CognitiveState,
    motivation: MotivationState,
    stability: StabilitySnapshot
  ): NeuralContextVector {
    const features: number[] = [];

    // ---- 1) Encode Motivation ----
    features.push(
      this.normalize(motivation.urgency),
      this.normalize(motivation.curiosity),
      this.normalize(motivation.claritySeeking),
      this.normalize(motivation.consolidation),
      this.normalize(motivation.goalBias),
      this.normalize(motivation.stabilityPressure)
    );

    // ---- 2) Encode Cognitive State ----
    features.push(
      this.bool(motivation.direction !== null),
      this.hash(cognitive.activeGoal?.type),
      this.normalize(cognitive.confidence ?? 0),
      this.normalize(cognitive.uncertainty ?? 0)
    );

    // ---- 3) Encode Stability Snapshot ----
    const stabilityScore = stability.score ?? stability.stabilityScore ?? 0;
    const recursionRisk = stability.recursionRisk ?? 0;
    const coherenceScore = stability.coherenceScore ?? stabilityScore;
    
    features.push(
      this.normalize(stabilityScore),
      this.normalize(recursionRisk),
      this.normalize(coherenceScore)
    );

    // ---- 4) Encode Reflection Summary ----
    features.push(
      this.hash(cognitive.lastReflection?.reason),
      this.normalize(cognitive.lastReflection?.pressure ?? 0)
    );

    // ---- 5) Pad or trim vector for model compatibility ----
    const vector = this.stabilizeVector(features);

    const neuralContext: NeuralContextVector = {
      vector,
      confidence: 1.0, // PRIME is confident in its own state representation
      timestamp: Date.now(),
      metadata: {
        goal: cognitive.activeGoal?.type ?? null,
        uncertainty: cognitive.uncertainty,
        stability: stabilityScore
      }
    };

    // A73: Write to neural memory bank
    NeuralMemory.addMemory({
      embedding: vector,
      tag: "context",
      timestamp: neuralContext.timestamp,
      weight: 1.0,
      metadata: { 
        stateSummary: cognitive.activeGoal?.type ?? "unknown",
        goal: cognitive.activeGoal?.type ?? null,
        uncertainty: cognitive.uncertainty,
        stability: stabilityScore
      }
    });

    return neuralContext;
  }

  // -------------------------- Utilities --------------------------

  private normalize(v: number): number {
    if (isNaN(v)) return 0;
    return Math.max(-1, Math.min(1, v));
  }

  private bool(v: boolean): number {
    return v ? 1 : 0;
  }

  private hash(input?: string | null): number {
    if (!input) return 0;

    let h = 0;
    for (let i = 0; i < input.length; i++) {
      h = (h * 31 + input.charCodeAt(i)) % 1000;
    }
    return h / 1000; // normalize to [0,1]
  }

  private stabilizeVector(f: number[]): number[] {
    const arr = [...f];
    while (arr.length < this.VECTOR_SIZE) arr.push(0);
    return arr.slice(0, this.VECTOR_SIZE);
  }
}

