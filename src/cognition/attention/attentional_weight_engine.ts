// src/cognition/attention/attentional_weight_engine.ts
// A121: Adaptive Attentional Weight Shifting Engine

import type { CognitiveState } from "../cognitive_state.ts";
import type { DriftReport } from "../../situation_model/drift_engine.ts";

export interface AttentionalWeights {
  micro: number;
  meso: number;
  macro: number;
}

export class AttentionalWeightEngine {
  private learningRate = 0.05;
  private baseWeights: AttentionalWeights = {
    micro: 0.33,
    meso: 0.33,
    macro: 0.33
  };
  private weights: AttentionalWeights = { ...this.baseWeights };

  computeOptimalShift(
    drift: DriftReport | null,
    state: Partial<CognitiveState>
  ): AttentionalWeights {
    // No drift → gentle recentering
    if (!drift || !drift.needsCorrection) {
      this.recenterWeights();
      return this.weights;
    }

    const level = drift.dominantDrift;

    console.log(`[PRIME-AWE] drift signal detected: ${level} → recalibrating attention weights.`);

    // Increase emphasis on the domain that needs correction
    if (level === "micro") this.weights.micro += this.learningRate;
    if (level === "meso") this.weights.meso += this.learningRate;
    if (level === "macro") this.weights.macro += this.learningRate;

    this.normalize();

    console.log("[PRIME-AWE] updated attentional weights:", {
      micro: this.weights.micro.toFixed(3),
      meso: this.weights.meso.toFixed(3),
      macro: this.weights.macro.toFixed(3)
    });

    return this.weights;
  }

  private recenterWeights() {
    // Move slowly back toward balanced attention
    Object.keys(this.weights).forEach(key => {
      const target = this.baseWeights[key as keyof AttentionalWeights];
      const current = this.weights[key as keyof AttentionalWeights];
      this.weights[key as keyof AttentionalWeights] += (target - current) * 0.02;
    });
    this.normalize();
  }

  private normalize() {
    const sum = this.weights.micro + this.weights.meso + this.weights.macro;
    if (sum > 0) {
      this.weights.micro /= sum;
      this.weights.meso /= sum;
      this.weights.macro /= sum;
    }
  }

  getWeights(): AttentionalWeights {
    return { ...this.weights };
  }

  reset() {
    this.weights = { ...this.baseWeights };
  }
}

