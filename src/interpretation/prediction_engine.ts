// src/interpretation/prediction_engine.ts

import { PredictiveCoherence } from "../cognition/predictive_coherence.ts";

interface PredictionOptions {
  depth: "short" | "medium" | "long";
  requireIntent: boolean;
}

export class PredictionEngine {
  generate(options: PredictionOptions) {
    if (options.requireIntent) {
      // Only generate prediction if there's an active intent
      // For now, return consensus prediction
      const consensus = PredictiveCoherence.computeConsensus();
      return {
        horizon: consensus.horizon,
        stabilityTrend: consensus.stabilityTrend,
        likelyNextIntent: consensus.likelyNextIntent,
        recursionRisk: consensus.recursionRisk,
      };
    }
    
    return {
      horizon: "idle" as const,
      stabilityTrend: "stable" as const,
      likelyNextIntent: null,
      recursionRisk: 0,
    };
  }
}

export const predictionEngine = new PredictionEngine();

