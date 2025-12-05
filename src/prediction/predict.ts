// src/prediction/predict.ts

import { PredictiveCoherence } from "../cognition/predictive_coherence.ts";

export function runPrediction() {
  const consensus = PredictiveCoherence.computeConsensus();
  return {
    horizon: consensus.horizon,
    likelyNextIntent: consensus.likelyNextIntent,
    recursionRisk: consensus.recursionRisk,
  };
}

