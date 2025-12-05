// src/distributed/prediction_vector.ts

import type { Prediction } from "../cognition/predictive_horizon.ts";

/**
 * Converts a Prediction object to a numeric vector for consensus computation
 */
export function predictionToVector(prediction: Prediction): number[] {
  // Encode stability trend: rising=1, stable=0.5, falling=0
  let stabilityValue = 0.5;
  if (prediction.stabilityTrend === "rising") stabilityValue = 1;
  else if (prediction.stabilityTrend === "falling") stabilityValue = 0;

  // Encode horizon: medium=1, short=0
  const horizonValue = prediction.horizon === "medium" ? 1 : 0;

  // Vector: [recursionRisk, stabilityTrend, horizon]
  return [
    prediction.recursionRisk,
    stabilityValue,
    horizonValue,
  ];
}

