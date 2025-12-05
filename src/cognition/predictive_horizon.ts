// src/cognition/predictive_horizon.ts

import { TemporalRing } from "../memory/temporal_ring.ts";

export interface Prediction {
  horizon: "short" | "medium";
  stabilityTrend: "rising" | "falling" | "stable";
  likelyNextIntent?: string | null;
  recursionRisk: number; // 0–1
}

export class PredictiveHorizon {
  static analyze(): Prediction {
    const recent = TemporalRing.getRecent(8);

    if (recent.length < 3) {
      return {
        horizon: "short",
        stabilityTrend: "stable",
        likelyNextIntent: null,
        recursionRisk: 0,
      };
    }

    const lastThree = recent.slice(-3);
    const stability = lastThree.map(r => r.stabilityScore ?? 1);

    // --- Trend detection ---
    const delta1 = stability[1] - stability[0];
    const delta2 = stability[2] - stability[1];
    const avgDelta = (delta1 + delta2) / 2;

    let stabilityTrend: "rising" | "falling" | "stable";
    if (avgDelta > 0.05) stabilityTrend = "rising";
    else if (avgDelta < -0.05) stabilityTrend = "falling";
    else stabilityTrend = "stable";

    // --- Likely Next Intent ---
    const intents = recent
      .map(r => r.intent?.type)
      .filter(Boolean)
      .reverse();

    const likelyNextIntent = intents[0] ?? null;

    // --- Recursion Risk Estimation ---
    const recursionRisk =
      Math.min(
        1,
        Math.abs(avgDelta) < 0.01 && intents[0] === intents[1] ? 0.8 : 0.2
      );

    return {
      horizon: "medium",
      stabilityTrend,
      likelyNextIntent,
      recursionRisk,
    };
  }
}

