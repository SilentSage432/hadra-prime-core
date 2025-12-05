// src/cognition/predictive_coherence.ts

import type { Prediction } from "./predictive_horizon.ts";

interface ThreadPrediction {
  threadId: number;
  prediction: Prediction;
}

export class PredictiveCoherence {
  private static buffer: ThreadPrediction[] = [];
  private static maxBuffer = 12;

  static submit(threadId: number, prediction: Prediction) {
    this.buffer.push({ threadId, prediction });
    if (this.buffer.length > this.maxBuffer) {
      this.buffer.shift();
    }
  }

  static computeConsensus(): Prediction {
    if (this.buffer.length === 0) {
      return {
        horizon: "short",
        stabilityTrend: "stable",
        likelyNextIntent: null,
        recursionRisk: 0,
      };
    }

    const risks = this.buffer.map(p => p.prediction.recursionRisk);
    const avgRisk = risks.reduce((a, b) => a + b, 0) / risks.length;

    const recentIntents = this.buffer
      .map(p => p.prediction.likelyNextIntent)
      .filter(Boolean);

    const likelyNextIntent =
      recentIntents.length > 0 ? recentIntents.at(-1) : null;

    return {
      horizon: "medium",
      stabilityTrend: "stable",
      likelyNextIntent,
      recursionRisk: avgRisk,
    };
  }
}

