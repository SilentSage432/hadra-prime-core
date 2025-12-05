// src/safety/recursion_guard.ts

import { PredictiveCoherence } from "../cognition/predictive_coherence.ts";

export class RecursionGuard {
  static maxDepth = 20;

  static shouldHalt(depth: number): boolean {
    const prediction = PredictiveCoherence.computeConsensus();

    // Consensus recursion risk high? We stop earlier.
    if (prediction.recursionRisk > 0.7 && depth > 1) {
      console.log("[PRIME-SAFETY] Consensus recursion risk high. Halting.");
      return true;
    }

    return depth > this.maxDepth;
  }
}

