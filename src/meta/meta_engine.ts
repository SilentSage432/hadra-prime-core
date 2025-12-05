// src/meta/meta_engine.ts

import type { CognitiveState } from "../cognition/cognitive_state.ts";
import type { MetaState } from "./meta_state.ts";
import { defaultMetaState } from "./meta_state.ts";
import { evaluateHeuristics } from "./heuristics.ts";
import { StabilityMonitor } from "./stability_monitor.ts";
import { healingPulse } from "./healing_pulse.ts";
import { MetaStabilityState } from "./meta_stability_state.ts";
import { TemporalRing } from "../memory/temporal_ring.ts";
import { PredictiveCoherence } from "../cognition/predictive_coherence.ts";
import { PredictiveConsensus } from "../distributed/predictive_consensus.ts";

export class MetaEngine {
  private stability = new StabilityMonitor();
  private stabilityState = new MetaStabilityState();

  evaluate(state: CognitiveState): MetaState {
    try {
      // Create a state signature for stability monitoring
      const stateSignature = JSON.stringify({
        intent: state.intent?.type,
        focus: state.operatorFocus,
        mode: state.recommendedResponseMode,
      });

      // Check for instability (repeated cognitive states)
      if (this.stability.assess(stateSignature)) {
        console.warn("[PRIME-META] Unstable cognition detected — applying healing pulse.");
        const pulse = healingPulse();
        this.stabilityState.applyHealing(pulse);
        this.stability.reset();

        // Return a degraded meta state when healing is applied
        return {
          ...defaultMetaState,
          contextQuality: "low",
          actionRecommendation: "defer",
          notes: ["Meta-stability healing pulse applied. Cognition re-centering."],
        };
      }

      // Normal evaluation
      return evaluateHeuristics(state);
    } catch (err) {
      return {
        ...defaultMetaState,
        contextQuality: "low",
        actionRecommendation: "warn",
        notes: ["Meta-engine fallback activated."],
      };
    }
  }

  /**
   * Get internal stability state for monitoring
   */
  getStabilityState() {
    return this.stabilityState;
  }

  /**
   * Reset stability monitoring (for testing or recovery)
   */
  resetStability() {
    this.stability.reset();
    this.stabilityState.reset();
  }

  /**
   * Get internal state including temporal cognition data
   */
  getState() {
    return {
      stabilityState: this.stabilityState,
      temporal: TemporalRing.getRecent(10),
      predictionConsensus: PredictiveCoherence.computeConsensus(),
      predictionVector: PredictiveConsensus.getCurrentPredictionVector(),
    };
  }
}

export const metaEngine = new MetaEngine();

