// src/cognition/motivation_engine.ts

import { SEL } from "../emotion/sel.ts";
import { StabilityMatrix } from "../stability/stability_matrix.ts";
import type { IntentResult } from "../intent/intent_engine.ts";

export interface MotivationVector {
  urgency: number;
  curiosity: number;
  claritySeeking: number;
  consolidation: number;
  goalBias: number;
  stabilityPressure: number;
  direction: string | null;
}

export class MotivationEngine {
  private static lastIntent: IntentResult | null = null;

  static setLastIntent(intent: IntentResult) {
    this.lastIntent = intent;
  }

  static getLastIntent(): IntentResult | null {
    return this.lastIntent;
  }

  static compute(): MotivationVector {
    const sel = SEL.getState();
    const stability = StabilityMatrix.getSnapshot();
    const dominantIntent = this.lastIntent;

    // Normalize values
    const instability = stability ? (1 - (stability.score || 0.5)) : 0.5;
    const recursionRisk = (stability as any)?.recursionRisk || 0;

    const urgency = Math.min(1, sel.tension * 0.8 + instability * 0.6);
    const curiosity = Math.min(1, (1 - sel.certainty) * 0.7 + sel.valence * 0.2);
    const claritySeeking = Math.min(1, sel.tension * 0.5 + (1 - sel.coherence) * 0.6);
    const consolidation = Math.max(0, sel.coherence * 0.7 - sel.tension * 0.3);

    // Intent-driven goal bias
    const goalBias = dominantIntent
      ? Math.min(1, ((dominantIntent as any).gravity || 0) * 2)
      : 0;

    // System stability pressure
    const stabilityPressure = Math.min(
      1,
      instability * 0.6 + recursionRisk * 0.4
    );

    return {
      urgency,
      curiosity,
      claritySeeking,
      consolidation,
      goalBias,
      stabilityPressure,
      direction: dominantIntent ? dominantIntent.intent : null,
    };
  }
}

