// src/meta/drift_predictor.ts

export class DriftPredictor {
  evaluate(smv: any): number {
    // Drift increases when consolidation tension exceeds exploration
    const drift = Math.max(
      0,
      smv.consolidationTension - (smv.explorationBias + smv.clarityIndex * 0.4)
    );

    return Math.min(1.0, drift);
  }
}

