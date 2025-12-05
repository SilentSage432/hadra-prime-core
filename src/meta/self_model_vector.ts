// src/meta/self_model_vector.ts

export class SelfModelVector {
  stabilityIndex: number = 0.95;
  clarityIndex: number = 0.60;
  explorationBias: number = 0.30;
  consolidationTension: number = 0.70;
  driftRisk: number = 0.10;

  updateFromCycle(goalSummary: any, motivation: any) {
    // Update from consolidation vs curiosity tension
    this.consolidationTension = motivation.consolidation || 0;
    this.explorationBias = motivation.curiosity || 0;
    this.clarityIndex = Math.max(0.01, 1 - this.consolidationTension + this.explorationBias * 0.3);

    // Stability increases when consolidation reduces slightly
    this.stabilityIndex = Math.min(1.0, 0.9 + (1 - (motivation.consolidation || 0)) * 0.1);
  }
}

