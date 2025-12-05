// src/meta/meta_learning_layer.ts

export class MetaLearningLayer {
  private lastClarity = 0.6;

  updateStrategies(smv: any, driftRisk: number) {
    const clarityGain = smv.clarityIndex - this.lastClarity;

    if (clarityGain > 0.01) {
      console.log("[PRIME-META] Meta-learning: reinforcing strategy (+clarity)");
    } else if (clarityGain < -0.01) {
      console.log("[PRIME-META] Meta-learning: pruning ineffective strategy");
    }

    if (driftRisk > 0.25) {
      console.log("[PRIME-META] Meta-learning: drift detected → adjusting exploration weight");
      smv.explorationBias += 0.02;
      // Clamp exploration bias
      smv.explorationBias = Math.max(0, Math.min(1, smv.explorationBias));
    }

    this.lastClarity = smv.clarityIndex;
  }
}

