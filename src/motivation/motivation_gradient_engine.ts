// src/motivation/motivation_gradient_engine.ts

export class MotivationGradientEngine {
  computeGradients(smv: any) {
    return {
      clarityGradient: 0.75 - smv.clarityIndex,
      stabilityGradient: 0.85 - smv.stabilityIndex,
      coherenceGradient: 0.80 - (1 - smv.consolidationTension),
      curiosityGradient: (smv.explorationBias || 0) < 0.45 ? 0.10 : -0.05
    };
  }

  formIntentions(gradients: any) {
    const intentions: any[] = [];

    if (gradients.stabilityGradient > 0.05) {
      intentions.push({
        type: "improve_stability",
        strength: gradients.stabilityGradient,
        horizon: 4000
      });
    }

    if (gradients.clarityGradient > 0.05) {
      intentions.push({
        type: "improve_clarity",
        strength: gradients.clarityGradient,
        horizon: 2500
      });
    }

    if (gradients.coherenceGradient > 0.05) {
      intentions.push({
        type: "reduce_tension",
        strength: gradients.coherenceGradient,
        horizon: 3000
      });
    }

    // curiosity intention (exploration modulation)
    intentions.push({
      type: gradients.curiosityGradient > 0 ? "increase_exploration" : "decrease_exploration",
      strength: Math.abs(gradients.curiosityGradient),
      horizon: 1500
    });

    return intentions;
  }
}

