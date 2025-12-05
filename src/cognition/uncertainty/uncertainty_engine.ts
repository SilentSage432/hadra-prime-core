// src/cognition/uncertainty/uncertainty_engine.ts
// A124: Multi-Layered Uncertainty Engine (Cognitive Risk Awareness)

export interface UncertaintyVector {
  epistemic: number;      // uncertainty in knowledge
  predictive: number;     // uncertainty in predictions
  conceptual: number;     // instability or drift in concepts
  situational: number;    // unclear environment
  relational: number;     // unclear operator or social intent
}

export interface UncertaintyInput {
  noise?: number;
  variance?: number;
  conceptDrift?: number;
  situationalAmbiguity?: number;
  trustFluctuation?: number;
}

export class UncertaintyEngine {
  compute(input: UncertaintyInput): UncertaintyVector {
    return {
      epistemic: (input.noise ?? 0.1) * (Math.random() * 0.2 + 0.9), // Add some variation
      predictive: Math.abs(input.variance ?? 0),
      conceptual: input.conceptDrift ?? 0,
      situational: input.situationalAmbiguity ?? 0,
      relational: input.trustFluctuation ?? 0
    };
  }

  summarize(vector: UncertaintyVector): number {
    return (
      0.25 * vector.epistemic +
      0.25 * vector.predictive +
      0.20 * vector.conceptual +
      0.20 * vector.situational +
      0.10 * vector.relational
    );
  }

  // Helper method to check if uncertainty is high
  isHigh(score: number): boolean {
    return score > 0.75;
  }

  // Helper method to check if uncertainty is moderate
  isModerate(score: number): boolean {
    return score > 0.55 && score <= 0.75;
  }

  // Helper method to check if uncertainty is low
  isLow(score: number): boolean {
    return score <= 0.55;
  }
}

