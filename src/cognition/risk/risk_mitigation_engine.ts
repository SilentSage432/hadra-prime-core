// src/cognition/risk/risk_mitigation_engine.ts
// A125: Cognitive Risk Mitigation Engine (Proactive Error Prevention)

export interface RiskSignal {
  conceptDrift: number;
  predictionVariance: number;
  uncertainty: number;
  narrativeInstability: number;
  emotionalDrift: number;
}

export type MitigationStrategy = 
  | "normal"
  | "slow_down_reasoning"
  | "increase_reflection_depth"
  | "stabilize_and_pause"
  | "halt_and_request_operator";

export class RiskMitigationEngine {
  assess(signals: Partial<RiskSignal>): number {
    const conceptDrift = signals.conceptDrift ?? 0;
    const predictionVariance = signals.predictionVariance ?? 0;
    const uncertainty = signals.uncertainty ?? 0;
    const narrativeInstability = signals.narrativeInstability ?? 0;
    const emotionalDrift = signals.emotionalDrift ?? 0;

    return (
      0.25 * uncertainty +
      0.20 * conceptDrift +
      0.20 * predictionVariance +
      0.20 * narrativeInstability +
      0.15 * emotionalDrift
    );
  }

  chooseMitigation(score: number): MitigationStrategy {
    if (score > 0.85) return "halt_and_request_operator";
    if (score > 0.70) return "stabilize_and_pause";
    if (score > 0.55) return "increase_reflection_depth";
    if (score > 0.40) return "slow_down_reasoning";
    return "normal";
  }

  // Helper method to check if risk is critical
  isCritical(score: number): boolean {
    return score > 0.85;
  }

  // Helper method to check if risk is high
  isHigh(score: number): boolean {
    return score > 0.70;
  }

  // Helper method to check if risk is moderate
  isModerate(score: number): boolean {
    return score > 0.55 && score <= 0.70;
  }
}

