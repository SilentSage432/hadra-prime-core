// src/cognition/prediction/foresight_engine.ts
// A79: PRIME's Long-Range Predictive Model (Proto-Foresight Engine)

import type { MotivationState } from "../../shared/types.ts";
import { Knowledge } from "../knowledge/knowledge_graph.ts";
import { PRIME_TEMPORAL } from "../../temporal/reasoner.ts";

export interface MotivationProjection {
  urgency: number;
  curiosity: number;
  claritySeeking: number;
  consolidation: number;
  goalBias: number;
  stabilityPressure: number;
  direction: string | null;
}

export interface SELProjection {
  valence: number;
  arousal: number;
  tension: number;
  coherence: number;
  certainty: number;
  affinity: number;
  stabilityPressure?: number;
}

export interface DomainForecast {
  id: string;
  projectedStrength: number;
}

export interface SystemForecast {
  projectedMotivation: MotivationProjection[];
  projectedSEL: SELProjection[];
  domainForecast: DomainForecast[];
}

export class ForesightEngine {
  /** Predict future motivation gradients over N steps */
  predictMotivationTrajectory(
    current: MotivationState,
    steps = 50
  ): MotivationProjection[] {
    const projections: MotivationProjection[] = [];
    let m: MotivationProjection = { ...current };

    for (let i = 0; i < steps; i++) {
      // Simple projection rules for proto-foresight
      // Urgency decays naturally over time
      m.urgency = Math.max(0, m.urgency * 0.98);

      // Curiosity tends to grow slightly over time
      m.curiosity = Math.min(1, m.curiosity + 0.0005);

      // Clarity-seeking increases when consolidation is high
      m.claritySeeking = Math.min(1, m.claritySeeking + 0.001);

      // Consolidation naturally decays over time
      m.consolidation = Math.max(0, m.consolidation * 0.99);

      // Goal bias maintains with slight decay
      m.goalBias = Math.max(0, m.goalBias * 0.995);

      // Stability pressure adjusts based on consolidation
      if (m.consolidation > 0.5) {
        m.stabilityPressure = Math.min(1, m.stabilityPressure + 0.0002);
      } else {
        m.stabilityPressure = Math.max(0, m.stabilityPressure * 0.998);
      }

      // Direction remains unless significant shift
      // (simplified model - in reality this would be more complex)

      projections.push({ ...m });
    }

    return projections;
  }

  /** Predict SEL evolution */
  predictSelfRegulation(sel: any, steps = 50): SELProjection[] {
    const results: SELProjection[] = [];
    let s: SELProjection = {
      valence: sel.valence ?? 0.5,
      arousal: sel.arousal ?? 0.5,
      tension: sel.tension ?? 0.5,
      coherence: sel.coherence ?? 0.5,
      certainty: sel.certainty ?? 0.5,
      affinity: sel.affinity ?? 0.5,
      stabilityPressure: sel.stabilityPressure ?? 0,
    };

    for (let i = 0; i < steps; i++) {
      // Stability pressure increases when coherence is high (paradoxical - high coherence creates pressure)
      s.stabilityPressure = (s.stabilityPressure || 0) + (s.coherence * -0.0004);

      // Coherence gradually decays (needs maintenance)
      s.coherence = Math.max(0, s.coherence * 0.999);

      // Tension decreases when coherence is high
      if (s.coherence > 0.7) {
        s.tension = Math.max(0, s.tension * 0.995);
      } else {
        s.tension = Math.min(1, s.tension * 1.001);
      }

      // Certainty drifts toward coherence
      s.certainty = s.certainty * 0.9 + s.coherence * 0.1;

      // Affinity increases with stability
      if (s.coherence > 0.6) {
        s.affinity = Math.min(1, s.affinity * 1.001);
      }

      // Valence adjusts based on tension
      s.valence = s.valence * 0.95 + (1 - s.tension) * 0.05;

      // Arousal adjusts based on tension
      s.arousal = s.arousal * 0.95 + s.tension * 0.05;

      results.push({ ...s });
    }

    return results;
  }

  /** Predict future domain activations based on knowledge graph strength */
  predictDomainTrends(): DomainForecast[] {
    const graph = Knowledge.exportGraph();
    const domainNodes = graph.nodes.filter((n) => n.type === "domain");

    const domainScores: DomainForecast[] = domainNodes.map((d) => ({
      id: d.id,
      projectedStrength: d.weight * 1.02, // Mild growth model
    }));

    return domainScores;
  }

  /** High-level foresight summary */
  forecastSystemState(
    currentMotivation: MotivationState,
    sel: any,
    neuralCoherence?: any
  ): SystemForecast {
    // A112: Compute temporal embedding for foresight
    const temporalVector = PRIME_TEMPORAL.computeTemporalEmbedding(
      PRIME_TEMPORAL.long,
      "foresight_projection"
    );
    
    // A113: Apply neural coherence weight to predictions
    let predictionScore = 1.0;
    if (neuralCoherence) {
      predictionScore = 1 + neuralCoherence.relevance * 0.05;
      console.log("[PRIME-PREDICTION] Neural coherence weight applied:", {
        baseScore: 1.0,
        adjustedScore: predictionScore.toFixed(3),
        relevance: neuralCoherence.relevance.toFixed(3)
      });
    }
    
    const forecast: SystemForecast = {
      projectedMotivation: this.predictMotivationTrajectory(currentMotivation, 25),
      projectedSEL: this.predictSelfRegulation(sel, 25),
      domainForecast: this.predictDomainTrends(),
    };
    
    // A112: Attach temporal vector to forecast
    (forecast as any).temporalVector = temporalVector;
    
    // A113: Attach neural coherence score
    (forecast as any).neuralCoherenceScore = predictionScore;
    
    return forecast;
  }

  /** Predict action outcomes (what will happen if we take this action) */
  predictActionOutcome(
    actionType: string,
    currentMotivation: MotivationState,
    currentSEL: any
  ): {
    motivationChange: Partial<MotivationState>;
    selChange: Partial<SELProjection>;
    confidence: number;
  } {
    // Simplified model - predicts immediate effects of actions
    let motivationChange: Partial<MotivationState> = {};
    let selChange: Partial<SELProjection> = {};
    let confidence = 0.5;

    switch (actionType) {
      case "reflect":
        motivationChange.claritySeeking = 0.05;
        selChange.coherence = 0.02;
        confidence = 0.7;
        break;

      case "explore":
        motivationChange.curiosity = 0.1;
        motivationChange.claritySeeking = -0.02;
        selChange.tension = 0.03;
        confidence = 0.65;
        break;

      case "consolidate":
        motivationChange.consolidation = 0.15;
        motivationChange.curiosity = -0.05;
        selChange.coherence = 0.03;
        confidence = 0.75;
        break;

      default:
        confidence = 0.3;
    }

    return { motivationChange, selChange, confidence };
  }
}

export const Foresight = new ForesightEngine();

