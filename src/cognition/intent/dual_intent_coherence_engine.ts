// src/cognition/intent/dual_intent_coherence_engine.ts
// A128: PRIME/SAGE Dual-Intent Coherence Engine

export interface UnifiedIntent {
  unifiedGoal: any;
  unifiedUrgency: number;
  unifiedPredictionVector: number[] | null;
  weights: { prime: number; sage: number };
  rawSignals: any;
  timestamp: number;
  source: "dual-mind";
}

export interface IntentSignals {
  primeGoal: any;
  sageGoal: any;
  primeMotivation: any;
  sageMotivation: any;
  primeUrgency: number;
  sageUrgency: number;
  primePredictionVector: number[] | null;
  sagePredictionVector: number[] | null;
}

export interface MergedIntent {
  unifiedGoal: any;
  unifiedUrgency: number;
  unifiedPredictionVector: number[] | null;
}

export class DualIntentCoherenceEngine {
  constructor() {}

  computeCoherentIntent(primeIntent: any, sageIntent: any): UnifiedIntent {
    const raw = this.extractSignals(primeIntent, sageIntent);
    const weights = this.computeWeights(raw);
    const merged = this.merge(raw, weights);

    return {
      ...merged,
      weights,
      rawSignals: raw,
      timestamp: Date.now(),
      source: "dual-mind"
    };
  }

  extractSignals(prime: any, sage: any): IntentSignals {
    return {
      primeGoal: prime.topGoal ?? prime.activeGoal,
      sageGoal: sage.topGoal ?? sage.activeGoal,
      primeMotivation: prime.motivation,
      sageMotivation: sage.motivation,
      primeUrgency: prime.urgency ?? prime.motivation?.urgency ?? 0,
      sageUrgency: sage.urgency ?? sage.motivation?.urgency ?? 0,
      primePredictionVector: prime.predictionVector ?? null,
      sagePredictionVector: sage.predictionVector ?? null
    };
  }

  computeWeights(signals: IntentSignals): { prime: number; sage: number } {
    const urgencyDelta = Math.abs(signals.primeUrgency - signals.sageUrgency);
    const avgUrgency = (signals.primeUrgency + signals.sageUrgency) / 2;

    // Normalize weights so they sum to 1.0
    const primeWeight = Math.max(0.2, 1 - urgencyDelta + avgUrgency * 0.5);
    const sageWeight = Math.max(0.2, urgencyDelta + avgUrgency * 0.5);
    const sum = primeWeight + sageWeight;

    return {
      prime: primeWeight / sum,
      sage: sageWeight / sum
    };
  }

  merge(raw: IntentSignals, weights: { prime: number; sage: number }): MergedIntent {
    return {
      unifiedGoal: this.blendGoals(raw.primeGoal, raw.sageGoal, weights),
      unifiedUrgency:
        (raw.primeUrgency * weights.prime + raw.sageUrgency * weights.sage) /
        (weights.prime + weights.sage),
      unifiedPredictionVector:
        this.mergeVectors(raw.primePredictionVector, raw.sagePredictionVector, weights)
    };
  }

  blendGoals(primeGoal: any, sageGoal: any, weights: { prime: number; sage: number }): any {
    if (!primeGoal) return sageGoal;
    if (!sageGoal) return primeGoal;

    // If goals match → no need to blend
    if (primeGoal.type === sageGoal.type) {
      return {
        ...primeGoal,
        priority: (primeGoal.priority ?? 0.5) * weights.prime + (sageGoal.priority ?? 0.5) * weights.sage
      };
    }

    return {
      type: `${primeGoal.type ?? "unknown"}|${sageGoal.type ?? "unknown"}`,
      priority: ((primeGoal.priority ?? 0.5) * weights.prime + 
                 (sageGoal.priority ?? 0.5) * weights.sage) /
                (weights.prime + weights.sage)
    };
  }

  mergeVectors(
    v1: number[] | null,
    v2: number[] | null,
    weights: { prime: number; sage: number }
  ): number[] | null {
    if (!v1 && !v2) return null;
    if (!v1) return v2;
    if (!v2) return v1;

    const merged: number[] = [];
    const len = Math.min(v1.length, v2.length);

    for (let i = 0; i < len; i++) {
      merged.push(
        (v1[i] * weights.prime + v2[i] * weights.sage) /
        (weights.prime + weights.sage)
      );
    }

    return merged;
  }
}

