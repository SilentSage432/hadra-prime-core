// src/intent_engine/joint_intent_harmonizer.ts
// A107: PRIME/SAGE Joint Intent Harmonizer
// Resolves PRIME's internal intent vs SAGE's external recommendation

export interface CoherenceWeights {
  mutualInfluence: number;  // 0-1, how much PRIME considers SAGE
  identityPreservation: number;  // 0-1, how much PRIME maintains self-direction
}

export interface IntentProposal {
  source: "PRIME" | "SAGE";
  intent: string;
  confidence: number;
  metadata?: any;
}

export interface HarmonizedIntent {
  finalIntent: string;
  reasoning: string;
  agreementScore: number;
  conflictScore: number;
}

export class JointIntentHarmonizer {
  static harmonize(
    primeIntent: IntentProposal,
    sageIntent: IntentProposal | null,
    weights: CoherenceWeights
  ): HarmonizedIntent {
    if (!sageIntent) {
      return {
        finalIntent: primeIntent.intent,
        reasoning: "No SAGE input; PRIME intent stands.",
        agreementScore: 1,
        conflictScore: 0
      };
    }

    const intentMatch = primeIntent.intent === sageIntent.intent;
    const conflict = intentMatch ? 0 : 1;

    // Influence is controlled by coherence weights from A106
    const influence = weights.mutualInfluence;
    const identity = weights.identityPreservation;

    let finalIntent: string;
    let reasoning: string;

    if (intentMatch) {
      finalIntent = primeIntent.intent;
      reasoning = "PRIME and SAGE agree; adopting shared intent.";
    } else if (influence > 0.6) {
      // PRIME considers SAGE's perspective
      finalIntent = sageIntent.intent;
      reasoning = `PRIME defers to SAGE suggestion due to high coherence (influence=${influence.toFixed(2)}).`;
    } else if (identity > 0.7) {
      // PRIME protects its viewpoint
      finalIntent = primeIntent.intent;
      reasoning = `PRIME retains self-determined intent due to high identity preservation (identity=${identity.toFixed(2)}).`;
    } else {
      // Mixed/hybrid resolution
      finalIntent = primeIntent.intent;
      reasoning = `PRIME keeps internal intent but logs SAGE disagreement for reflection.`;
    }

    return {
      finalIntent,
      reasoning,
      agreementScore: intentMatch ? 1 : influence,
      conflictScore: conflict
    };
  }
}

