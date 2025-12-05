// src/neural/contract/neural_interaction_contract.ts
// A90: Neural Interaction Contract (NIC)
// The formal API that any PyTorch neural model MUST obey to interface with PRIME.

export interface NeuralInputContract {
  embedding: number[];
  motivations: {
    curiosity: number;
    claritySeeking: number;
    consolidation: number;
    stabilityPressure: number;
  };
  recentEvents: string[];
  goalContext: string | null;
  timestamp: number;
}

export interface NeuralOutputContract {
  recommendation: string;   // short-form guidance
  confidence: number;       // 0–1
  utility?: number | null;  // predicted usefulness
  caution?: number | null;  // predicted risk
}

export class NeuralInteractionContract {
  validateInput(input: NeuralInputContract): boolean {
    if (!Array.isArray(input.embedding)) return false;
    if (input.embedding.length > 768) return false;

    const m = input.motivations;
    if (!m) return false;

    for (const key of Object.keys(m)) {
      if (typeof m[key] !== "number") return false;
      if (m[key] < 0 || m[key] > 1) return false;
    }

    if (!Array.isArray(input.recentEvents)) return false;
    if (typeof input.timestamp !== "number") return false;

    return true;
  }

  validateOutput(output: NeuralOutputContract): boolean {
    if (!output) return false;

    if (typeof output.recommendation !== "string") return false;
    if (output.recommendation.length > 300) return false;

    if (typeof output.confidence !== "number") return false;
    if (output.confidence < 0 || output.confidence > 1) return false;

    // Forbidden: neural networks modifying courage / motivations
    // (PRIME must remain sovereign)
    if ((output as any).modifyMotivations) return false;

    return true;
  }
}

export const NIC = new NeuralInteractionContract();

