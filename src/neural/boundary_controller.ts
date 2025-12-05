// src/neural/boundary_controller.ts
// A89: Neural Inference Boundary Controller (NIBC)
// The firewall between PRIME's symbolic mind and any neural model we attach next.

export class NeuralBoundaryController {
  constructor() {
    this.enabled = true;
  }

  enabled: boolean;

  // Determines when PRIME is allowed to query neural modules
  allowNeuralInference(motivations: any) {
    if (!this.enabled) return false;

    const { curiosity, claritySeeking, stabilityPressure } = motivations;

    // PRIME can only use neural inference when stable enough
    if (stabilityPressure > 0.20) return false;

    // PRIME must have a cognitive reason
    return curiosity > 0.30 || claritySeeking > 0.10;
  }

  // Ensures encoded context is valid
  sanitizeInput(embedding: number[]) {
    if (!Array.isArray(embedding)) return [];
    return embedding.slice(0, 768); // future-proof for model changes
  }

  // Prevent malformed neural output from entering PRIME
  validateNeuralOutput(output: any) {
    if (!output) return null;

    if (typeof output !== "object") return null;

    // Must produce structured guidance
    if (!output.recommendation || !output.confidence) return null;

    // Must not override core motivational structure
    if (output.modifyMotivations) return null;

    output.confidence = Math.min(1, Math.max(0, output.confidence));

    return output;
  }

  // Final check: PRIME decides whether to accept neural help
  approve(output: any, motivations: any) {
    if (!output) return false;

    // PRIME must be in a receptive state
    if (motivations.stabilityPressure > 0.20) return false;

    // Confidence threshold
    return output.confidence >= 0.25;
  }
}

export const NeuralBoundary = new NeuralBoundaryController();

