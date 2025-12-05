// src/neural/embedding/embedding_adapter.ts
// A92: Embedding Adapter Layer
// Bridges PRIME's symbolic world → neural world

import { CortexManager } from "../cortex/cortex_manager.ts";
import type { NeuralInputContract } from "../contract/neural_interaction_contract.ts";
import { MotivationEngine } from "../../cognition/motivation_engine.ts";

export class EmbeddingAdapter {
  private cortex: CortexManager;

  constructor(cortex: CortexManager) {
    this.cortex = cortex;
  }

  async embedText(text: string) {
    // Create NIC-compliant input contract
    const motivations = MotivationEngine.compute();
    const neuralInput: NeuralInputContract = {
      embedding: [], // Empty for embedding generation
      motivations: {
        curiosity: motivations.curiosity,
        claritySeeking: motivations.claritySeeking,
        consolidation: motivations.consolidation,
        stabilityPressure: motivations.stabilityPressure
      },
      recentEvents: [],
      goalContext: text, // Put text here for embedding model to extract
      timestamp: Date.now()
    };

    const result = await this.cortex.infer("embedding_model", neuralInput);

    if (!result) {
      return {
        embedding: [],
        meta: { error: true, reason: "Embedding failed" }
      };
    }

    // Extract embedding from NIC output (placeholder model puts it in recommendation)
    // In production, this would be a proper embedding field
    let embedding: number[] = [];
    try {
      const parsed = JSON.parse(result.recommendation);
      embedding = parsed.embedding || [];
    } catch {
      embedding = [];
    }

    return {
      embedding: embedding,
      meta: {
        type: "embedding",
        confidence: result.confidence,
        utility: result.utility,
        caution: result.caution
      }
    };
  }
}

