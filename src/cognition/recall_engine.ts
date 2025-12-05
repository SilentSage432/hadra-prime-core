// src/cognition/recall_engine.ts
// A74: Neural Recall Engine

import { NeuralMemory } from "./neural/neural_memory_bank.ts";
import { cosineSimilarity } from "./neural/similarity.ts";

export interface RecallResult {
  embedding: number[];
  relevance: number;
  metadata?: Record<string, any>;
}

export interface RecallSummary {
  intuition: number;
  reference: any | null;
}

export class RecallEngine {
  recall(embedding: number[], limit = 5): RecallResult[] {
    const matches = NeuralMemory.findClosest(embedding, limit);

    return matches.map((m) => ({
      embedding: m.entry.embedding,
      relevance: m.similarity,
      metadata: m.entry.metadata
    }));
  }

  /** A summarized intuition signal PRIME can use during planning */
  summarizeRecall(results: RecallResult[]): RecallSummary | null {
    if (results.length === 0) return null;

    const avgRelevance =
      results.reduce((a, r) => a + r.relevance, 0) / results.length;
    const strongest = results[0];

    return {
      intuition: avgRelevance,
      reference: strongest.metadata || null,
    };
  }
}

export const Recall = new RecallEngine();

