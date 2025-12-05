// A113 — PRIME Embedding Adapter
// Converts PRIME's internal cognition embeddings into shared embedding space
// Enables PRIME → SAGE neural communication

import type { SharedEmbedding } from "../../shared/embedding.ts";

export class PrimeEmbeddingAdapter {
  /**
   * Normalize a vector to unit length
   */
  normalize(vec: number[]): number[] {
    if (vec.length === 0) return vec;
    
    const mag = Math.sqrt(vec.reduce((a, b) => a + b * b, 0));
    if (mag === 0) return vec;
    
    return vec.map(v => v / mag);
  }

  /**
   * Compress vector to shared-space dimensionality
   * Future: PCA or small MLP compressor
   */
  compress(vec: number[]): number[] {
    // Temporary shared-space dimensionality: 256
    // This will be replaced with learned compression in future phases
    if (vec.length <= 256) return vec;
    
    // Simple truncation for now
    return vec.slice(0, 256);
  }

  /**
   * Convert PRIME embedding to shared embedding format
   */
  toSharedEmbedding(
    vec: number[],
    signalType: "cognition" | "state" | "intent" | "emotion" = "cognition"
  ): SharedEmbedding {
    const norm = this.normalize(vec);
    const compressed = this.compress(norm);

    return {
      vector: compressed,
      origin: "PRIME",
      epoch: Date.now(),
      signalType
    };
  }

  /**
   * Batch convert multiple PRIME embeddings
   */
  toSharedEmbeddings(
    vectors: number[],
    signalType: "cognition" | "state" | "intent" | "emotion" = "cognition"
  ): SharedEmbedding[] {
    return vectors.map(vec => this.toSharedEmbedding(vec, signalType));
  }
}

export const primeEmbeddingAdapter = new PrimeEmbeddingAdapter();

