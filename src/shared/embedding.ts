// src/shared/embedding.ts
// A113: Shared Embedding Schema for PRIME-SAGE neural communication

export function generateEmbedding(size = 16): number[] {
  return Array.from({ length: size }, () => Math.random());
}

/**
 * A113: Shared Embedding Schema
 * Unified format for neural embeddings between PRIME and SAGE
 */
export interface SharedEmbedding {
  vector: number[];
  origin: "PRIME" | "SAGE";
  epoch: number; // Timeline index / timestamp
  signalType: "cognition" | "state" | "intent" | "emotion";
}

