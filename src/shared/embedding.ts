// src/shared/embedding.ts

export function generateEmbedding(size = 16): number[] {
  return Array.from({ length: size }, () => Math.random());
}

