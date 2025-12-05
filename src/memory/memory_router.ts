// src/memory/memory_router.ts
// A93: Neural Memory Recall API
// Provides unified access to neural memory recall functionality

import { getNeuralMemory } from "../kernel/index.ts";

export async function recallSimilar(inputEmbedding: number[], limit = 5) {
  const neuralMem = getNeuralMemory();
  if (!neuralMem) return [];
  return neuralMem.searchSimilar(inputEmbedding, limit);
}

