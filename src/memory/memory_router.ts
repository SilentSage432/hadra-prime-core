// src/memory/memory_router.ts
// A93: Neural Memory Recall API
// Provides unified access to neural memory recall functionality
// A129: Shared Memory Coherence integration

import { getNeuralMemory } from "../kernel/index.ts";

export async function recallSimilar(inputEmbedding: number[], limit = 5) {
  const neuralMem = getNeuralMemory();
  if (!neuralMem) return [];
  return neuralMem.searchSimilar(inputEmbedding, limit);
}

/**
 * A129: Integrate shared memory from dual-mind coherence engine
 */
export function integrateSharedMemory(merged: any): void {
  const neuralMem = getNeuralMemory();
  const episodicArchive = (globalThis as any).__PRIME_EPISODIC_ARCHIVE__;

  if (merged.embedding && neuralMem) {
    // Store external embedding if neural memory supports it
    if (neuralMem.storeExternalEmbedding) {
      neuralMem.storeExternalEmbedding(merged.embedding);
    } else if (neuralMem.store) {
      // Fallback: use standard store method
      neuralMem.store(merged.embedding);
    }
  }

  if (merged.episodic && episodicArchive) {
    // Store external episodes if episodic archive supports it
    if (episodicArchive.storeExternalEpisode) {
      for (const ep of merged.episodic) {
        episodicArchive.storeExternalEpisode(ep);
      }
    } else {
      // Fallback: use standard store method
      for (const ep of merged.episodic) {
        episodicArchive.store(ep);
      }
    }
  }

  if (merged.narrative) {
    // Update narrative if narrative engine is available
    const narrativeEngine = (globalThis as any).__PRIME_NARRATIVE_ENGINE__;
    if (narrativeEngine?.updateWithExternal) {
      narrativeEngine.updateWithExternal(merged.narrative);
    }
  }

  console.log("[MEMORY] Shared memory integrated.");
}

