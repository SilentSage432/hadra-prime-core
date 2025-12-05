// src/memory/coherence/dual_memory_coherence_engine.ts
// A129: PRIME/SAGE Shared Memory Coherence Engine

export interface AlignedRepresentations {
  primeEmbedding: number[];
  sageEmbedding: number[];
  primeNarrative: any | null;
  sageNarrative: any | null;
  primeEpisodic: any[];
  sageEpisodic: any[];
}

export interface MergedMemory {
  embedding: number[];
  episodic: any[];
  narrative: any | null;
}

export interface SharedMemoryCoherencePacket {
  type: "shared-memory-coherence";
  mergedMemory: MergedMemory;
  drift: number;
  timestamp: number;
}

export class DualMemoryCoherenceEngine {
  constructor() {}

  generateCoherencePacket(primeMem: any, sageMem: any): SharedMemoryCoherencePacket {
    const aligned = this.alignRepresentations(primeMem, sageMem);
    const merged = this.merge(aligned);
    const drift = this.detectDrift(aligned);

    return {
      type: "shared-memory-coherence",
      mergedMemory: merged,
      drift,
      timestamp: Date.now()
    };
  }

  alignRepresentations(prime: any, sage: any): AlignedRepresentations {
    return {
      primeEmbedding: prime.embedding ?? [],
      sageEmbedding: sage.embedding ?? [],
      primeNarrative: prime.narrative ?? null,
      sageNarrative: sage.narrative ?? null,
      primeEpisodic: prime.episodic ?? [],
      sageEpisodic: sage.episodic ?? []
    };
  }

  merge(aligned: AlignedRepresentations): MergedMemory {
    // Merge embeddings
    const len = Math.min(
      aligned.primeEmbedding.length,
      aligned.sageEmbedding.length
    );
    const mergedEmbedding: number[] = [];

    for (let i = 0; i < len; i++) {
      mergedEmbedding.push(
        (aligned.primeEmbedding[i] + aligned.sageEmbedding[i]) / 2
      );
    }

    // Merge episodic traces
    const episodes = [
      ...aligned.primeEpisodic,
      ...aligned.sageEpisodic
    ].slice(-50); // keep recent 50 merged episodes

    return {
      embedding: mergedEmbedding,
      episodic: episodes,
      narrative: this.mergeNarratives(
        aligned.primeNarrative,
        aligned.sageNarrative
      )
    };
  }

  mergeNarratives(primeNarr: any, sageNarr: any): any | null {
    if (!primeNarr && !sageNarr) return null;
    if (!primeNarr) return sageNarr;
    if (!sageNarr) return primeNarr;

    return {
      combined: `${primeNarr.text ?? primeNarr} | ${sageNarr.text ?? sageNarr}`,
      confidence:
        ((primeNarr.confidence ?? 0.5) + (sageNarr.confidence ?? 0.5)) / 2
    };
  }

  detectDrift(aligned: AlignedRepresentations): number {
    if (!aligned.primeEmbedding || !aligned.sageEmbedding) return 0;
    if (aligned.primeEmbedding.length === 0 || aligned.sageEmbedding.length === 0) return 0;

    let dist = 0;
    const len = Math.min(
      aligned.primeEmbedding.length,
      aligned.sageEmbedding.length
    );

    for (let i = 0; i < len; i++) {
      const a = aligned.primeEmbedding[i];
      const b = aligned.sageEmbedding[i];
      dist += Math.abs(a - b);
    }

    return dist / len;
  }
}

