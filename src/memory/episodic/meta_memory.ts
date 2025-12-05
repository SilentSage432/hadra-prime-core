// src/memory/episodic/meta_memory.ts
// A68: Meta-Memory Layer

import type { Episode } from "./episode_builder.ts";
import type { ReinforcementSignal } from "./reinforcement_engine.ts";

export interface EpisodeEmbedding {
  id: string;
  vector: number[];
  reinforcement: ReinforcementSignal;
  title: string;
  timestamp: number;
}

export class MetaMemory {
  private embeddings: EpisodeEmbedding[] = [];

  createEmbedding(episode: Episode, reinforcement: ReinforcementSignal) {
    const vector = [
      reinforcement.clarityDelta,
      reinforcement.consolidationDelta,
      reinforcement.stabilityDelta,
      reinforcement.predictionSuppressionScore
    ];

    const embedding: EpisodeEmbedding = {
      id: episode.id,
      vector,
      reinforcement,
      title: episode.title,
      timestamp: episode.startedAt
    };

    this.embeddings.push(embedding);
    if (this.embeddings.length > 5000) this.embeddings.shift();

    return embedding;
  }

  getEmbeddings() {
    return [...this.embeddings];
  }

  findSimilar(vector: number[], threshold = 0.8) {
    return this.embeddings.filter(e => {
      const dot =
        e.vector[0] * vector[0] +
        e.vector[1] * vector[1] +
        e.vector[2] * vector[2] +
        e.vector[3] * vector[3];

      const magA = Math.sqrt(
        e.vector.reduce((s, x) => s + x * x, 0)
      );
      const magB = Math.sqrt(vector.reduce((s, x) => s + x * x, 0));

      const sim = dot / (magA * magB || 1);
      return sim >= threshold;
    });
  }
}

