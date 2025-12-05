// src/memory/neural/neural_memory_store.ts
// A93: Neural Memory Encoding (Embedding → Memory Storage)
// PRIME gains the ability to remember neurally — not just symbolically.

export class NeuralMemoryStore {
  private store: {
    id: string;
    embedding: number[];
    timestamp: number;
    tags: string[];
  }[] = [];

  saveEmbedding(id: string, embedding: number[], tags: string[] = []) {
    this.store.push({
      id,
      embedding,
      timestamp: Date.now(),
      tags
    });
  }

  // Simple cosine similarity for now
  private cosine(a: number[], b: number[]) {
    const dot = a.reduce((s, v, i) => s + v * b[i], 0);
    const magA = Math.sqrt(a.reduce((s, v) => s + v * v, 0));
    const magB = Math.sqrt(b.reduce((s, v) => s + v * v, 0));
    return dot / (magA * magB + 1e-9);
  }

  searchSimilar(embedding: number[], limit = 5) {
    return this.store
      .map(m => ({
        ...m,
        score: this.cosine(embedding, m.embedding)
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, limit);
  }

  getAll() {
    return this.store;
  }
}

