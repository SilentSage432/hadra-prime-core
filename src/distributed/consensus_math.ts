// src/distributed/consensus_math.ts

export class ConsensusMath {
  /**
   * Computes similarity score between two vectors (0 to 1)
   */
  static cosineSimilarity(a: number[], b: number[]): number {
    if (a.length !== b.length) return 0;

    const dot = a.reduce((s, v, i) => s + v * b[i], 0);
    const magA = Math.sqrt(a.reduce((s, v) => s + v * v, 0));
    const magB = Math.sqrt(b.reduce((s, v) => s + v * v, 0));

    if (magA === 0 || magB === 0) return 0;

    return dot / (magA * magB);
  }

  /**
   * Computes a consensus vector (simple weighted average for now)
   */
  static consensusVector(vectors: number[][]): number[] {
    if (vectors.length === 0) return [];

    const length = vectors[0].length;
    const sum = new Array(length).fill(0);

    for (const vec of vectors) {
      for (let i = 0; i < length; i++) {
        sum[i] += vec[i];
      }
    }

    return sum.map(v => v / vectors.length);
  }
}

