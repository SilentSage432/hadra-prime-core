// src/cognition/neural/neural_memory_bank.ts
// A73: Neural Memory Bank

import { cosineSimilarity } from "./similarity.ts";

export interface NeuralMemoryEntry {
  embedding: number[];
  tag: string;
  weight: number;
  timestamp: number;
  metadata?: Record<string, any>;
}

export class NeuralMemoryBank {
  private memory: NeuralMemoryEntry[] = [];
  private decayRate = 0.00005;   // gradual decay over time
  private maxSize = 5000;        // prevent runaway memory growth

  addMemory(entry: NeuralMemoryEntry) {
    if (!entry.embedding || entry.embedding.length === 0) return;

    this.memory.push(entry);
    this.enforceSizeLimit();
  }

  private enforceSizeLimit() {
    if (this.memory.length > this.maxSize) {
      this.memory = this.memory.sort((a, b) => b.weight - a.weight).slice(0, this.maxSize);
    }
  }

  /** Find N closest memories using cosine similarity */
  findClosest(embedding: number[], limit = 10) {
    return this.memory
      .map((m) => ({
        entry: m,
        similarity: cosineSimilarity(embedding, m.embedding),
      }))
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, limit);
  }

  /** Reinforcement learning adjustment */
  reinforce(embedding: number[], delta: number) {
    for (const mem of this.memory) {
      const sim = cosineSimilarity(embedding, mem.embedding);
      if (sim > 0.85) {
        mem.weight += delta * sim;
      }
    }
  }

  /** Apply time-based decay */
  decay() {
    const now = Date.now();
    for (const mem of this.memory) {
      const age = now - mem.timestamp;
      mem.weight *= Math.exp(-this.decayRate * age);
    }
  }

  getSnapshot() {
    return {
      count: this.memory.length,
      avgWeight:
        this.memory.reduce((a, b) => a + b.weight, 0) / (this.memory.length || 1),
    };
  }
}

export const NeuralMemory = new NeuralMemoryBank();

