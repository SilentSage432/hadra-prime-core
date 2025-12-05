// src/cognition/neural/neural_memory_bank.ts
// A73: Neural Memory Bank
// A77: Knowledge Graph Integration

import { cosineSimilarity } from "./similarity.ts";
import { Knowledge } from "../knowledge/knowledge_graph.ts";

export interface NeuralMemoryEntry {
  id?: string; // A75: Unique identifier for concept clustering
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

    // A75: Ensure entry has an ID for concept clustering
    if (!entry.id) {
      entry.id = `mem_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    this.memory.push(entry);
    this.enforceSizeLimit();

    // A77: Add memory to knowledge graph
    Knowledge.addNode(entry.id, "memory", entry, entry.weight);

    // A77: Link similar memories (check against recent memories for performance)
    const recentMemories = this.memory.slice(-50); // Check last 50 memories
    for (const other of recentMemories) {
      if (other.id === entry.id) continue; // Skip self
      
      const sim = cosineSimilarity(entry.embedding, other.embedding);
      if (sim > 0.5) {
        Knowledge.linkSimilar(entry.id, other.id, sim);
      }
    }
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

  /** A75: Export all memory entries for concept clustering */
  exportAll(): Array<{ id: string; embedding: number[] }> {
    return this.memory.map(m => ({
      id: m.id || `mem_${this.memory.indexOf(m)}`,
      embedding: m.embedding
    }));
  }
}

export const NeuralMemory = new NeuralMemoryBank();

