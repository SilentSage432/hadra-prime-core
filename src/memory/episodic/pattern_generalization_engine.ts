// src/memory/episodic/pattern_generalization_engine.ts
// A69: Pattern Generalization Engine

import type { EpisodeEmbedding } from "./meta_memory.ts";
import crypto from "crypto";

export interface PatternCluster {
  id: string;
  embeddings: EpisodeEmbedding[];
  signature: number[]; // averaged feature vector
}

export class PatternGeneralizationEngine {
  private clusters: PatternCluster[] = [];
  private similarityThreshold = 0.75;

  // cosine similarity
  similarity(a: number[], b: number[]): number {
    const dot = a.reduce((s, x, i) => s + x * b[i], 0);
    const magA = Math.sqrt(a.reduce((s, x) => s + x * x, 0));
    const magB = Math.sqrt(b.reduce((s, x) => s + x * x, 0));
    return dot / ((magA * magB) || 1);
  }

  // Update a cluster's signature vector
  private updateSignature(cluster: PatternCluster) {
    if (cluster.embeddings.length === 0) return;

    const dim = cluster.embeddings[0].vector.length;
    const sum = new Array(dim).fill(0);

    for (const e of cluster.embeddings) {
      for (let i = 0; i < dim; i++) {
        sum[i] += e.vector[i];
      }
    }

    cluster.signature = sum.map(v => v / cluster.embeddings.length);
  }

  // Add new embedding to a cluster or create a new one
  addEmbedding(embedding: EpisodeEmbedding) {
    for (const cluster of this.clusters) {
      const sim = this.similarity(cluster.signature, embedding.vector);
      if (sim >= this.similarityThreshold) {
        cluster.embeddings.push(embedding);
        this.updateSignature(cluster);
        return {
          clusterId: cluster.id,
          similarity: sim,
          createdNewCluster: false
        };
      }
    }

    const newCluster: PatternCluster = {
      id: crypto.randomUUID(),
      embeddings: [embedding],
      signature: embedding.vector
    };

    this.clusters.push(newCluster);

    return {
      clusterId: newCluster.id,
      similarity: 1,
      createdNewCluster: true
    };
  }

  listClusters() {
    return [...this.clusters];
  }

  getClusterSignature(clusterId: string) {
    return this.clusters.find(c => c.id === clusterId)?.signature || null;
  }

  getCluster(clusterId: string) {
    return this.clusters.find(c => c.id === clusterId) || null;
  }
}

