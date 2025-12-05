// src/memory/concepts/concept_merge_split_engine.ts
// A97: Concept Merging & Splitting Engine
// PRIME learns how to reorganize its own conceptual universe

import { ConceptGraph, createConcept } from "./concept_store.ts";
import { cosineSimilarity } from "../../cognition/neural/similarity.ts";
import { kMeans } from "../embedding/kmeans.ts";
import { getNeuralMemory } from "../../kernel/index.ts";

export class ConceptMergeSplitEngine {
  mergeThreshold = 0.92;
  overlapThreshold = 0.6;
  splitVarianceThreshold = 0.15;

  tick() {
    this.handleMerges();
    this.handleSplits();
  }

  handleMerges() {
    const concepts = [...ConceptGraph];
    
    for (let i = 0; i < concepts.length; i++) {
      for (let j = i + 1; j < concepts.length; j++) {
        const c1 = concepts[i];
        const c2 = concepts[j];
        
        // Skip if either concept was already merged
        if (!ConceptGraph.includes(c1) || !ConceptGraph.includes(c2)) continue;
        
        const sim = cosineSimilarity(c1.centroid, c2.centroid);
        if (sim < this.mergeThreshold) continue;

        const overlap = this.computeMemoryOverlap(c1, c2);
        if (overlap < this.overlapThreshold) continue;

        console.log(`[PRIME-CONCEPTS] Merging '${c1.label}' + '${c2.label}' (sim=${sim.toFixed(3)}, overlap=${overlap.toFixed(3)})`);
        this.mergeConcepts(c1, c2);
      }
    }
  }

  computeMemoryOverlap(c1: any, c2: any): number {
    const set1 = new Set(c1.members);
    const set2 = new Set(c2.members);
    const intersection = [...set1].filter(x => set2.has(x));
    return intersection.length / Math.min(set1.size, set2.size);
  }

  mergeConcepts(c1: any, c2: any) {
    // Merge memory refs (members)
    c1.members = Array.from(new Set([...c1.members, ...c2.members]));

    // Recompute centroid as average
    for (let i = 0; i < c1.centroid.length; i++) {
      c1.centroid[i] = (c1.centroid[i] + c2.centroid[i]) / 2;
    }

    // Stability and confidence adjust
    c1.stability = Math.min(1, (c1.stability + c2.stability) / 2 + 0.1);
    c1.confidence = Math.min(1, (c1.confidence + c2.confidence) / 2);
    
    // Update label to reflect merge
    c1.label = `${c1.label}_${c2.label}`;
    
    // Update timestamp
    c1.lastUpdated = Date.now();
    
    // Reset drift rate
    c1.decayRate = 0.0005;

    // Remove c2 from graph
    const index = ConceptGraph.indexOf(c2);
    if (index > -1) {
      ConceptGraph.splice(index, 1);
    }
  }

  handleSplits() {
    const concepts = [...ConceptGraph]; // Copy to avoid modification during iteration
    
    for (const c of concepts) {
      // Skip if concept was already removed
      if (!ConceptGraph.includes(c)) continue;
      
      const variance = this.estimateVariance(c.centroid);
      if (variance < this.splitVarianceThreshold) continue;

      // Check if we have enough members to split
      if (c.members.length < 4) continue;

      console.log(`[PRIME-CONCEPTS] Splitting concept '${c.label}' — variance=${variance.toFixed(3)}`);
      this.splitConcept(c);
    }
  }

  estimateVariance(vec: number[]): number {
    const mean = vec.reduce((a, b) => a + b, 0) / vec.length;
    const variance = vec.reduce((s, v) => s + Math.pow(v - mean, 2), 0) / vec.length;
    return variance;
  }

  splitConcept(c: any) {
    // Get embeddings for memories
    const neuralMem = getNeuralMemory();
    if (!neuralMem) return;

    const embeddings: number[][] = [];
    const memberIndices: number[] = [];

    // Find embeddings for concept members
    for (let i = 0; i < c.members.length; i++) {
      const memId = c.members[i];
      const memory = neuralMem.getById(memId);
      if (memory && memory.embedding && memory.embedding.length > 0) {
        embeddings.push(memory.embedding);
        memberIndices.push(i);
      }
    }

    if (embeddings.length < 4) return;

    // Perform k-means clustering
    const result = kMeans(embeddings, 2, 6);
    
    if (result.centroids.length < 2 || result.clusters.length !== embeddings.length) return;

    // Split members into two groups
    const group1: string[] = [];
    const group2: string[] = [];

    for (let i = 0; i < memberIndices.length; i++) {
      const memId = c.members[memberIndices[i]];
      if (result.clusters[i] === 0) {
        group1.push(memId);
      } else {
        group2.push(memId);
      }
    }

    // Create two new concepts
    if (group1.length > 0 && group2.length > 0) {
      const newConcept1 = createConcept(
        `${c.label}_a`,
        result.centroids[0],
        group1[0]
      );
      newConcept1.members = group1;
      newConcept1.stability = c.stability * 0.8;
      newConcept1.confidence = c.confidence * 0.8;

      const newConcept2 = createConcept(
        `${c.label}_b`,
        result.centroids[1],
        group2[0]
      );
      newConcept2.members = group2;
      newConcept2.stability = c.stability * 0.8;
      newConcept2.confidence = c.confidence * 0.8;

      // Remove original concept
      const index = ConceptGraph.indexOf(c);
      if (index > -1) {
        ConceptGraph.splice(index, 1);
      }

      console.log(`[PRIME-CONCEPTS] Split '${c.label}' into '${newConcept1.label}' (${group1.length} members) and '${newConcept2.label}' (${group2.length} members)`);
    }
  }
}

