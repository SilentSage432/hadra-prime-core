// src/memory/concepts/concept_engine.ts
// A95: Concept Formation Engine
// Pulls from fused memory + embeddings to form clusters

import { ConceptGraph, createConcept, updateConcept } from "./concept_store.ts";
import { cosineSimilarity } from "../../cognition/neural/similarity.ts";

const SIM_THRESHOLD = 0.82;  // similarity threshold to join concept

export class ConceptEngine {
  formConceptFromMemory(memory: any) {
    if (!memory.meaningVector || memory.meaningVector.length === 0) return null;

    const vec = memory.meaningVector;

    // Look for similar existing concepts
    let bestMatch = null;
    let bestScore = 0;

    for (const c of ConceptGraph) {
      const score = cosineSimilarity(vec, c.centroid);
      if (score > SIM_THRESHOLD && score > bestScore) {
        bestMatch = c;
        bestScore = score;
      }
    }

    if (bestMatch) {
      return updateConcept(bestMatch, vec, memory.id);
    }

    // No match → Create new concept
    const label = this.generateLabel(memory);
    return createConcept(label, vec, memory.id);
  }

  generateLabel(memory: any): string {
    if (memory.symbolic?.type === "text_observation") {
      const raw = memory.symbolic.raw.toLowerCase();
      return `text:${raw.split(" ").slice(0, 2).join("_")}`;
    }
    return "emergent_concept";
  }
}

