// src/memory/concepts/semantic_compression_engine.ts
// A98: Semantic Compression Engine
// PRIME learns to compress meaning itself - turning 100 experiences into 1 principle

import { ConceptGraph, createConcept } from "./concept_store.ts";
import { cosineSimilarity } from "../../cognition/neural/similarity.ts";

export class SemanticCompressionEngine {
  abstractionThreshold = 0.88;     // similarity required to form a general rule
  minimumClusterSize = 4;

  tick() {
    const concepts = [...ConceptGraph]; // Copy array

    if (concepts.length < this.minimumClusterSize) return;

    const clusters = this.clusterConcepts(concepts);
    this.formAbstractions(clusters);
  }

  clusterConcepts(concepts: any[]) {
    // Very simple cluster algorithm
    const clusters: Array<{ representative: any; members: any[] }> = [];

    for (const c of concepts) {
      let placed = false;

      for (const group of clusters) {
        const score = cosineSimilarity(
          c.centroid,
          group.representative.centroid
        );

        if (score > this.abstractionThreshold) {
          group.members.push(c);
          placed = true;
          break;
        }
      }

      if (!placed) {
        clusters.push({ representative: c, members: [c] });
      }
    }

    return clusters;
  }

  formAbstractions(clusters: Array<{ representative: any; members: any[] }>) {
    for (const group of clusters) {
      if (group.members.length < this.minimumClusterSize) continue;

      console.log(`[PRIME-COMPRESS] Abstraction candidate cluster found (${group.members.length} members)`);

      const abstraction = this.buildAbstraction(group.members);
      this.storeAbstraction(group.members, abstraction);
    }
  }

  buildAbstraction(members: any[]) {
    const dim = members[0].centroid.length;
    const centroid = new Array(dim).fill(0);

    for (const m of members) {
      for (let i = 0; i < dim; i++) {
        centroid[i] += m.centroid[i];
      }
    }

    for (let i = 0; i < dim; i++) {
      centroid[i] /= members.length;
    }

    // Collect all memory references from members
    const allMembers: string[] = [];
    for (const m of members) {
      allMembers.push(...m.members);
    }
    const uniqueMembers = Array.from(new Set(allMembers));

    return {
      label: this.inferAbstractionLabel(members),
      centroid,
      members: uniqueMembers,
      stability: 0.55,
      confidence: 0.65
    };
  }

  inferAbstractionLabel(members: any[]): string {
    const labels = members.map(m => m.label.toLowerCase());

    if (labels.every(l => l.includes("error") || l.includes("fault"))) {
      return "error-handling-pattern";
    }

    if (labels.every(l => l.includes("tyson"))) {
      return "tyson-profile-generalization";
    }

    return "abstract-concept";
  }

  storeAbstraction(members: any[], abstraction: any) {
    console.log(`[PRIME-COMPRESS] New abstraction created: '${abstraction.label}'`);

    // Create new abstract concept
    const firstMemberId = abstraction.members[0] || `abstract-${Date.now()}`;
    const newConcept = createConcept(
      abstraction.label,
      abstraction.centroid,
      firstMemberId
    );

    // Set all properties
    newConcept.members = abstraction.members;
    newConcept.stability = abstraction.stability;
    newConcept.confidence = abstraction.confidence;
    newConcept.lastUpdated = Date.now();

    // Reduce noise: weaken original concepts a bit
    for (const m of members) {
      m.stability *= 0.9;
      m.confidence *= 0.9;
    }
  }
}

