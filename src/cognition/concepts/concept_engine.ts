// src/cognition/concepts/concept_engine.ts
// A75: Neural Abstraction Layer (Concept Formation Engine)
// A76: Hierarchical Concept Networks
// A77: Knowledge Graph Integration

import { NeuralMemory } from "../neural/neural_memory_bank.ts";
import { cosineSimilarity } from "../neural/similarity.ts";
import { Hierarchy } from "./concept_hierarchy.ts";
import { Knowledge } from "../knowledge/knowledge_graph.ts";
import crypto from "crypto";

export interface Concept {
  id: string;
  prototype: number[];
  members: string[];
  label?: string;
  strength: number;
}

export class ConceptEngine {
  private concepts: Concept[] = [];

  /** Cluster memories into concepts */
  deriveConcepts(threshold = 0.80) {
    const entries = NeuralMemory.exportAll();
    
    if (entries.length < 2) {
      this.concepts = [];
      return;
    }

    const embeddings = entries.map((e) => e.embedding);
    const used = new Set<number>();
    const newConcepts: Concept[] = [];

    // Iterate through all entries to find clusters
    for (let i = 0; i < embeddings.length; i++) {
      if (used.has(i)) continue;

      const clusterMembers = [entries[i]];
      used.add(i);

      // Find all similar entries
      for (let j = i + 1; j < embeddings.length; j++) {
        if (used.has(j)) continue;

        const sim = cosineSimilarity(embeddings[i], embeddings[j]);
        if (sim >= threshold) {
          clusterMembers.push(entries[j]);
          used.add(j);
        }
      }

      // Only create concept if cluster has multiple members
      if (clusterMembers.length > 1) {
        const prototype = this.computePrototype(clusterMembers);
        const conceptId = crypto.randomUUID();
        
        newConcepts.push({
          id: conceptId,
          prototype,
          members: clusterMembers.map((m) => m.id),
          strength: clusterMembers.length,
        });

        console.log(`[PRIME-CONCEPT] Derived concept ${conceptId}: ${clusterMembers.length} members`);

        // A77: Add concept to knowledge graph
        Knowledge.addNode(conceptId, "concept", {
          prototype,
          members: clusterMembers.map((m) => m.id),
          strength: clusterMembers.length,
        }, clusterMembers.length);

        // A77: Link concept to its member memories
        clusterMembers.forEach((member) => {
          Knowledge.linkContainment(conceptId, member.id);
        });
      }
    }

    this.concepts = newConcepts;
    console.log(`[PRIME-CONCEPT] Total concepts: ${this.concepts.length}`);

    // A77: Link similar concepts in knowledge graph
    for (let i = 0; i < newConcepts.length; i++) {
      for (let j = i + 1; j < newConcepts.length; j++) {
        const sim = cosineSimilarity(
          newConcepts[i].prototype,
          newConcepts[j].prototype
        );
        Knowledge.linkSimilar(newConcepts[i].id, newConcepts[j].id, sim);
      }
    }

    // A76: Build meta-concepts and domains after concepts form
    if (this.concepts.length > 1) {
      Hierarchy.buildMetaConcepts(this.concepts);
      Hierarchy.buildDomains();
    }
  }

  computePrototype(members: Array<{ embedding: number[] }>) {
    if (members.length === 0) return [];
    
    const len = members[0].embedding.length;
    const proto = new Array(len).fill(0);

    for (const m of members) {
      for (let i = 0; i < len; i++) {
        proto[i] += m.embedding[i];
      }
    }

    // Average the embeddings
    for (let i = 0; i < len; i++) {
      proto[i] /= members.length;
    }

    return proto;
  }

  getConcepts(): Concept[] {
    return [...this.concepts];
  }

  /** Find which concept current embedding fits best */
  matchConcept(embedding: number[]): Concept | null {
    if (this.concepts.length === 0) return null;

    let best: Concept | null = null;
    let bestScore = 0;

    for (const c of this.concepts) {
      const sim = cosineSimilarity(embedding, c.prototype);
      if (sim > bestScore && sim >= 0.75) { // Minimum threshold for concept match
        bestScore = sim;
        best = c;
      }
    }

    return best;
  }

  /** Get concept by ID */
  getConcept(id: string): Concept | null {
    return this.concepts.find(c => c.id === id) || null;
  }
}

export const Concepts = new ConceptEngine();

