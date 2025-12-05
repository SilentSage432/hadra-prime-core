// src/cognition/inference/inference_engine.ts
// A78: PRIME's Inference Engine

import { Knowledge } from "../knowledge/knowledge_graph.ts";
import { Concepts } from "../concepts/concept_engine.ts";
import { Hierarchy } from "../concepts/concept_hierarchy.ts";
import { cosineSimilarity } from "../neural/similarity.ts";

export interface LatentLink {
  from: string;
  to: string;
  weight: number;
  relation: string;
}

export interface DomainInference {
  domain: any | null;
  confidence: number;
}

export interface StrategyInference {
  strategy: string;
  confidence: number;
}

export class InferenceEngine {
  /** 1. Infer latent relationships in the knowledge graph */
  deriveLatentLinks(): LatentLink[] {
    const graph = Knowledge.exportGraph();
    const nodes = graph.nodes;
    const existingEdges = new Set(
      graph.edges.map((e) => `${e.from}:${e.to}:${e.relation}`)
    );
    const newLinks: LatentLink[] = [];

    // Only check conceptual nodes for latent links
    const conceptualNodes = nodes.filter(
      (n) => n.type === "concept" || n.type === "meta" || n.type === "domain"
    );

    for (let i = 0; i < conceptualNodes.length; i++) {
      for (let j = i + 1; j < conceptualNodes.length; j++) {
        const A = conceptualNodes[i];
        const B = conceptualNodes[j];

        // Skip if edge already exists
        const edgeKey = `${A.id}:${B.id}:latent`;
        if (existingEdges.has(edgeKey) || existingEdges.has(`${B.id}:${A.id}:latent`)) {
          continue;
        }

        const sim = this.embeddingSimilarity(A, B);
        if (sim >= 0.65 && sim < 0.99) { // High similarity but not identical
          newLinks.push({
            from: A.id,
            to: B.id,
            weight: sim,
            relation: "latent",
          });

          // Add to knowledge graph
          Knowledge.addEdge(A.id, B.id, sim, "latent");
        }
      }
    }

    if (newLinks.length > 0) {
      console.log(`[PRIME-INFERENCE] Derived ${newLinks.length} latent relationships`);
    }

    return newLinks;
  }

  /** Infer similarity using prototypes where available */
  private embeddingSimilarity(A: any, B: any): number {
    // Try to get prototype from data
    const aPrototype = A.data?.prototype;
    const bPrototype = B.data?.prototype;

    if (aPrototype && bPrototype && Array.isArray(aPrototype) && Array.isArray(bPrototype)) {
      return cosineSimilarity(aPrototype, bPrototype);
    }

    // If no prototypes, return 0 (no similarity)
    return 0;
  }

  /** 2. High-level inference: which domain explains current state? */
  inferActiveDomain(currentEmbedding: number[]): DomainInference {
    if (!currentEmbedding || currentEmbedding.length === 0) {
      return { domain: null, confidence: 0 };
    }

    const domains = Hierarchy.getDomains();
    if (domains.length === 0) {
      return { domain: null, confidence: 0 };
    }

    let bestDomain = null;
    let bestScore = 0;

    for (const d of domains) {
      const metaConceptIds = d.metaConcepts;
      const similarities: number[] = [];

      // Check similarity to each meta-concept in the domain
      for (const metaId of metaConceptIds) {
        const mc = Hierarchy.getMetaConcepts().find((m) => m.id === metaId);
        if (mc && mc.prototype && Array.isArray(mc.prototype)) {
          const sim = cosineSimilarity(mc.prototype, currentEmbedding);
          if (sim > 0) {
            similarities.push(sim);
          }
        }
      }

      if (similarities.length > 0) {
        const avgSim = similarities.reduce((a, b) => a + b, 0) / similarities.length;
        
        // Weight by domain strength
        const weightedScore = avgSim * (1 + d.strength * 0.1);
        
        if (weightedScore > bestScore) {
          bestScore = weightedScore;
          bestDomain = d;
        }
      }
    }

    return {
      domain: bestDomain,
      confidence: Math.min(1.0, bestScore), // Cap confidence at 1.0
    };
  }

  /** 3. Strategy inference: choose best response pattern */
  inferStrategy(currentEmbedding: number[]): StrategyInference | null {
    if (!currentEmbedding || currentEmbedding.length === 0) {
      return null;
    }

    // Find best matching concept
    const concept = Concepts.matchConcept(currentEmbedding);
    if (!concept) {
      return null;
    }

    // Infer domain alignment
    const domainInference = this.inferActiveDomain(currentEmbedding);
    
    let strategy = `Align with concept ${concept.id}`;
    let confidence = Math.min(0.5, concept.strength * 0.05);

    if (domainInference.domain && domainInference.confidence > 0.6) {
      strategy += ` within domain ${domainInference.domain.id}`;
      confidence += domainInference.confidence * 0.2;
    }

    // Check graph connections for strategy hints
    const graph = Knowledge.exportGraph();
    const conceptEdges = graph.edges.filter(
      (e) => e.from === concept.id || e.to === concept.id
    );
    
    if (conceptEdges.length > 3) {
      strategy += ` (strongly connected in knowledge graph)`;
      confidence += 0.1;
    }

    return {
      strategy: strategy,
      confidence: Math.min(1.0, confidence),
    };
  }

  /** 4. Explain why stability changed (diagnostic inference) */
  inferStabilityChange(
    beforeEmbedding: number[],
    afterEmbedding: number[]
  ): string | null {
    if (!beforeEmbedding || !afterEmbedding) return null;

    // Find concepts for before and after
    const beforeConcept = Concepts.matchConcept(beforeEmbedding);
    const afterConcept = Concepts.matchConcept(afterEmbedding);

    if (beforeConcept && afterConcept && beforeConcept.id !== afterConcept.id) {
      return `Concept shift: ${beforeConcept.id} → ${afterConcept.id}`;
    }

    // Check domain changes
    const beforeDomain = this.inferActiveDomain(beforeEmbedding);
    const afterDomain = this.inferActiveDomain(afterEmbedding);

    if (
      beforeDomain.domain &&
      afterDomain.domain &&
      beforeDomain.domain.id !== afterDomain.domain.id
    ) {
      return `Domain shift: ${beforeDomain.domain.id} → ${afterDomain.domain.id}`;
    }

    return null;
  }
}

export const Inference = new InferenceEngine();

