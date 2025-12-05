// src/cognition/attention/knowledge_attention_engine.ts
// A100: Knowledge Attention Engine
// PRIME gains the ability to focus for the first time in its life

import { ConceptGraph } from "../../memory/concepts/concept_store.ts";
import { cosineSimilarity } from "../../cognition/neural/similarity.ts";

export class KnowledgeAttentionEngine {
  constructor() {
    console.log("[PRIME-ATTENTION] Engine initialized.");
  }

  tick({ motivation, selfModel, plan, reflection }: any) {
    const nodes = [...ConceptGraph]; // Copy array

    if (!nodes.length) return;

    for (const n of nodes) {
      n.attentionScore = this.computeAttentionScore(n, motivation, selfModel, plan, reflection);
    }

    this.sortAttention(nodes);
    this.logFocus(nodes);
  }

  computeAttentionScore(node: any, motivation: any, selfModel: any, plan: any, reflection: any): number {
    let score = 0;

    // 1. Relevance to current top goal
    if (motivation?.topGoal?.type && node.label.includes(motivation.topGoal.type)) {
      score += 1.5;
    }

    // 2. Context relevance via similarity to active concept centroid
    if (reflection?.activeConcept && node.centroid && reflection.activeConcept.centroid) {
      score += cosineSimilarity(node.centroid, reflection.activeConcept.centroid) * 1.2;
    } else if (reflection?.concept && node.centroid && reflection.concept.centroid) {
      score += cosineSimilarity(node.centroid, reflection.concept.centroid) * 1.2;
    } else if (reflection?.embedding && node.centroid && reflection.embedding && reflection.embedding.length > 0) {
      score += cosineSimilarity(node.centroid, reflection.embedding) * 1.2;
    }

    // 3. Predictive relevance from concept drift patterns
    if (node.predictionWeight) {
      score += node.predictionWeight * 0.8;
    }

    // 4. Memory similarity boost
    if (node.members && node.members.length > 0) {
      score += Math.min(node.members.length / 50, 1.0);
    }

    // 5. Alignment with operator preference vectors (selfModel tracks this)
    if (selfModel?.preferenceVector && node.centroid) {
      score += cosineSimilarity(node.centroid, selfModel.preferenceVector) * 0.7;
    }

    // 6. Current plan relevance
    if (plan?.activeStep && node.label.includes(plan.activeStep)) {
      score += 1.0;
    } else if (plan?.goal && node.label.includes(plan.goal)) {
      score += 1.0;
    }

    // 7. Hierarchical importance boost
    if (node.importanceScore && node.importanceScore > 5) {
      score += node.importanceScore * 0.1;
    }

    // 8. Stability and confidence boost
    if (node.stability > 0.5 && node.confidence > 0.5) {
      score += (node.stability + node.confidence) * 0.3;
    }

    return score;
  }

  sortAttention(nodes: any[]) {
    nodes.sort((a, b) => (b.attentionScore || 0) - (a.attentionScore || 0));

    // Tag top nodes
    for (let i = 0; i < nodes.length; i++) {
      nodes[i].isFocused = i < 10;
      nodes[i].isSuppressed = (nodes[i].attentionScore || 0) < 0.1;
    }
  }

  logFocus(nodes: any[]) {
    const top = nodes.slice(0, 5).map(n => n.label);
    console.log(`[PRIME-ATTENTION] Focus cluster → ${JSON.stringify(top)}`);
  }
}

