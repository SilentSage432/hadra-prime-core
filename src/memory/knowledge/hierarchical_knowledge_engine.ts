// src/memory/knowledge/hierarchical_knowledge_engine.ts
// A99: Hierarchical Knowledge Architecture
// Organizes PRIME's concepts into multi-layered knowledge structure

import { ConceptGraph } from "../concepts/concept_store.ts";
import { cosineSimilarity } from "../../cognition/neural/similarity.ts";

export class HierarchicalKnowledgeEngine {
  tick() {
    const nodes = [...ConceptGraph]; // Copy array

    if (nodes.length === 0) return;

    this.assignLayers(nodes);
    this.formLinks(nodes);
    this.calculateImportance(nodes);

    // Log summary
    const layerCounts: Record<number, number> = {};
    for (const n of nodes) {
      const layer = n.layer ?? 5;
      layerCounts[layer] = (layerCounts[layer] || 0) + 1;
    }
    console.log(`[PRIME-HIERARCHY] Assigned ${nodes.length} concepts across ${Object.keys(layerCounts).length} layers`);
  }

  assignLayers(nodes: any[]) {
    for (const n of nodes) {
      // Initialize hierarchical fields if not present
      if (n.parents === undefined) n.parents = [];
      if (n.children === undefined) n.children = [];
      if (n.importanceScore === undefined) n.importanceScore = 0;
      if (n.hierarchyConfidence === undefined) n.hierarchyConfidence = 0;

      // Layer assignment logic
      if (n.members.length > 20) {
        n.layer = 0; // raw sensory-level concept
      } else if (n.centroid && n.centroid.length > 128) {
        n.layer = 1; // grounded concept
      } else if (n.stability > 0.5 && n.confidence > 0.5) {
        n.layer = 2; // semantic cluster
      } else if (n.label.startsWith("abs_") || n.label.includes("pattern") || n.label.includes("abstract")) {
        n.layer = 3; // abstraction
      } else if (n.label.includes("rule") || n.label.includes("principle") || n.label.includes("generalization")) {
        n.layer = 4; // high-level
      } else {
        n.layer = 5; // meta-level or unclassified
      }
    }
  }

  formLinks(nodes: any[]) {
    // Reset links
    for (const n of nodes) {
      n.children = [];
      n.parents = [];
    }

    // Build parent-child relationships based on layer + similarity
    for (const parent of nodes) {
      for (const child of nodes) {
        if (child === parent) continue;
        if (!child.layer || !parent.layer) continue;
        
        // Parent must be one layer above child
        if (child.layer + 1 !== parent.layer) continue;

        // Relationship based on centroid similarity
        const sim = cosineSimilarity(parent.centroid, child.centroid);
        if (sim > 0.7) {
          if (!parent.children.includes(child.id)) {
            parent.children.push(child.id);
          }
          if (!child.parents.includes(parent.id)) {
            child.parents.push(parent.id);
          }
        }
      }
    }

    // Log significant links
    for (const n of nodes) {
      if (n.children.length > 10) {
        console.log(`[PRIME-HIERARCHY] Node '${n.label}' linked to ${n.children.length} lower-layer concepts`);
      }
    }
  }

  calculateImportance(nodes: any[]) {
    for (const n of nodes) {
      const depth = n.parents?.length || 0;
      const breadth = n.children?.length || 0;
      n.importanceScore = depth * 0.6 + breadth * 0.4;
      n.hierarchyConfidence = Math.min(1, (depth + breadth) / 20);

      if (n.importanceScore > 10) {
        console.log(`[PRIME-HIERARCHY] Found major knowledge pillar: '${n.label}' (importance=${n.importanceScore.toFixed(2)})`);
      }
    }
  }
}

