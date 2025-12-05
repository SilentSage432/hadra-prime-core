// src/memory/concepts/concept_store.ts
// A95: Concept Store
// A96: Concept Drift Tracking
// A99: Hierarchical Knowledge Architecture
// A structure where concepts live, grow, get refined, and merge.

export interface ConceptNode {
  id: string;
  label: string;
  centroid: number[];
  members: string[];     // memory IDs
  stability: number;     // 0–1
  confidence: number;    // 0–1
  drift: number;         // how fast centroid adjusts
  createdAt: number;
  lastUpdated: number;
  decayRate: number;     // how fast stability fades when unused
  layer?: number;        // A99: hierarchical layer (0-5)
  parents?: string[];    // A99: higher-layer concept IDs
  children?: string[];   // A99: lower-layer concept IDs
  importanceScore?: number;  // A99: computed importance
  hierarchyConfidence?: number;  // A99: confidence in hierarchy placement
  attentionScore?: number;  // A100: computed attention score
  isFocused?: boolean;  // A100: whether concept is in focus
  isSuppressed?: boolean;  // A100: whether concept is suppressed
  predictionWeight?: number;  // A100: predictive relevance weight
}

export const ConceptGraph: ConceptNode[] = [];

export function createConcept(label: string, vec: number[], memoryId: string): ConceptNode {
  const node: ConceptNode = {
    id: `concept-${Date.now()}-${Math.floor(Math.random()*99999)}`,
    label,
    centroid: vec,
    members: [memoryId],
    stability: 0.1,
    confidence: 0.2,
    drift: 0.15,
    createdAt: Date.now(),
    lastUpdated: Date.now(),
    decayRate: 0.0005,
    // A99: Initialize hierarchical fields
    layer: undefined,
    parents: [],
    children: [],
    importanceScore: 0,
    hierarchyConfidence: 0
  };

  ConceptGraph.push(node);
  return node;
}

export function updateConcept(node: ConceptNode, vec: number[], memoryId: string) {
  // Move centroid toward vec
  node.centroid = node.centroid.map((c, i) => c + (vec[i] - c) * node.drift);
  node.members.push(memoryId);
  
  // Update timestamp
  node.lastUpdated = Date.now();
  
  // Increase confidence with exposure
  node.confidence = Math.min(1, node.confidence + 0.02);
  
  // Increase stability over time
  node.stability = Math.min(1, node.stability + 0.01);
  
  // Slightly reduce decay rate as concept stabilizes
  node.decayRate = Math.max(0.0001, node.decayRate * 0.98);
  
  return node;
}

