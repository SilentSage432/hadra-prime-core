// src/memory/concepts/concept_store.ts
// A95: Concept Store
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
    createdAt: Date.now()
  };

  ConceptGraph.push(node);
  return node;
}

export function updateConcept(node: ConceptNode, vec: number[], memoryId: string) {
  // Move centroid toward vec
  node.centroid = node.centroid.map((c, i) => c + (vec[i] - c) * node.drift);
  node.members.push(memoryId);
  
  // Increase confidence with exposure
  node.confidence = Math.min(1, node.confidence + 0.02);
  node.stability = Math.min(1, node.stability + 0.01);
  
  return node;
}

