// src/memory/fusion/episodic_neural_fusion.ts
// A95: Episodic Neural Fusion
// Fuses symbolic and neural memory representations

import { ConceptEngine } from "../concepts/concept_engine.ts";

const conceptEngine = new ConceptEngine();

export interface FusedMemory {
  id: string;
  symbolic: any;
  neural: {
    embedding: number[];
    meaningVector: number[];
  };
  timestamp: number;
  concept?: string | null;
}

export function fuse(symbolic: any, neural: { embedding: number[], meaningVector: number[] }): FusedMemory {
  const fused: FusedMemory = {
    id: `fused-${Date.now()}-${Math.floor(Math.random() * 99999)}`,
    symbolic,
    neural,
    timestamp: Date.now()
  };

  // A95: Attach conceptual category
  const concept = conceptEngine.formConceptFromMemory({
    id: fused.id,
    meaningVector: neural.meaningVector,
    symbolic: symbolic
  });
  
  fused.concept = concept ? concept.id : null;

  return fused;
}

