// src/context/context_manager.ts

import { ContextLattice } from "./context_lattice.ts";
import { MemoryStore } from "../memory/memory_store.ts";

export class ContextManager {
  private lattice = new ContextLattice();
  private memory: MemoryStore;

  constructor(memory: MemoryStore) {
    this.memory = memory;
  }

  update(label: string, type: string, value: any, ttl?: number, neuralCoherence?: any) {
    const nodeData: any = {
      type,
      value,
      ttl,
      timestamp: Date.now(),
    };

    // A113: Bind neural coherence to context node
    if (neuralCoherence) {
      nodeData.neuralRelevance = neuralCoherence.relevance;
      nodeData.neuralTags = neuralCoherence.semanticTags;
      nodeData.neuralAnchors = neuralCoherence.contextAnchors;
      console.log("[PRIME-CONTEXT] Context node updated with neural anchors:", {
        label,
        relevance: neuralCoherence.relevance.toFixed(3),
        tags: neuralCoherence.semanticTags
      });
    }

    this.lattice.addContext(label, nodeData);

    // Also store important context to memory
    this.memory.logInteraction("context", { label, type, value });
  }

  get(label: string) {
    return this.lattice.getContext(label);
  }

  latest(label: string) {
    return this.lattice.latest(label);
  }

  // For PRIME: query continuity (e.g., "what were we doing?")
  summarize(label: string) {
    const context = this.get(label);
    return context.map(c => `${c.type}: ${JSON.stringify(c.value)}`);
  }
}

