// src/context/context_manager.ts

import { ContextLattice } from "./context_lattice.ts";
import { MemoryStore } from "../memory/memory_store.ts";

export class ContextManager {
  private lattice = new ContextLattice();
  private memory: MemoryStore;

  constructor(memory: MemoryStore) {
    this.memory = memory;
  }

  update(label: string, type: string, value: any, ttl?: number) {
    this.lattice.addContext(label, {
      type,
      value,
      ttl,
      timestamp: Date.now(),
    });

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

