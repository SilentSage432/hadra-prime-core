// src/dual_core/shared_context_seed.ts
// A104b: Shared Context Seed
// Defines the "bridge memory" PRIME + SAGE share during joint cognition

export interface SharedContextSeed {
  version: string;
  primeFocus: string | null;
  sageFocus: string | null;
  lastSync: number;
}

export class SharedContextSeedBuilder {
  static create(primeFocus: string | null, sageFocus: string | null): SharedContextSeed {
    return {
      version: "1.0.0",
      primeFocus,
      sageFocus,
      lastSync: Date.now()
    };
  }
}

