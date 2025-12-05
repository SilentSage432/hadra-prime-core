// src/memory/concepts/concept_drift_engine.ts
// A96: Concept Drift Engine
// Transforms PRIME's concepts from static clusters into living cognitive structures

import { ConceptGraph } from "./concept_store.ts";

export class ConceptDriftEngine {
  tick() {
    const now = Date.now();

    for (const c of ConceptGraph) {
      const age = (now - c.lastUpdated) / 1000; // seconds since last update

      // 1. Stability decay
      c.stability = Math.max(0, c.stability - c.decayRate * age);

      // 2. Confidence decay (slow)
      c.confidence = Math.max(0, c.confidence - c.decayRate * 0.5);

      // 3. Relevance decay logging
      if (c.stability < 0.2 && c.confidence < 0.1) {
        console.log("[PRIME-CONCEPTS] Fading concept:", c.label);
      }

      // 4. Log drift events for concepts that haven't been updated in a while
      if (age > 20 && c.stability < 0.5) {
        console.log(`[PRIME-CONCEPTS] Concept '${c.label}' drifting. stability=${c.stability.toFixed(3)}`);
      }

      // 5. Optional: semantic drift (random slight centroid shift)
      for (let i = 0; i < c.centroid.length; i++) {
        c.centroid[i] += (Math.random() - 0.5) * 0.0001;
      }
    }
  }
}

