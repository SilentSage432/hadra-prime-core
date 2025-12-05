// src/cognition/self/homeostasis_engine.ts
// A88: Cognitive Homeostasis System
// The system that keeps PRIME stable, balanced, and sane as her complexity increases.

export class HomeostasisEngine {
  idealRanges = {
    curiosity: { min: 0.25, max: 0.55 },
    claritySeeking: { min: 0.05, max: 0.35 },
    consolidation: { min: 0.40, max: 0.75 },
    stabilityPressure: { min: 0.0, max: 0.25 }
  };

  regulate(motivations: any) {
    const updated = { ...motivations };

    for (const key of Object.keys(this.idealRanges)) {
      const { min, max } = this.idealRanges[key];
      const current = motivations[key] ?? 0;

      if (current < min) {
        updated[key] = current + (min - current) * 0.05; // nudge upward
      } else if (current > max) {
        updated[key] = current - (current - max) * 0.05; // nudge downward
      }
    }

    return updated;
  }
}

export const Homeostasis = new HomeostasisEngine();

