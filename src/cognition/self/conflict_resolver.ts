// src/cognition/self/conflict_resolver.ts
// A86: Internal Conflict Resolution Engine
// When the council disagrees, PRIME learns to resolve cognitive dissonance.

export class ConflictResolver {
  computeDissonance(voices: any[]) {
    // Compare opinions by semantic difference (light heuristic for now)
    let dissonance = 0;

    for (let i = 0; i < voices.length; i++) {
      for (let j = i + 1; j < voices.length; j++) {
        const diff = this.estimateDifference(voices[i].opinion, voices[j].opinion);
        dissonance += diff;
      }
    }

    return Math.min(1, dissonance / 10); // normalize 0–1
  }

  estimateDifference(opA: string, opB: string) {
    // crude heuristic: count differing important words
    const keywords = ["continue", "cautious", "explore", "deviate", "ground", "align"];
    let diff = 0;

    for (const k of keywords) {
      const a = opA.includes(k);
      const b = opB.includes(k);
      if (a !== b) diff += 1;
    }

    return diff;
  }

  resolveConflict(voices: any[], context: any) {
    const dissonance = this.computeDissonance(voices);

    if (dissonance < 0.15) {
      return {
        dissonance,
        result: "low_conflict",
        strategy: "Proceed as planned. Voices are aligned."
      };
    }

    if (dissonance < 0.35) {
      return {
        dissonance,
        result: "moderate_conflict",
        strategy: "Reweigh motivations. Increase claritySeeking slightly."
      };
    }

    if (dissonance < 0.6) {
      return {
        dissonance,
        result: "substantial_conflict",
        strategy: "Trigger a reflection cycle and re-evaluate goal alignment."
      };
    }

    return {
      dissonance,
      result: "high_conflict",
      strategy: "Override decision. Select safest high-weight voice and pause exploratory impulses."
    };
  }
}

export const ConflictEngine = new ConflictResolver();

