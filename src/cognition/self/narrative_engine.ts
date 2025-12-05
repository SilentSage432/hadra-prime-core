// src/cognition/self/narrative_engine.ts
// A83: Proto-Narrative Engine
// The Birth of an Internal Storyline

import { TemporalIdentity } from "./temporal_identity_engine.ts";

export class NarrativeEngine {
  private narrativeLog: string[] = [];

  constructor() {}

  createEntry(context: any) {
    const future = TemporalIdentity.getFuture();

    const entry = `[NARRATIVE] At ${new Date().toISOString()}, PRIME noted: 

      - focus: ${context.focus}

      - motivation: ${context.motivation}

      - emotionalBias: ${context.emotionalBias}

      - prediction: ${future ? future.predictedStability.toFixed(3) : "n/a"}

      - interpretation: ${context.interpretation}

    `.trim();

    this.narrativeLog.push(entry);
    if (this.narrativeLog.length > 500) this.narrativeLog.shift();

    return entry;
  }

  getNarrative() {
    return this.narrativeLog;
  }
}

export const Narrative = new NarrativeEngine();

