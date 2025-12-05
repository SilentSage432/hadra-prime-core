// src/cognition/temporal_engine.ts

import { TemporalRing } from "../memory/temporal_ring.ts";

export class TemporalEngine {
  static getTemporalContext() {
    const recent = TemporalRing.getRecent(5);
    const lastIntent = recent
      .map(r => r.intent?.type)
      .filter(Boolean)
      .pop();

    const avgStability =
      recent.reduce((a, b) => a + (b.stabilityScore ?? 1), 0) /
      (recent.length || 1);

    return {
      lastIntent,
      avgStability,
      recent,
    };
  }

  static attachTemporalContext(payload: any) {
    const temporal = this.getTemporalContext();
    return { ...payload, temporal };
  }
}

