// src/temporal/window.ts
// A66: Temporal Reasoning Window

export interface TemporalSnapshot {
  t: number; // timestamp
  clarity: number;
  consolidation: number;
  curiosity: number;
  stability: number;
}

export class TemporalWindow {
  private window: TemporalSnapshot[] = [];
  private maxEntries: number;

  constructor(maxEntries = 300) { // ~5 minutes at 1 snapshot/sec (event-driven)
    this.maxEntries = maxEntries;
  }

  record(snapshot: TemporalSnapshot) {
    this.window.push(snapshot);
    if (this.window.length > this.maxEntries) {
      this.window.shift();
    }
  }

  getAll() {
    return [...this.window];
  }

  getDelta() {
    if (this.window.length < 2) return null;

    const first = this.window[0];
    const last = this.window[this.window.length - 1];

    return {
      clarityDelta: last.clarity - first.clarity,
      consolidationDelta: last.consolidation - first.consolidation,
      curiosityDelta: last.curiosity - first.curiosity,
      stabilityDelta: last.stability - first.stability,
      durationMs: last.t - first.t,
    };
  }
}

