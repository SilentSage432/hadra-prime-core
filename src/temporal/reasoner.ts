// src/temporal/reasoner.ts
// A66: Temporal Reasoning Engine

import { TemporalWindow } from "./window.ts";
import { SEL } from "../emotion/sel.ts";
import { StabilityMatrix } from "../stability/stability_matrix.ts";

export class TemporalReasoner {
  short = new TemporalWindow(30);   // ~30 sec
  medium = new TemporalWindow(300); // ~5 minutes
  long = new TemporalWindow(3600);  // ~1 hour (lightweight rolling)

  record() {
    const selState = SEL.getState();
    const stabilitySnapshot = StabilityMatrix.getSnapshot();

    const snapshot = {
      t: Date.now(),
      clarity: selState.coherence, // Use coherence as clarity proxy
      consolidation: selState.affinity, // Use affinity as consolidation proxy
      curiosity: selState.certainty, // Use certainty as curiosity proxy
      stability: stabilitySnapshot?.score || 0.5, // Use stability score
    };

    this.short.record(snapshot);
    this.medium.record(snapshot);
    this.long.record(snapshot);
  }

  summarize() {
    const shortDelta = this.short.getDelta();
    const mediumDelta = this.medium.getDelta();
    const longDelta = this.long.getDelta();

    return {
      short: shortDelta,
      medium: mediumDelta,
      long: longDelta,
    };
  }
}

export const PRIME_TEMPORAL = new TemporalReasoner();

