// src/cognition/stability/cognitive_stabilizer.ts

import { StabilityMatrix } from "../../stability/stability_matrix.ts";

export interface FusionPayload {
  meaning: any;
  intent: any;
  context: any;
}

export class CognitiveStabilizer {
  static maxFusionSize = 5000;
  static maxLatency = 40; // ms
  static enabled = true;

  static stabilize(
    fusion: FusionPayload,
    latency: number
  ): FusionPayload | null {
    if (!this.enabled) return fusion;

    // 1. Too large? Trim the structure
    const fusionSize = JSON.stringify(fusion).length;
    if (fusionSize > this.maxFusionSize) {
      console.warn("[PRIME-FUSION] Oversized fusion payload trimmed.");
      fusion.meaning = "[TRIMMED]";
    }

    // 2. Latency too high? Drop non-critical elements
    if (latency > this.maxLatency) {
      console.warn("[PRIME-FUSION] Fusion latency high — reducing detail.");
      fusion.intent = { simplified: true, original: fusion.intent };
    }

    // 3. Ensure no contradictions
    if (fusion.meaning && fusion.context) {
      // Primitive example — will evolve
      if (fusion.meaning.topic && fusion.context.topic &&
          fusion.meaning.topic !== fusion.context.topic) {
        console.warn("[PRIME-FUSION] Context mismatch — smoothing meaning.");
        fusion.meaning.topic = fusion.context.topic;
      }
    }

    // 4. Check global system stability
    if (StabilityMatrix.unstable()) {
      console.warn("[PRIME-FUSION] Global instability — returning simplified fusion.");
      return {
        meaning: { degraded: true },
        intent: fusion.intent,
        context: fusion.context,
      };
    }

    return fusion;
  }
}

