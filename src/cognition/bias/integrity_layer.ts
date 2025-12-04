// src/cognition/bias/integrity_layer.ts

import { BiasFilter } from "./bias_filter.ts";

export class ReasoningIntegrity {
  static evaluate(fusion: any) {
    const sample = {
      meaning: fusion.meaning,
      intent: fusion.intent,
      context: fusion.context,
      confidence: fusion.intent?.confidence ?? 0.5,
    };

    const integrity = BiasFilter.computeIntegrity(sample);

    return {
      integrity,
      flags: {
        anchoring: BiasFilter.detectAnchoring(sample),
        overconfidence: BiasFilter.detectOverconfidence(sample),
        hallucination: BiasFilter.detectHallucination(sample),
        contradiction: BiasFilter.detectContradiction(sample),
      },
    };
  }

  static correct(fusion: any, integrity: number) {
    if (integrity === 1) return fusion;

    // reduce confidence
    if (fusion.intent) {
      fusion.intent.confidence *= integrity;
    }

    // clamp meaning if too inconsistent
    if (integrity < 0.4) {
      fusion.meaning = "[DEGRADED — low integrity]";
    }

    return fusion;
  }
}

