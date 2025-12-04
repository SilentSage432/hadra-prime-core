// src/cognition/bias/bias_filter.ts

export interface ReasoningSample {
  meaning: any;
  intent: any;
  context: any;
  confidence: number; // 0–1
}

export class BiasFilter {
  static detectAnchoring(sample: ReasoningSample): boolean {
    // simple heuristic for now
    return sample.confidence > 0.8 && !sample.context?.verified;
  }

  static detectOverconfidence(sample: ReasoningSample): boolean {
    return sample.confidence > 0.9 && !sample.intent?.evidence;
  }

  static detectHallucination(sample: ReasoningSample): boolean {
    return (
      sample.meaning &&
      typeof sample.meaning === "string" &&
      sample.meaning.includes("[INFERRED]") // placeholder until ML layer
    );
  }

  static detectContradiction(sample: ReasoningSample): boolean {
    if (!sample.context || !sample.meaning) return false;

    return (
      sample.context.topic &&
      sample.meaning.topic &&
      sample.context.topic !== sample.meaning.topic
    );
  }

  static computeIntegrity(sample: ReasoningSample): number {
    let score = 1;
    if (this.detectAnchoring(sample)) score -= 0.2;
    if (this.detectOverconfidence(sample)) score -= 0.3;
    if (this.detectHallucination(sample)) score -= 0.3;
    if (this.detectContradiction(sample)) score -= 0.4;
    return Math.max(0, score);
  }
}

